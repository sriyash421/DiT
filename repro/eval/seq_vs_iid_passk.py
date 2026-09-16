"""Sequential vs iid pass@k on clevr_g6 held-out captions, cell-aware.

Two sampling regimes over the SAME prompts and the SAME sample budget N:

  sequential   one chain per prompt. x_T is drawn once and held FIXED for all N attempts, so the
               only thing that changes between attempt t and t+1 is the verifier's critique. This
               isolates "did the model use the feedback" from "did it get a luckier noise draw".
  iid          N independent samples, fresh noise each time, no feedback. This is best-of-N.

Both are graded by the cell-aware ClevrDetectorVerifier, so an object of the right colour and shape
in the wrong cell counts as wrong (35% of this model's held-out failures are pure placement).

Per prompt we store the per-attempt score and per-attempt exact flag for both regimes, which is
enough to compute downstream, WITHOUT resampling:
  sequential pass@k  = any of the first k chain attempts exact
  iid pass@k         = unbiased 1 - C(n-c,k)/C(n,k)  (and max@N / avg@N)
  per-attempt panel  = mean score of attempt t alone

Compute asymmetry, for the record: sequential-N is N serial generations + (N-1) verifier calls;
iid-N is N generations and no verifier. "Equal N" is equal samples, not equal cost.
"""
import argparse, json, os, sys, time
from pathlib import Path

# Which checkout to import the model code from. The curriculum run's config references ctor args
# (lora_dropout) that only exist in the worktree copy, so evaluating that checkpoint with the main
# checkout's code would die on an unexpected kwarg. Defaults to the main repo for every other run.
sys.path.insert(0, os.environ.get("REPO_DIR", "/mmfs1/gscratch/socialrl/sriyash/DiT"))
import hydra
import torch
from omegaconf import OmegaConf

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from algorithms.eval import select_eval_batch


def derange(n, rng):
    """A permutation with no fixed point, so no row can receive its own critique.

    Falls back to the identity for n < 2, where a derangement does not exist; callers should not
    rely on shuffling a one-row batch.
    """
    if n < 2:
        return list(range(n))
    while True:
        perm = list(range(n))
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            return perm
from verifiers.detector_clevr import ClevrDetectorVerifier

PROBE = "/gscratch/scrubbed/sriyash/models/clevr_shape_probe_g6.pt"
CELLS = [[37, 194], [110, 194], [182, 194], [74, 58], [146, 58], [219, 58]]


def build_verifier():
    v = ClevrDetectorVerifier(device="cuda", use_owl=False, probe_path=PROBE,
                              cell_centres=CELLS, cell_tol=30.0)
    v._lazy_init()
    return v


def select_eval_batch_excluding(dataset, seed, count, exclude_seed, exclude_n):
    """select_eval_batch, but drawing from rows the reported eval never touches.

    Mirrors the repo helper's sampling so the excluded set is exactly the reported one.
    """
    import random as _random

    from algorithms.eval import pad_contexts

    excluded = set(_random.Random(exclude_seed).sample(range(len(dataset)), exclude_n))
    pool = [i for i in range(len(dataset)) if i not in excluded]
    if len(pool) < count:
        raise RuntimeError(f"only {len(pool)} rows left after excluding {len(excluded)}")
    indices = _random.Random(seed).sample(pool, count)
    contexts, captions, gt_images = [], [], []
    for idx in indices:
        item = dataset[idx]
        if dataset.context_dim is not None:
            contexts.append(item["context_tokens"].float())
        captions.append(item["caption"])
        gt_images.append(dataset.image_for_index(idx).convert("RGB"))
    tokens, mask = pad_contexts(contexts) if contexts else (None, None)
    return {"indices": indices, "context_tokens": tokens, "context_mask": mask,
            "caption": captions, "gt_images": gt_images}


BREAKDOWN_KEYS = ("score", "presence", "shape", "color", "quality", "precision")


def grade(verifier, captions, images, histories=None):
    """-> (scores, exact_flags, feedbacks, breakdowns).

    Runs the detector ONCE per image and derives the critique and the fine-grained breakdown from
    the same detection, rather than calling verify() and then detecting again. `breakdowns` is one
    dict per image with keys BREAKDOWN_KEYS, so the report can plot the component metrics
    (presence / shape / colour / quality / precision) and not just the aggregate.

    Grading is caption-only -- the detector never sees the ground-truth image or the history.
    """
    from verifiers.base import enumeration_breakdown, enumeration_feedback

    scores, exact, feedback, breakdowns = [], [], [], []
    for caption, img in zip(captions, images):
        try:
            seen = verifier.detect(img)
            bd = enumeration_breakdown(caption, seen)
            cmd, _rule, _ = enumeration_feedback(caption, seen, max_edits=verifier.max_edits)
        except Exception:                       # a detector failure must not kill the sweep
            seen, bd, cmd = [], None, ""
        bd = bd or {k: 0.0 for k in BREAKDOWN_KEYS}
        scores.append(float(bd["score"]))
        feedback.append("" if cmd is None else str(cmd))
        exact.append(str(cmd).strip().lower().rstrip(".") == "no update")
        breakdowns.append({k: float(bd.get(k, 0.0)) for k in BREAKDOWN_KEYS})
    return scores, exact, feedback, breakdowns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="experiment dir containing config.yaml")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--num_prompts", type=int, default=200)
    ap.add_argument("--n", type=int, default=8, help="sample budget N for both regimes")
    ap.add_argument("--prompt_seed", type=int, default=0)
    ap.add_argument("--sample_seed", type=int, default=1234)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_sampling_steps", type=int, default=25)
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument("--history_window", type=int, default=3,
                    help="keep only the last W (attempt, critique) pairs in context. 3 == the most "
                         "the model ever saw in training (rollout.length 4), so every chain step "
                         "stays in-distribution no matter how long the chain runs. 0 = accumulate "
                         "everything, which is what training does but only ever up to 3.")
    ap.add_argument("--save_traces", type=int, default=0,
                    help="save the full chain (every attempt image + its critique) for the first N "
                         "prompts, so the mechanism can be shown rather than only summarised")
    ap.add_argument("--trace_dir", default=None)
    ap.add_argument("--shuffle_feedback", action="store_true",
                    help="PLACEBO CONTROL: condition each chain on another caption's critique "
                         "while keeping its own attempt image and its own grading. Isolates the "
                         "critique's content from the mere presence of a previous image.")
    ap.add_argument("--constant_feedback", default=None,
                    help="COMPLIANCE TEST: replace every critique with this fixed string (e.g. "
                         "'no update'). Unlike --shuffle_feedback this is not a neutral placebo -- "
                         "'no update' is an explicit 'your image is correct, hold it' instruction, "
                         "so it measures whether the model obeys a hold command rather than "
                         "whether critique content carries information.")
    ap.add_argument("--regimes", default="seq,iid")
    ap.add_argument("--tag", default="")
    ap.add_argument("--exclude_n", type=int, default=0,
                    help="draw prompts from the test split MINUS this many rows -- used to pick a "
                         "checkpoint on data disjoint from the reported evaluation set")
    ap.add_argument("--exclude_seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.cuda.set_device(0)
    cfg = OmegaConf.load(Path(args.run_dir) / "config.yaml")
    ds = hydra.utils.instantiate(cfg.dataset, split=args.split)
    if args.exclude_n:
        batch = select_eval_batch_excluding(ds, args.prompt_seed, args.num_prompts,
                                            args.exclude_seed, args.exclude_n)
        print(f"validation prompts: {args.num_prompts} drawn disjoint from the "
              f"{args.exclude_n} reported captions", flush=True)
    else:
        batch = select_eval_batch(ds, args.prompt_seed, args.num_prompts)
    model = hydra.utils.instantiate(cfg.model, context_dim=ds.context_dim, device=0)
    model.load(str(Path(args.run_dir) / "checkpoints" / f"{args.step:07d}.pt"), use_ema=False)
    model.net.eval()
    verifier = build_verifier()
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    W = int(args.history_window)
    print(f"history window: {W or 'unbounded (accumulate)'}", flush=True)

    def gen(caps, seed, feedback_history=None, attempt_images=None, init_latents=None):
        ctx = {"caption": list(caps),
               "feedback_history": feedback_history or [[] for _ in caps],
               "attempt_images": attempt_images or [[] for _ in caps],
               "attempt_paths": [[] for _ in caps]}
        return model.generate(ctx, num_sampling_steps=args.num_sampling_steps,
                              cfg_scale=args.cfg_scale, ddim_eta=0.0, seed=seed,
                              return_latents=True, init_latents=init_latents)

    out = {"tag": args.tag, "step": args.step, "split": args.split, "n": args.n,
           "shuffle_feedback": bool(args.shuffle_feedback),
           "constant_feedback": args.constant_feedback,
           "history_window": W, "run_dir": args.run_dir, "prompts": [], "seq": [], "iid": []}

    # Trace capture: the GT plus every attempt image and the critique computed from it. Written
    # alongside the scores so a figure can show the actual repair happening.
    traces = []
    trace_dir = Path(args.trace_dir) if args.trace_dir else None
    if args.save_traces and trace_dir:
        (trace_dir / "img").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, args.num_prompts, args.batch_size):
            stop = min(start + args.batch_size, args.num_prompts)
            caps = [batch["caption"][i] for i in range(start, stop)]
            B = len(caps)
            out["prompts"].extend(caps)

            if "seq" in regimes:
                # ---- sequential: one chain, noise fixed for the whole chain ------------------
                hist = [[] for _ in range(B)]
                imgs_hist = [[] for _ in range(B)]
                chain_noise = None
                rows = [{"score": [], "exact": [], "feedback": [], "bd": []} for _ in range(B)]
                # which rows of this batch are being traced, and where their GT was written
                trace_rows = {}
                if args.save_traces and trace_dir:
                    for i in range(B):
                        gidx = start + i
                        if gidx >= args.save_traces:
                            continue
                        gt_path = trace_dir / "img" / f"{gidx:04d}_gt.png"
                        batch["gt_images"][gidx].save(gt_path)
                        trace_rows[i] = {"index": gidx, "caption": caps[i],
                                         "gt_path": str(gt_path), "attempts": []}
                for t in range(args.n):
                    imgs, lat = gen(caps, args.sample_seed + start,
                                    feedback_history=hist, attempt_images=imgs_hist,
                                    init_latents=chain_noise)
                    if chain_noise is None:
                        # generate() returns a list of per-item CPU tensors; init_latents wants a
                        # batched tensor.
                        chain_noise = torch.stack([t.detach() for t in lat])
                    sc, ex, fb, bds = grade(verifier, caps, imgs, hist)
                    # Identity unless the placebo control is on: perm[i] picks whose critique
                    # conditions row i's NEXT attempt. Grading above is already done on row i's
                    # own image, so only the conditioning text moves.
                    perm = (derange(B, random.Random(args.sample_seed + 7919 * t + start))
                            if args.shuffle_feedback else list(range(B)))
                    for i in range(B):
                        if i in trace_rows:
                            p = trace_dir / "img" / f"{trace_rows[i]['index']:04d}_a{t}.png"
                            imgs[i].save(p)
                            # the critique stored here is the one computed FROM this attempt, i.e.
                            # the text that conditions the next one
                            trace_rows[i]["attempts"].append(
                                {"path": str(p), "feedback": fb[i],
                                 "score": sc[i], "exact": bool(ex[i])})
                        rows[i]["score"].append(sc[i])
                        rows[i]["exact"].append(ex[i])
                        rows[i]["feedback"].append(fb[i])
                        rows[i]["bd"].append(bds[i])
                        # Accumulate the FULL history, exactly as OnPolicyTrainer.collect does
                        # (it appends to attempt_image_history each step). Conditioning on only
                        # the latest attempt here would grade the model under a context it was
                        # never trained on. Empty feedback is appended raw -- history_instruction
                        # turns it into the explicit "no change" string.
                        # Chains are never cut short on success: the model must HOLD a correct
                        # image, not just repair a wrong one.
                        hist[i].append(args.constant_feedback if args.constant_feedback is not None
                                       else fb[perm[i]])
                        imgs_hist[i].append(imgs[i])
                        if W:
                            # Sliding window. For t <= W this is a no-op, so the in-distribution
                            # part of the curve is bit-identical to full accumulation; past it,
                            # the context stays the shape training used instead of growing into a
                            # length the model has never seen.
                            hist[i] = hist[i][-W:]
                            imgs_hist[i] = imgs_hist[i][-W:]
                out["seq"].extend(rows)
                traces.extend(trace_rows[i] for i in sorted(trace_rows))

            if "iid" in regimes:
                # ---- iid best-of-N: fresh noise, no feedback ---------------------------------
                rows = [{"score": [], "exact": [], "bd": []} for _ in range(B)]
                for t in range(args.n):
                    imgs, _ = gen(caps, args.sample_seed + 100000 * (t + 1) + start)
                    sc, ex, _, bds = grade(verifier, caps, imgs)
                    for i in range(B):
                        rows[i]["score"].append(sc[i])
                        rows[i]["exact"].append(ex[i])
                        rows[i]["bd"].append(bds[i])
                out["iid"].extend(rows)

            print(f"  {stop}/{args.num_prompts}  ({time.time()-t0:.0f}s)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    if traces and trace_dir:
        (trace_dir / "traces.json").write_text(json.dumps(traces, indent=1))
        print(f"saved {len(traces)} traces -> {trace_dir / 'traces.json'}")
    for key in regimes:
        rows = out[key]
        if not rows:
            continue
        print(f"\n=== {args.tag} [{key}] n={len(rows)} ===")
        for k in range(1, args.n + 1):
            if key == "seq":
                p = sum(1 for r in rows if any(r["exact"][:k])) / len(rows)
                label = "sequential pass@"
            else:
                # unbiased 1 - C(n-c,k)/C(n,k)
                from math import comb
                tot = 0.0
                for r in rows:
                    c = sum(r["exact"]); n = len(r["exact"])
                    tot += 1.0 - (comb(n - c, k) / comb(n, k) if n - c >= k else 0.0)
                p = tot / len(rows)
                label = "iid pass@"
            print(f"  {label}{k}: {p:.4f}")
        print("  per-attempt mean score:",
              " ".join(f"{sum(r['score'][t] for r in rows)/len(rows):.3f}" for t in range(args.n)))
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
