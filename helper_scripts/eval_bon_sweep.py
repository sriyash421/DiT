"""Sharded, preemption-safe best-of-N sweep worker.

For each (checkpoint, split, prompt) unit assigned to this shard, generate N iid samples, score each with
the structured CLEVR distance metric (via an OpenRouter model, default glm-4.6v), and write one atomic
per-unit JSON with every sample's sub-metrics. Re-running skips units whose result file already exists,
so SLURM --requeue after preemption resumes cleanly.

Runs in .venv-omni (OmniGen generation + OpenRouter scoring). Collate with collate_bon_sweep.py."""
import argparse
import json
import os
from pathlib import Path

import hydra
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from omegaconf import OmegaConf

from algorithms.eval import select_eval_batch
from verifiers.base import score_structured_breakdown
from verifiers.open_router import OpenRouterVerifier

SPLIT_SEED_OFFSET = {"train": 0, "val": 10_000_000}
SUBMETRICS = ("combined", "content", "presence", "shape", "color", "extra")


def build_units(steps, splits, num_prompts):
    """Step-major unit list so contiguous shards mostly share one checkpoint (fewer 22GB reloads)."""
    return [(step, split, p) for step in steps for split in splits for p in range(num_prompts)]


def save_atomic(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def unit_done(path, num_samples):
    if not path.exists():
        return False
    try:
        return len(json.loads(path.read_text()).get("samples", [])) == num_samples
    except (json.JSONDecodeError, ValueError, OSError):
        return False


def generate_samples(model, caption, context_row, context_mask_row, args, base_seed):
    imgs = []
    for start in range(0, args.num_samples, args.batch_size):
        b = min(args.batch_size, args.num_samples - start)
        chunk = {"caption": [caption] * b}
        if context_row is not None:
            chunk["context_tokens"] = context_row.unsqueeze(0).repeat(b, 1, 1)
            chunk["context_mask"] = context_mask_row.unsqueeze(0).repeat(b, 1)
        imgs.extend(model.generate(
            chunk, num_sampling_steps=args.num_sampling_steps, cfg_scale=args.cfg_scale,
            ddim_eta=args.ddim_eta, seed=base_seed + start // args.batch_size,
        ))
    return imgs


def score_samples(scorer, caption, images):
    results = scorer.score_distance([caption] * len(images), [None] * len(images), images)
    samples = []
    for s, r in enumerate(results):
        row = {"sample": s, "ok": bool(r.ok and r.score is not None)}
        breakdown = score_structured_breakdown(r.feedback, caption) if r.ok else None
        for key in SUBMETRICS:
            row[key] = (breakdown[key] if breakdown is not None else None)
        samples.append(row)
    return samples


def main():
    args = parse_args()
    device = 0
    torch.cuda.set_device(device)

    cfg = OmegaConf.load(Path(args.run_dir) / "config.yaml")
    ckpt_dir = Path(args.run_dir) / "checkpoints"
    out_root = Path(args.out_dir) / "results"
    out_root.mkdir(parents=True, exist_ok=True)

    units = build_units(args.steps, args.splits, args.num_prompts)
    per = (len(units) + args.num_shards - 1) // args.num_shards
    my_units = units[args.shard * per:(args.shard + 1) * per]
    print(f"shard {args.shard}/{args.num_shards}: {len(my_units)} of {len(units)} units", flush=True)

    prompt_cache = {}

    def get_prompts(split):
        if split not in prompt_cache:
            ds = hydra.utils.instantiate(cfg.dataset, split=split)
            prompt_cache[split] = (ds, select_eval_batch(ds, args.prompt_seed, args.num_prompts))
        return prompt_cache[split]

    ds0, _ = get_prompts(my_units[0][1])
    model = hydra.utils.instantiate(cfg.model, context_dim=ds0.context_dim, device=device)
    scorer = OpenRouterVerifier(model=args.scorer_model, temperature=0.0, max_tokens=1024,
                                workers=args.score_workers, timeout=120)

    loaded_step = None
    done = 0
    for step, split, pidx in my_units:
        out_file = out_root / f"{step}_{split}_{pidx:04d}.json"
        if unit_done(out_file, args.num_samples):
            done += 1
            continue
        if loaded_step != step:
            model.load(str(ckpt_dir / f"{step:07d}.pt"), use_ema=args.use_ema)
            model.net.eval()
            loaded_step = step
            print(f"loaded checkpoint {step}", flush=True)

        _, batch = get_prompts(split)
        caption = batch["caption"][pidx]
        ctx = batch["context_tokens"][pidx] if batch["context_tokens"] is not None else None
        ctx_mask = batch["context_mask"][pidx] if batch["context_mask"] is not None else None
        base_seed = args.sample_seed + SPLIT_SEED_OFFSET[split] + pidx * args.num_samples

        images = generate_samples(model, caption, ctx, ctx_mask, args, base_seed)
        samples = score_samples(scorer, caption, images)
        save_atomic(out_file, {
            "step": step, "split": split, "prompt_index": pidx, "caption": caption,
            "scorer_model": args.scorer_model, "num_samples": args.num_samples, "samples": samples,
        })
        done += 1
        n_ok = sum(s["ok"] for s in samples)
        print(f"[{done}/{len(my_units)}] {step} {split} p{pidx}: {n_ok}/{args.num_samples} scored"
              f"  cost=${getattr(scorer, 'session_cost', 0.0):.3f}", flush=True)

    print(f"shard {args.shard} DONE: {done}/{len(my_units)} units, "
          f"session_cost=${getattr(scorer, 'session_cost', 0.0):.3f}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--out_dir", required=True, help="Sweep root; per-unit JSON goes to <out_dir>/results/.")
    p.add_argument("--steps", type=int, nargs="+", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--num_prompts", type=int, default=256)
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--num_shards", type=int, default=64)
    p.add_argument("--scorer_model", default="z-ai/glm-4.6v")
    p.add_argument("--score_workers", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_sampling_steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--ddim_eta", type=float, default=0.0)
    p.add_argument("--prompt_seed", type=int, default=0)
    p.add_argument("--sample_seed", type=int, default=1234)
    p.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


if __name__ == "__main__":
    main()
