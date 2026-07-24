"""Best-of-N generation eval: for each prompt, iid-generate N samples, score each, and report the
running-best score over the first k samples (in generation order, NOT expected best-of-n). Dumps a
per-sample CSV, a curve CSV/plot, and every generated image (organized by prompt/sample) so a later
LLM-as-judge can compare models.

All scorers are higher-is-better, so "best" is always the running max. Reuses the training run's saved
config.yaml to rebuild the model/dataset (same as eval_generation.py)."""
import argparse
import csv
from pathlib import Path

import hydra
import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from tqdm import tqdm

from omegaconf import OmegaConf

from algorithms.eval import select_eval_batch
from eval_generation import resolve_run
from verifiers.eval_metrics import score, scorer_from_eval_cfg


def generate_all(model, batch, tasks, args):
    """Generate one image per (prompt_idx, sample_idx) task; returns predictions aligned to `tasks`."""
    ctx = batch["context_tokens"]
    predictions = []
    for start in tqdm(range(0, len(tasks), args.batch_size), desc="generate", unit="batch"):
        chunk_tasks = tasks[start:start + args.batch_size]
        chunk = {"caption": [batch["caption"][p] for p, _ in chunk_tasks]}
        if ctx is not None:
            rows = [p for p, _ in chunk_tasks]
            chunk["context_tokens"] = ctx[rows]
            chunk["context_mask"] = batch["context_mask"][rows]
        predictions.extend(
            model.generate(
                chunk,
                num_sampling_steps=args.num_sampling_steps,
                cfg_scale=args.cfg_scale,
                ddim_eta=args.ddim_eta,
                seed=args.sample_seed + start,
            )
        )
    return predictions


def running_best(scores):
    """Running max over the first k scores (generation order), skipping None. best[k-1] = best of first k."""
    out, best = [], None
    for value in scores:
        if value is not None and (best is None or value > best):
            best = value
        out.append(best)
    return out


def main():
    args = parse_args()
    device = 0
    torch.cuda.set_device(device)

    train_cfg, ckpt = resolve_run(args.run_dir, args.step)
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path("results") / f"bon_{Path(args.run_dir).name}_ckpt{Path(ckpt).stem}"
    )
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    print(f"Saving results to {out_dir}")

    build_dataset = lambda split: hydra.utils.instantiate(train_cfg.dataset, split=split)
    dataset = build_dataset(args.split)
    model = hydra.utils.instantiate(train_cfg.model, context_dim=dataset.context_dim, device=device)
    model.load(ckpt, use_ema=args.use_ema)
    model.net.eval()
    scorer = scorer_from_eval_cfg(OmegaConf.select(train_cfg, "trainer.eval"))

    batch = select_eval_batch(dataset, args.seed, args.num_prompts)
    tasks = [(p, s) for p in range(args.num_prompts) for s in range(args.max_samples)]
    predictions = generate_all(model, batch, tasks, args)

    captions_flat = [batch["caption"][p] for p, _ in tasks]
    gt_flat = [batch["gt_images"][p] for p, _ in tasks]
    scores_flat = score(scorer, captions_flat, gt_flat, predictions)

    # Save all images (+ one GT per prompt) and lay scores/images out per prompt.
    per_prompt = [[] for _ in range(args.num_prompts)]
    image_paths = {}
    for (p, s), pred in zip(tasks, predictions):
        prompt_dir = out_dir / "images" / f"prompt_{p:03d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        if s == 0:
            batch["gt_images"][p].save(prompt_dir / "gt.png")
        path = prompt_dir / f"sample_{s:03d}.png"
        pred.save(path)
        image_paths[(p, s)] = str(path)
    for (p, s), value in zip(tasks, scores_flat):
        per_prompt[p].append((s, value))

    # Per-sample CSV with the running-best up to each sample.
    with open(out_dir / "bon_samples.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["prompt_id", "caption", "sample_index", "seed", "score", "running_best", "image_path"])
        prompt_best = {}
        for p in range(args.num_prompts):
            samples = sorted(per_prompt[p])
            best = running_best([v for _, v in samples])
            prompt_best[p] = best
            for (s, value), rb in zip(samples, best):
                writer.writerow([
                    p, batch["caption"][p], s, args.sample_seed + s,
                    "" if value is None else f"{value:.6f}",
                    "" if rb is None else f"{rb:.6f}", image_paths[(p, s)],
                ])

    # Curve: mean over prompts of running-best@k (skip prompts with no scored sample yet at k).
    ks, mean_best, std_best = [], [], []
    for k in range(1, args.max_samples + 1):
        column = [prompt_best[p][k - 1] for p in range(args.num_prompts) if prompt_best[p][k - 1] is not None]
        if not column:
            continue
        mean = sum(column) / len(column)
        var = sum((v - mean) ** 2 for v in column) / len(column)
        ks.append(k)
        mean_best.append(mean)
        std_best.append(var ** 0.5)

    with open(out_dir / "bon_curve.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["k", "mean_best", "std_best"])
        for k, mean, std in zip(ks, mean_best, std_best):
            writer.writerow([k, f"{mean:.6f}", f"{std:.6f}"])

    plt.figure(figsize=(6, 4))
    plt.plot(ks, mean_best, marker="o")
    plt.fill_between(ks, [m - s for m, s in zip(mean_best, std_best)],
                     [m + s for m, s in zip(mean_best, std_best)], alpha=0.2)
    plt.xlabel("k (samples drawn)")
    plt.ylabel("best score of first k (mean over prompts)")
    plt.title(f"Best-of-N — {Path(args.run_dir).name} @ {Path(ckpt).stem}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "bon_curve.png", dpi=150)
    print(f"Wrote bon_curve.png, bon_curve.csv, bon_samples.csv and {len(tasks)} images to {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True, help="Training run dir (holds config.yaml and checkpoints/).")
    p.add_argument("--step", type=int, default=-1, help="Checkpoint step to load; -1 for the latest.")
    p.add_argument("--out_dir", default=None, help="Output dir; defaults to results/bon_<run>_ckpt<step>/.")
    p.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--split", default="val")
    p.add_argument("--num_prompts", type=int, default=50)
    p.add_argument("--max_samples", type=int, default=8, help="N: iid samples generated per prompt.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0, help="Seed for selecting the prompts.")
    p.add_argument("--sample_seed", type=int, default=1234, help="Base seed for the sampling noise.")
    p.add_argument("--num_sampling_steps", type=int, default=100)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--ddim_eta", type=float, default=0.0)
    return p.parse_args()


if __name__ == "__main__":
    main()
