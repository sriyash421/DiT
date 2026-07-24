"""Collate the per-checkpoint CompBench-complex generation evals into three figures.

Reads results/eval_generation/compbench_complex/step_<step>/{train,val}_results.json (+ the saved
{split}_gt_pred.png grids) that `job_eval_generation.sh` writes, then produces:

  1. metric_curve.png     - x=checkpoint step, y=mean metric, TWO curves (train + val) on one axes.
  2. samples_train.png    - one row per checkpoint, each row = 4 GT|Pred pairs (fixed samples, so
  3. samples_val.png        rows are directly comparable down the column as training progresses).

The eval batch is picked with a fixed seed and the same num_samples for every checkpoint, so the
first N samples (and their GTs) are identical across steps -> cropping them straight out of each
saved grid keeps everything aligned without reloading the model or the dataset.
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# Cell geometry from algorithms.eval.save_gt_pred_grid (captions ARE passed by eval_generation.py).
TILE = 128
LABEL_H = 24
GAP = 12
COLS = 4
CAPTION_H = 3 * 12 + 6           # caption_lines * 12 + 6
CELL_H = TILE + LABEL_H + CAPTION_H
PAIR_W = TILE * 2


def crop_pairs(grid_path, n):
    """Crop the first `n` (GT, Pred) 128x128 tiles out of a saved gt_pred grid PNG."""
    img = Image.open(grid_path).convert("RGB")
    pairs = []
    for idx in range(n):
        row, col = idx // COLS, idx % COLS
        x = col * (PAIR_W + GAP)
        y = row * (CELL_H + GAP)
        gt = img.crop((x, y + LABEL_H, x + TILE, y + LABEL_H + TILE))
        pred = img.crop((x + TILE, y + LABEL_H, x + PAIR_W, y + LABEL_H + TILE))
        pairs.append((gt, pred))
    return pairs


def load_runs(root):
    """Return {step: {split: results_dict}} for every step_<n> dir that has result JSONs."""
    runs = {}
    for d in sorted(root.glob("step_*")):
        try:
            step = int(d.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        for split in ("train", "val"):
            res = d / f"{split}_results.json"
            if res.exists():
                runs.setdefault(step, {})[split] = json.loads(res.read_text())
    return runs


def mean_stderr(res):
    """Mean and standard error of the mean over the per-sample distances in a results dict."""
    vals = [d for d in res.get("distances", []) if d is not None]
    if not vals:
        return None, 0.0
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var) / math.sqrt(len(vals))


def plot_metric_curve(runs, out_path):
    steps = sorted(runs)
    fig, ax = plt.subplots(figsize=(7, 5))
    for split, color in (("train", "tab:blue"), ("val", "tab:orange")):
        xs, ys, errs = [], [], []
        for s in steps:
            res = runs[s].get(split)
            if not res:
                continue
            m, se = mean_stderr(res)
            if m is None:
                continue
            xs.append(s); ys.append(m); errs.append(se)
        if xs:
            ax.errorbar(xs, ys, yerr=errs, marker="o", color=color, label=split,
                        capsize=4, elinewidth=1.2)
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("mean metric ± SE (higher = better)")
    ax.set_title("CompBench-complex: eval metric vs. training step")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_sample_grid(runs, split, n_samples, out_path):
    steps = [s for s in sorted(runs) if split in runs[s] and Path(runs[s][split]["grid"]).exists()]
    if not steps:
        print(f"[{split}] no grids found, skipping sample figure")
        return
    label_w = 132                       # left gutter for the "step / metric" label
    pad = 8
    cell = TILE
    # Each sample is a GT|Pred pair; n_samples pairs across, one checkpoint per row.
    row_h = cell + pad
    row_w = label_w + n_samples * (2 * cell + pad)
    header_h = 22
    canvas = Image.new("RGB", (row_w, header_h + len(steps) * row_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for j in range(n_samples):
        gx = label_w + j * (2 * cell + pad)
        draw.text((gx + 4, 6), f"sample {j}  (GT | Pred)", fill=(20, 20, 20))
    for i, step in enumerate(steps):
        y = header_h + i * row_h
        metric = runs[step][split].get("mean_distance")
        mtxt = f"{metric:.4f}" if metric is not None else "n/a"
        draw.text((6, y + cell // 2 - 12), f"step {step}", fill=(0, 0, 0))
        draw.text((6, y + cell // 2 + 2), f"metric {mtxt}", fill=(60, 60, 60))
        pairs = crop_pairs(runs[step][split]["grid"], n_samples)
        for j, (gt, pred) in enumerate(pairs):
            gx = label_w + j * (2 * cell + pad)
            canvas.paste(gt, (gx, y))
            canvas.paste(pred, (gx + cell, y))
            draw.rectangle((gx, y, gx + cell - 1, y + cell - 1), outline=(0, 0, 0), width=3)
            draw.rectangle((gx + cell, y, gx + 2 * cell - 1, y + cell - 1), outline=(0, 0, 0))
    canvas.save(out_path)
    print(f"wrote {out_path}  ({len(steps)} checkpoints x {n_samples} samples)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="results/eval_generation/compbench_complex",
                   help="Dir holding the step_<n>/ eval outputs.")
    p.add_argument("--n_samples", type=int, default=4, help="GT|Pred pairs per checkpoint row.")
    args = p.parse_args()
    root = Path(args.root)
    runs = load_runs(root)
    if not runs:
        raise SystemExit(f"No step_*/*_results.json found under {root}")
    print(f"found evals for steps: {sorted(runs)}")
    out = root / "plots"
    out.mkdir(exist_ok=True)
    plot_metric_curve(runs, out / "metric_curve.png")
    plot_sample_grid(runs, "train", args.n_samples, out / "samples_train.png")
    plot_sample_grid(runs, "val", args.n_samples, out / "samples_val.png")


if __name__ == "__main__":
    main()
