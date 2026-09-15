"""Collate the per-unit BoN sweep results into best-of-N curves.

Per metric (combined + sub-metrics), per checkpoint & split: per prompt take the running-best over the
first k samples (higher=better -> running max), then average over prompts. Outputs, all with mean +/- SE:
- curves/: per-metric linear-k and log-k plots (one line per checkpoint), + bon_curves.csv
- train_val_compare/: for the latest checkpoint, a grid comparing train vs val across all metrics."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUBMETRICS = ("score", "presence", "shape", "color", "quality", "precision")
POW2 = [1, 2, 4, 8, 16, 32, 64]


def running_best(values):
    out, best = [], None
    for v in values:
        if v is not None and (best is None or v > best):
            best = v
        out.append(best)
    return out


def _series(curve_sd, log):
    ks, means, stds, ns = curve_sd
    if log:
        pts = [(k, m, s, nn) for k, m, s, nn in zip(ks, means, stds, ns) if k in POW2]
        if not pts:
            return [], [], []
        ks, means, stds, ns = map(list, zip(*pts))
    sem = [s / (nn ** 0.5) if nn else 0.0 for s, nn in zip(stds, ns)]
    return list(ks), list(means), sem


def _log_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(POW2)
    ax.set_xticklabels(POW2)


def plot_metric(metric, curve, steps, splits, curves_dir, log):
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 4.5), squeeze=False)
    for ax, split in zip(axes[0], splits):
        for step in steps:
            ks, means, sem = _series(curve[metric].get((step, split), ([], [], [], [])), log)
            if not ks:
                continue
            line, = ax.plot(ks, means, marker="o", markersize=4, label=f"{step // 1000}k")
            ax.fill_between(ks, [m - e for m, e in zip(means, sem)],
                            [m + e for m, e in zip(means, sem)], alpha=0.2, color=line.get_color())
        if log:
            _log_axis(ax)
        ax.set_title(f"{metric}  ({split})")
        ax.set_xlabel("k (samples drawn)" + (" [log2]" if log else ""))
        ax.set_ylabel(f"best {metric} of first k (mean +/- SE)")
        ax.grid(True, alpha=0.3)
        ax.legend(title="checkpoint")
    fig.suptitle(f"Best-of-N - {metric}" + (" (log-k)" if log else ""))
    fig.tight_layout()
    fig.savefig(curves_dir / f"{'bon_log' if log else 'bon'}_{metric}.png", dpi=150)
    plt.close(fig)


def plot_train_val(curve, step, splits, folder, log):
    """Grid comparing train vs val across all metrics for one checkpoint."""
    folder.mkdir(parents=True, exist_ok=True)
    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.2 * nrows), squeeze=False)
    for idx, metric in enumerate(SUBMETRICS):
        ax = axes[idx // ncols][idx % ncols]
        for split in splits:
            ks, means, sem = _series(curve[metric].get((step, split), ([], [], [], [])), log)
            if not ks:
                continue
            line, = ax.plot(ks, means, marker="o", markersize=4, label=split)
            ax.fill_between(ks, [m - e for m, e in zip(means, sem)],
                            [m + e for m, e in zip(means, sem)], alpha=0.2, color=line.get_color())
        if log:
            _log_axis(ax)
        ax.set_title(metric)
        ax.set_xlabel("k" + (" [log2]" if log else ""))
        ax.set_ylabel("best of first k (mean +/- SE)")
        ax.grid(True, alpha=0.3)
        ax.legend(title="split")
    fig.suptitle(f"train vs val @ {step // 1000}k checkpoint (mean +/- SE)" + (" (log-k)" if log else ""))
    fig.tight_layout()
    fig.savefig(folder / f"train_val_{step // 1000}k{'_log' if log else ''}.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    results = sorted((out_dir / "results").glob("*.json"))
    if not results:
        raise SystemExit(f"No per-unit results in {out_dir/'results'}")

    groups = defaultdict(dict)
    num_samples = 0
    for path in results:
        r = json.loads(path.read_text())
        samples = sorted(r["samples"], key=lambda s: s["sample"])
        groups[(r["step"], r["split"])][r["prompt_index"]] = samples
        num_samples = max(num_samples, len(samples))

    steps = sorted({s for s, _ in groups})
    splits = sorted({sp for _, sp in groups})
    curves_dir = out_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    curve = defaultdict(dict)
    for metric in SUBMETRICS:
        for (step, split), prompts in groups.items():
            per_prompt_best = [running_best([s.get(metric) for s in samples]) for samples in prompts.values()]
            ks, means, stds, ns = [], [], [], []
            for k in range(1, num_samples + 1):
                col = [rb[k - 1] for rb in per_prompt_best if k <= len(rb) and rb[k - 1] is not None]
                if not col:
                    continue
                mean = sum(col) / len(col)
                std = (sum((v - mean) ** 2 for v in col) / len(col)) ** 0.5
                ks.append(k); means.append(mean); stds.append(std); ns.append(len(col))
                csv_rows.append([metric, step, split, k, f"{mean:.6f}", f"{std:.6f}",
                                 f"{std/(len(col)**0.5):.6f}", len(col)])
            curve[metric][(step, split)] = (ks, means, stds, ns)

    with open(out_dir / "bon_curves.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "step", "split", "k", "mean_best", "std_best", "std_err", "n_prompts"])
        w.writerows(csv_rows)

    for metric in SUBMETRICS:
        plot_metric(metric, curve, steps, splits, curves_dir, log=False)
        plot_metric(metric, curve, steps, splits, curves_dir, log=True)

    if steps and len(splits) > 1:
        latest = max(steps)  # 100k
        plot_train_val(curve, latest, splits, out_dir / "train_val_compare", log=False)
        plot_train_val(curve, latest, splits, out_dir / "train_val_compare", log=True)

    print(f"Collated {len(results)} units, {num_samples} samples, steps={steps}, splits={splits}")
    print(f"Wrote curves/ (+train_val_compare/ for {max(steps)//1000 if steps else '?'}k) and bon_curves.csv")


if __name__ == "__main__":
    main()
