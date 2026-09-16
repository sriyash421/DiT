"""The practical question, as a bar chart.

You have a budget of k generations for one caption. Three strategies:

  1. in-context model, chain, BEST attempt   -- run the feedback chain, keep the best image the
                                                verifier saw along the way
  2. in-context model, chain, FINAL attempt  -- run the chain, keep whatever it ended on
                                                (what you get if you cannot look back)
  3. converged base model, BEST of k         -- no feedback at all: train normally, draw k
                                                independent samples, let the verifier pick

(1) and (3) both use verifier selection, so they are the matched pair; (2) is what the chain gives
you without it. Everything is the verifier's own score -- no training loss anywhere.
"""
import argparse, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, "/mmfs1/gscratch/socialrl/sriyash/DiT/helper_scripts")
from plot_style import apply_house_style, legend_below, style_axes  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def best_score(rows, k):
    return np.array([max(r["score"][:k]) for r in rows])


def final_score(rows, k):
    return np.array([r["score"][k - 1] for r in rows])


def best_exact(rows, k):
    return np.array([1.0 if any(r["exact"][:k]) else 0.0 for r in rows])


def final_exact(rows, k):
    return np.array([1.0 if r["exact"][k - 1] else 0.0 for r in rows])


def mse(a):
    return a.mean(), a.std(ddof=1) / np.sqrt(len(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_context", required=True)
    ap.add_argument("--converged", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", default="1,2,4,8")
    a = ap.parse_args()

    ic = json.load(open(a.in_context))
    cv = json.load(open(a.converged))
    assert ic["prompts"] == cv["prompts"], "prompt sets differ"
    ks = [int(x) for x in a.ks.split(",")]

    bars = [
        ("in-context model — chain, best attempt", ic["seq"], best_score, best_exact, "#a8324e"),
        ("in-context model — chain, final attempt", ic["seq"], final_score, final_exact, "#e0a3b0"),
        ("converged base model — best of k independent", cv["iid"], best_score, best_exact, "#3a6ea5"),
    ]

    apply_house_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    width = 0.26
    x = np.arange(len(ks))

    for ax, which, ylab, title in (
            (axes[0], "score", "mean verifier score",
             "Average verifier score of the image you keep"),
            (axes[1], "exact", "fraction of captions exactly right",
             "How often that image is exactly right")):
        print(f"\n=== {title} ===")
        for j, (label, rows, fscore, fexact, colour) in enumerate(bars):
            f = fscore if which == "score" else fexact
            m = [mse(f(rows, k)) for k in ks]
            ax.bar(x + (j - 1) * width, [v[0] for v in m], width,
                   yerr=[v[1] for v in m], capsize=3, label=label, color=colour,
                   edgecolor="white", linewidth=0.6)
            print(f"  {label:<46s}" + "".join(f"{v[0]:>8.3f}" for v in m))
        ax.set_xticks(x); ax.set_xticklabels([f"k={k}" for k in ks])
        ax.set_xlabel("budget: generations spent on one caption")
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=11)
        style_axes(ax)

    legend_below(fig, handles_from=axes[0], ncol=1)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=190, bbox_inches="tight")
    print("\nsaved ->", a.out)


if __name__ == "__main__":
    main()
