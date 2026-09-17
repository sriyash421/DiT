"""Three on-policy training variants, same eval protocol.

Main panel : the verifier's overall SCORE of the best image within k attempts.
Lower strip: what that score is made of -- plus 'exact', the all-or-nothing conjunction, which is
             simply the fraction of images whose score is exactly 1.0.

The three arms are NOT matched on training steps or LoRA capacity; they are three different
configurations stopped at different points. The figure compares what exists, and the caption says
so -- it is not a controlled ablation.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, "/mmfs1/gscratch/socialrl/sriyash/DiT/helper_scripts")
sys.path.insert(0, "/gscratch/scrubbed/sriyash/onpolicy-exp/repro/analysis")
from plot_style import MARKERS, apply_house_style, legend_below, style_axes  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from plot_two_tier import iid_pass, load_ceilings, mse, seq_pass  # noqa: E402

# 'exact' first: it is the headline elsewhere, so keep it visible here too.
STRIP = [("exact", "exactly right"), ("presence", "right object present"),
         ("shape", "shape"), ("color", "colour"), ("precision", "no extras")]


def best_score(rows, k):
    """Overall verifier score of the best image among the first k attempts."""
    return np.array([max(a["score"] for a in r["bd"][:k]) for r in rows])


def best_comp(rows, k, key):
    if key == "exact":
        return None  # handled by the pass@k helpers, which differ for seq vs iid
    return np.array([max(a[key] for a in r["bd"][:k]) for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True,
                    help="label=path/to/eval.json, in the order to draw them")
    ap.add_argument("--ceiling_file", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    arms = []
    for spec in a.arms:
        label, path = spec.split("=", 1)
        arms.append((label, json.load(open(path))))
    ks = list(range(1, min(b["n"] for _, b in arms) + 1))
    for label, b in arms:
        assert b["prompts"] == arms[0][1]["prompts"], f"{label} used different prompts"
    top_ceil, comp_ceil = load_ceilings(a.ceiling_file)

    colours = ["#a8324e", "#3a6ea5", "#2e7d32", "#8c6d31"]
    apply_house_style()
    fig = plt.figure(figsize=(9.2, 9.0))
    gs = gridspec.GridSpec(2, len(STRIP), height_ratios=[2.9, 1.0], hspace=0.30, wspace=0.34,
                           top=0.95, bottom=0.05)
    ax = fig.add_subplot(gs[0, 1:4])

    lo, hi = 1.0, comp_ceil["score"]
    for i, (label, blob) in enumerate(arms):
        for kind, ls, suffix in (("seq", "-", "feedback chain"), ("iid", "--", "independent")):
            rows = blob.get(kind)
            if not rows:
                continue
            m = [mse(best_score(rows, k)) for k in ks]
            lo = min(lo, min(v[0] - v[1] for v in m))
            hi = max(hi, max(v[0] + v[1] for v in m))
            ax.errorbar(ks, [v[0] for v in m], yerr=[v[1] for v in m],
                        label=f"{label} · {suffix}", color=colours[i % len(colours)],
                        marker=MARKERS[i % len(MARKERS)], markersize=5, capsize=2, lw=1.8,
                        linestyle=ls)
    ax.axhline(comp_ceil["score"], ls=":", lw=1.2, color="0.45",
               label="verifier ceiling on real renders")
    ax.set_ylim(max(0.0, lo - 0.03), hi + 0.03)
    ax.set_xlabel("budget  k")
    ax.set_ylabel("verifier score of the best image within k")
    ax.set_title(f"On-policy training variants ({len(arms)}) — overall verifier score",
                 fontsize=12)
    ax.set_xticks(ks)
    style_axes(ax)

    for j, (key, nice) in enumerate(STRIP):
        sub = fig.add_subplot(gs[1, j])
        ceil = top_ceil if key == "exact" else comp_ceil[key]
        lo = hi = ceil
        for i, (label, blob) in enumerate(arms):
            for kind, ls in (("seq", "-"), ("iid", "--")):
                rows = blob.get(kind)
                if not rows:
                    continue
                if key == "exact":
                    v = [(seq_pass if kind == "seq" else iid_pass)(rows, k).mean() for k in ks]
                else:
                    v = [best_comp(rows, k, key).mean() for k in ks]
                lo, hi = min(lo, min(v)), max(hi, max(v))
                sub.plot(ks, v, color=colours[i % len(colours)], marker=MARKERS[i % len(MARKERS)],
                         markersize=3, lw=1.3, linestyle=ls)
        sub.axhline(ceil, ls=":", lw=1.2, color="0.45")
        pad = max(0.02, (hi - lo) * 0.12)
        sub.set_ylim(lo - pad, hi + pad)
        sub.set_title(nice, fontsize=9, color="0.3")
        sub.set_xticks(ks)
        sub.tick_params(labelsize=7)
        if j == 0:
            sub.set_ylabel("verifier metric", fontsize=8)
        style_axes(sub)

    legend_below(fig, handles_from=ax, ncol=2)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=185, bbox_inches="tight")

    print(f"{'arm':<34}{'score@1':>9}{'score@8':>9}{'exact@8':>9}")
    for label, blob in arms:
        r = blob.get("seq") or blob["iid"]
        print(f"{label:<34}{best_score(r, 1).mean():>9.3f}{best_score(r, ks[-1]).mean():>9.3f}"
              f"{seq_pass(r, ks[-1]).mean():>9.3f}")
    print("saved ->", a.out)


if __name__ == "__main__":
    main()
