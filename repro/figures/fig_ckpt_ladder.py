"""Checkpoint ladder: every offline (no-feedback) checkpoint's pass@k, against the feedback chain.

The offline model trained on the same 100 images is evaluated at each checkpoint along its run and
drawn as a light-to-dark blue gradient, so "train longer and sample more" is shown as the whole
family of baselines rather than one convenient checkpoint. The in-context model's feedback chain is
the dark purple line. Every curve carries the standard error across the 200 captions.

Also prints which offline checkpoint is best at k=8 -- that is the one the headline figure uses.
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, "/mmfs1/gscratch/socialrl/sriyash/DiT/helper_scripts")
sys.path.insert(0, "/gscratch/scrubbed/sriyash/onpolicy-exp/repro/analysis")
from plot_style import apply_house_style, legend_below, style_axes  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from plot_two_tier import COMPONENTS, comp_best, iid_pass, load_ceilings, mse, seq_pass  # noqa: E402

CHAIN_COLOUR = "#4b1d6b"      # dark purple: the feedback chain
BLUES = plt.get_cmap("Blues")


def load_ladder(pattern):
    pts = []
    for f in sorted(glob.glob(pattern)):
        m = re.search(r"bd_converged_(\d+)\.json", f)
        if m:
            pts.append((int(m.group(1)), json.load(open(f))))
    pts.sort()
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="/gscratch/socialrl/sriyash/clevr_g6_bon/seqiid/bd_converged_*.json")
    ap.add_argument("--in_context", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ceiling_file", default=None)
    ap.add_argument("--caption_set", default="held-out")
    ap.add_argument("--no_components", action="store_true")
    a = ap.parse_args()

    pts = load_ladder(a.glob)
    assert pts, "no offline checkpoints found"
    ic = json.load(open(a.in_context))
    ks = list(range(1, ic["n"] + 1))
    top_ceil, comp_ceil = load_ceilings(a.ceiling_file)

    # Shade light -> dark with training step, leaving out the palest end so every line stays legible.
    shades = np.linspace(0.32, 0.95, len(pts))

    apply_house_style()
    if a.no_components:
        fig = plt.figure(figsize=(6.0, 6.2))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = plt.figure(figsize=(9.2, 9.0))
        gs = gridspec.GridSpec(2, len(COMPONENTS), height_ratios=[2.9, 1.0],
                               hspace=0.30, wspace=0.34, top=0.95, bottom=0.05)
        ax = fig.add_subplot(gs[0, 1:4])

    hi = top_ceil
    best = None
    for (step, blob), sh in zip(pts, shades):
        m = [mse(iid_pass(blob["iid"], k)) for k in ks]
        hi = max(hi, max(v[0] + v[1] for v in m))
        ax.errorbar(ks, [v[0] for v in m], yerr=[v[1] for v in m], color=BLUES(sh),
                    marker="o", markersize=4, capsize=2, lw=1.6, linestyle="--",
                    label=f"offline {step:,} steps · best-of-k")
        if best is None or m[-1][0] > best[1]:
            best = (step, m[-1][0])

    m = [mse(seq_pass(ic["seq"], k)) for k in ks]
    hi = max(hi, max(v[0] + v[1] for v in m))
    ax.errorbar(ks, [v[0] for v in m], yerr=[v[1] for v in m], color=CHAIN_COLOUR,
                marker="D", markersize=5, capsize=2, lw=2.4,
                label="in-context model · feedback chain")

    ax.axhline(top_ceil, ls=":", lw=1.2, color="0.45", label="verifier ceiling on real renders")
    ax.set_ylim(0.0, hi * 1.06)
    ax.set_xlabel("budget  k")
    ax.set_ylabel("fraction of captions solved exactly")
    ax.set_title(f"pass@k on {a.caption_set} captions, against every offline checkpoint",
                 fontsize=12)
    ax.set_xticks(ks)
    style_axes(ax)

    for j, (key, nice) in enumerate([] if a.no_components else COMPONENTS):
        sub = fig.add_subplot(gs[1, j])
        lo = hi = comp_ceil[key]
        for (step, blob), sh in zip(pts, shades):
            v = [mse(comp_best(blob["iid"], k, key))[0] for k in ks]
            lo, hi = min(lo, min(v)), max(hi, max(v))
            sub.plot(ks, v, color=BLUES(sh), marker="o", markersize=3, lw=1.2, linestyle="--")
        v = [mse(comp_best(ic["seq"], k, key))[0] for k in ks]
        lo, hi = min(lo, min(v)), max(hi, max(v))
        sub.plot(ks, v, color=CHAIN_COLOUR, marker="D", markersize=3, lw=1.6)
        sub.axhline(comp_ceil[key], ls=":", lw=1.2, color="0.45")
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
    chain_k = mse(seq_pass(ic["seq"], ks[-1]))[0]
    print(f"best offline checkpoint at k={ks[-1]}: step {best[0]:,} -> {best[1]:.3f}")
    print(f"in-context feedback chain at k={ks[-1]}: {chain_k:.3f}")
    print("saved ->", a.out)


if __name__ == "__main__":
    main()
