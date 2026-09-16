"""The offline model's whole trajectory, so the claim is against the BEST no-feedback model rather
than one arbitrary checkpoint.

Left  : best-of-8 exact-scene rate vs training step for the offline (no-feedback) model, with the
        in-context model drawn as a horizontal band for comparison.
Right : the same for the average verifier score.

If the offline curve peaks and then declines, that is worth reporting plainly -- it says the extra
budget is being spent on memorising rather than generalising.
"""
import argparse, glob, json, re, sys
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, "/mmfs1/gscratch/socialrl/sriyash/DiT/helper_scripts")
from plot_style import MARKERS, apply_house_style, legend_below, style_axes  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def iid_pass(rows, k):
    out = []
    for r in rows:
        n, c = len(r["exact"]), int(sum(r["exact"]))
        out.append(1.0 - (comb(n - c, k) / comb(n, k) if n - c >= k else 0.0))
    return np.array(out)


def best_score(rows, k):
    return np.array([max(r["score"][:k]) for r in rows])


def seq_pass(rows, k):
    return np.array([1.0 if any(r["exact"][:k]) else 0.0 for r in rows])


def best_chain_score(rows, k):
    return np.array([max(r["score"][:k]) for r in rows])


def mse(a):
    return a.mean(), a.std(ddof=1) / np.sqrt(len(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="/gscratch/socialrl/sriyash/clevr_g6_bon/seqiid/bd_converged_*.json")
    ap.add_argument("--in_context", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pts = []
    for f in sorted(glob.glob(a.glob)):
        m = re.search(r"bd_converged_(\d+)\.json", f)
        if not m:
            continue
        pts.append((int(m.group(1)), json.load(open(f))))
    pts.sort()
    assert pts, "no converged-base evals found"
    ic = json.load(open(a.in_context))

    steps = [s for s, _ in pts]
    print(f"offline (no-feedback) checkpoints found: {steps}")
    apply_house_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))

    for ax, fn_iid, fn_chain, ylab, title in (
            (axes[0], iid_pass, seq_pass, f"exact-scene rate at k={a.k}",
             f"Best-of-{a.k}: how often exactly right"),
            (axes[1], best_score, best_chain_score, f"mean verifier score at k={a.k}",
             f"Best-of-{a.k}: average verifier score")):
        m = [mse(fn_iid(b["iid"], a.k)) for _, b in pts]
        ax.errorbar(steps, [v[0] for v in m], yerr=[v[1] for v in m], color="#2e7d32",
                    marker=MARKERS[0], markersize=6, capsize=3, lw=1.8,
                    label="offline model (no feedback), best-of-k")
        print(f"  {title}: " + "  ".join(f"{s}:{v[0]:.3f}" for s, v in zip(steps, m)))

        ref = mse(fn_chain(ic["seq"], a.k))
        ax.axhline(ref[0], ls="--", lw=1.8, color="#a8324e",
                   label="in-context model, feedback chain")
        ax.axhspan(ref[0] - ref[1], ref[0] + ref[1], color="#a8324e", alpha=0.12)
        print(f"     in-context chain: {ref[0]:.3f} +/- {ref[1]:.3f}")

        ax.set_xlabel("offline training steps on the same 100 images")
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=11)
        ax.set_xticks(steps); ax.tick_params(axis="x", labelrotation=45)
        style_axes(ax)

    legend_below(fig, handles_from=axes[0], ncol=2)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=190, bbox_inches="tight")
    print("saved ->", a.out)


if __name__ == "__main__":
    main()
