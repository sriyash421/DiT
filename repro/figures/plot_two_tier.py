"""Two-tier line figures: the headline metric on top, the verifier's component metrics below.

  pass@k  -- union over the first k attempts. Top panel is exact scenes; the lower strip shows the
             BEST value each component metric reached within k attempts.
  perf@k  -- attempt k alone. Top panel is exact scenes at attempt k; the lower strip shows each
             component metric at attempt k.

Components come from the verifier's own breakdown: score, presence, shape, colour, precision.
(quality is omitted from the strip -- it is ~1.0 everywhere, since blur is rare on these renders.)
"""
import argparse, glob, json, re, sys
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, "/mmfs1/gscratch/socialrl/sriyash/DiT/helper_scripts")
from plot_style import MARKERS, apply_house_style, legend_below, style_axes  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CEILING_FILE = "/gscratch/socialrl/sriyash/clevr_g6_bon/ceiling_g6.json"
COMPONENTS = [("score", "overall score"), ("presence", "right object present"),
              ("shape", "shape"), ("color", "colour"), ("precision", "no extras")]


def load_ceilings(path=None):
    """The verifier's own reading of the 200 held-out captions' REAL renders.

    A perfect generator cannot beat this: the detector misses objects, misreads a shape, nudges a
    cell. Each metric has its own ceiling, so each panel gets its own line rather than a shared 1.0.
    """
    c = json.load(open(path or CEILING_FILE))
    return c["exact"], {k: c[k] for k, _ in COMPONENTS}


def draw_ceiling(ax, y, label=None):
    ax.axhline(y, ls=":", lw=1.2, color="0.45", label=label)


def seq_pass(rows, k):
    return np.array([1.0 if any(r["exact"][:k]) else 0.0 for r in rows])


def iid_pass(rows, k):
    out = []
    for r in rows:
        n, c = len(r["exact"]), int(sum(r["exact"]))
        out.append(1.0 - (comb(n - c, k) / comb(n, k) if n - c >= k else 0.0))
    return np.array(out)


def perf_exact(rows, k):
    return np.array([1.0 if r["exact"][k - 1] else 0.0 for r in rows])


def comp_best(rows, k, key):
    """Best value of a component within the first k attempts."""
    return np.array([max(a[key] for a in r["bd"][:k]) for r in rows])


def comp_at(rows, k, key):
    return np.array([r["bd"][k - 1][key] for r in rows])


def mse(a):
    return a.mean(), a.std(ddof=1) / np.sqrt(len(a))


def draw(series, ks, top_fn, comp_fn, top_label, top_title, out_path, top_ceil,
         comp_ceil, components=True):
    apply_house_style()
    # Taller than wide on the headline panel: a 12x4 box squashes every curve into a flat line.
    if components:
        fig = plt.figure(figsize=(9.2, 9.0))
        gs = gridspec.GridSpec(2, len(COMPONENTS), height_ratios=[2.9, 1.0],
                               hspace=0.30, wspace=0.34, top=0.95, bottom=0.05)
        # Middle three columns only: a full-width headline panel flattens every curve.
        ax = fig.add_subplot(gs[0, 1:4])
    else:
        fig = plt.figure(figsize=(6.0, 6.2))
        ax = fig.add_subplot(1, 1, 1)
    hi = top_ceil
    for label, blob, kind, colour, i in series:
        rows = blob["seq"] if kind == "seq" else blob["iid"]
        m = [mse(top_fn(rows, k, kind)) for k in ks]
        hi = max(hi, max(v[0] + v[1] for v in m))
        ax.errorbar(ks, [v[0] for v in m], yerr=[v[1] for v in m], label=label, color=colour,
                    marker=MARKERS[i % len(MARKERS)], markersize=5, capsize=2, lw=1.8,
                    linestyle="-" if kind == "seq" else "--")
    draw_ceiling(ax, top_ceil, "verifier ceiling on real renders")
    ax.set_ylim(0.0, hi * 1.06)
    ax.set_xlabel("budget  k")
    ax.set_ylabel(top_label)
    ax.set_title(top_title, fontsize=12)
    ax.set_xticks(ks)
    style_axes(ax)

    for j, (key, nice) in enumerate(COMPONENTS if components else []):
        sub = fig.add_subplot(gs[1, j])
        lo = hi = comp_ceil[key]
        for label, blob, kind, colour, i in series:
            rows = blob["seq"] if kind == "seq" else blob["iid"]
            m = [mse(comp_fn(rows, k, key)) for k in ks]
            lo = min(lo, min(v[0] for v in m))
            hi = max(hi, max(v[0] for v in m))
            sub.plot(ks, [v[0] for v in m], color=colour, marker=MARKERS[i % len(MARKERS)],
                     markersize=3, lw=1.3, linestyle="-" if kind == "seq" else "--")
        draw_ceiling(sub, comp_ceil[key])
        pad = max(0.02, (hi - lo) * 0.12)
        sub.set_ylim(lo - pad, hi + pad)
        sub.set_title(nice, fontsize=9, color="0.3")
        sub.set_xticks(ks)
        sub.tick_params(labelsize=7)
        if j == 0:
            sub.set_ylabel("verifier metric", fontsize=8)
        style_axes(sub)

    legend_below(fig, handles_from=fig.axes[0], ncol=2)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=185, bbox_inches="tight")
    plt.close(fig)
    print("saved ->", out_path)


def pick_best_offline(pattern):
    """The offline model's strongest checkpoint by pass@8, so the baseline is its best form.

    Selection happens on the same 200 captions the figures report. That favours the baseline, which
    is the direction that makes the feedback claim harder, not easier.
    """
    cands = []
    for f in sorted(glob.glob(pattern)):
        m = re.search(r"bd_converged_(\d+)\.json", f)
        if not m:
            continue
        blob = json.load(open(f))
        cands.append((int(m.group(1)), f, blob, float(iid_pass(blob["iid"], blob["n"]).mean())))
    assert cands, f"no offline checkpoints matched {pattern}"
    for step, _f, _b, v in cands:
        print(f"  offline {step:>6,} steps  pass@8 = {v:.3f}")
    step, f, blob, v = max(cands, key=lambda c: c[3])
    print(f"  -> using the best: {step:,} steps ({v:.3f})")
    return step, blob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--converged", required=True,
                    help="the same 100 images trained normally to convergence, no feedback -- the "
                         "'just train and sample more' baseline everything has to beat")
    ap.add_argument("--trained", required=True)
    ap.add_argument("--nofeedback", default=None,
                    help="LoRA trained identically but with no feedback -- the fair baseline")
    ap.add_argument("--placebo", default=None,
                    help="the in-context model with every critique replaced by 'no update'")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--passk_name", default="fig6_passk.png")
    ap.add_argument("--perfk_name", default="fig7_perfk.png")
    ap.add_argument("--ceiling_file", default=None)
    ap.add_argument("--caption_set", default="held-out",
                    help="named in the titles, e.g. 'held-out' or 'training'")
    ap.add_argument("--no_components", action="store_true",
                    help="headline panel only, no per-metric strip")
    a = ap.parse_args()

    base, tr = json.load(open(a.base)), json.load(open(a.trained))
    nofb = json.load(open(a.nofeedback)) if a.nofeedback else None
    plac = json.load(open(a.placebo)) if a.placebo else None
    if any(ch in a.converged for ch in "*?["):
        cv_step, cv = pick_best_offline(a.converged)
    else:
        cv, cv_step = json.load(open(a.converged)), json.load(open(a.converged))["step"]
    for name, blob in (("base", base), ("trained", tr), ("converged", cv),
                       ("nofeedback", nofb), ("placebo", plac)):
        if blob is not None:
            assert blob["prompts"] == base["prompts"], f"{name} used different captions"
    for name, blob in (("base", base), ("converged", cv), ("trained", tr),
                       ("nofeedback", nofb), ("placebo", plac)):
        if blob is None:
            continue
        rows = blob.get("seq") or blob.get("iid")
        assert "bd" in rows[0], f"no breakdown in the {name} file -- re-run the eval"
    ks = list(range(1, min(b["n"] for b in (base, cv, tr, nofb, plac) if b is not None) + 1))
    top_ceil, comp_ceil = load_ceilings(a.ceiling_file)
    print("ceilings -> exact %.4f  " % top_ceil
          + "  ".join(f"{k}:{v:.4f}" for k, v in comp_ceil.items()))

    series = [("undertrained base · independent samples", base, "iid", "0.68", 0),
              ("undertrained base · feedback chain", base, "seq", "#8c6d31", 1),
              (f"offline full fine-tune, best checkpoint ({cv_step:,} steps) · independent samples",
               cv, "iid", "#2e7d32", 4)]
    if nofb is not None:
        # Same LoRA, same data, same steps, no critiques: the fair reference for the feedback claim.
        series.append(("no-feedback LoRA (same setup, no critiques) · independent samples",
                       nofb, "iid", "#6a3d9a", 5))
    series += [("in-context model · independent samples", tr, "iid", "#3a6ea5", 2),
               ("in-context model · feedback chain", tr, "seq", "#a8324e", 3)]
    if plac is not None:
        series.append(("in-context model · chain with critiques replaced by \u201cno update\u201d",
                       plac, "seq", "#e08214", 6))

    draw(series, ks,
         top_fn=lambda rows, k, kind: (seq_pass if kind == "seq" else iid_pass)(rows, k),
         comp_fn=lambda rows, k, key: comp_best(rows, k, key),
         top_label="fraction of captions solved exactly",
         top_title=f"pass@k on {a.caption_set} captions — did any of the first k attempts land "
                   "exactly right?",
         out_path=str(Path(a.out_dir) / a.passk_name),
         top_ceil=top_ceil, comp_ceil=comp_ceil, components=not a.no_components)

    draw(series, ks,
         top_fn=lambda rows, k, kind: perf_exact(rows, k),
         comp_fn=lambda rows, k, key: comp_at(rows, k, key),
         top_label="fraction exact at attempt k",
         top_title=f"performance@k on {a.caption_set} captions — is attempt k itself correct?",
         out_path=str(Path(a.out_dir) / a.perfk_name),
         top_ceil=top_ceil, comp_ceil=comp_ceil, components=not a.no_components)


if __name__ == "__main__":
    main()
