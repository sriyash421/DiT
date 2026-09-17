"""Does the critique's CONTENT carry the chain's benefit?

Same model, same captions, same fixed noise, same previous image in context. The only change is
that every critique is replaced by the verifier's own "no update" string. Reuses the two-tier
layout of the other pass@k figures so it reads against them directly.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, "/gscratch/scrubbed/sriyash/onpolicy-exp/repro/analysis")
from plot_two_tier import comp_best, draw, iid_pass, load_ceilings, seq_pass  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--placebo", required=True)
    ap.add_argument("--ceiling_file", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    real, plac = json.load(open(a.real)), json.load(open(a.placebo))
    assert real["prompts"] == plac["prompts"], "different captions -- not comparable"
    ks = list(range(1, min(real["n"], plac["n"]) + 1))
    top_ceil, comp_ceil = load_ceilings(a.ceiling_file)

    series = [
        ("feedback chain · real critiques", real, "seq", "#a8324e", 3),
        ("feedback chain · every critique replaced by “no update”", plac, "seq", "#8c6d31", 1),
        ("same model · independent samples", real, "iid", "#3a6ea5", 2),
    ]

    draw(series, ks,
         top_fn=lambda rows, k, kind: (seq_pass if kind == "seq" else iid_pass)(rows, k),
         comp_fn=lambda rows, k, key: comp_best(rows, k, key),
         top_label="fraction of captions solved exactly",
         top_title="Is it the critique, or just having a previous image in context?",
         out_path=a.out, top_ceil=top_ceil, comp_ceil=comp_ceil)

    import numpy as np
    for label, blob, kind, _c, _i in series:
        rows = blob["seq"] if kind == "seq" else blob["iid"]
        fn = seq_pass if kind == "seq" else iid_pass
        print(f"  {label:<52} pass@1 {np.mean(fn(rows, 1)):.3f}  pass@8 {np.mean(fn(rows, ks[-1])):.3f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
