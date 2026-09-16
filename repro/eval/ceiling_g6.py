"""Per-metric verifier ceiling on clevr_g6.

Grades the GROUND-TRUTH renders of the very same 200 held-out captions used for every pass@k
number, through the identical cell-aware verifier. A real image does not score 1.0 -- the detector
misses objects, misreads a shape, nudges a cell -- so every panel needs its own upper bound rather
than a shared 1.0. Writes one JSON the plotting code reads.
"""
import argparse
import json
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, "/mmfs1/gscratch/socialrl/sriyash/DiT/.claude/worktrees/g6-anchor")
sys.path.insert(0, "/gscratch/scrubbed/sriyash/.claude/jobs/9b095793/tmp")
from algorithms.eval import select_eval_batch  # noqa: E402
from seq_vs_iid_passk import BREAKDOWN_KEYS, build_verifier, grade  # noqa: E402

RUN = "/gscratch/socialrl/sriyash/OmniGen-clevr-g6-tiny/clevr_g6_tiny"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--out", default="/gscratch/socialrl/sriyash/clevr_g6_bon/ceiling_g6.json")
    a = ap.parse_args()

    cfg = OmegaConf.load(Path(RUN) / "config.yaml")
    ds = hydra.utils.instantiate(cfg.dataset, split=a.split)
    batch = select_eval_batch(ds, 0, a.count)
    verifier = build_verifier()

    scores, exact, bds = [], [], []
    for s in range(0, a.count, 25):
        caps = batch["caption"][s:s + 25]
        imgs = batch["gt_images"][s:s + 25]
        sc, ex, _fb, bd = grade(verifier, caps, imgs)
        scores += sc
        exact += list(ex)
        bds += bd
        print(f"  graded {s + len(caps)}/{a.count}", flush=True)

    out = {"n": len(exact), "exact": float(np.mean(exact))}
    for k in BREAKDOWN_KEYS:
        out[k] = float(np.mean([b[k] for b in bds]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)

    print(f"\n=== verifier ceiling on the {a.count} {a.split} captions' real renders ===")
    print(f"  exact scene : {out['exact']:.4f}")
    for k in BREAKDOWN_KEYS:
        print(f"  {k:<11} : {out[k]:.4f}")
    print("saved ->", a.out)


if __name__ == "__main__":
    main()
