"""The three claims, with the numbers that back each one.

A. the in-context policy is better AT STEP 0 (k=1) than a no-feedback model given the same training
B. it beats iid sampling from the undertrained base or any offline checkpoint, at every budget
C. at convergence everything looks alike ON TRAIN -- except the undertrained base

Prints '--' for arms whose eval has not landed yet rather than inventing a number.
"""
import glob
import json
import os
import re
from math import comb

import numpy as np

TE = "/gscratch/socialrl/sriyash/clevr_g6_bon/seqiid"
TR = "/gscratch/socialrl/sriyash/clevr_g6_bon/seqiid_train"


def load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def mse(a):
    a = np.asarray(a, float)
    return a.mean(), a.std(ddof=1) / np.sqrt(len(a))


def seq_pass(rows, k):
    return mse([1.0 if any(r["exact"][:k]) else 0.0 for r in rows])


def iid_pass(rows, k):
    out = []
    for r in rows:
        n, c = len(r["exact"]), int(sum(r["exact"]))
        out.append(1.0 - (comb(n - c, k) / comb(n, k) if n - c >= k else 0.0))
    return mse(out)


def cell(blob, kind, k):
    if blob is None or not blob.get(kind):
        return f"{'--':>14}"
    fn = seq_pass if kind == "seq" else iid_pass
    m, s = fn(blob[kind], k)
    return f"{m:>8.3f}+/-{s:.3f}"


STEPS = sorted(int(re.search(r"_(\d+)\.json", f).group(1))
               for f in glob.glob(os.path.join(TE, "bd_converged_*.json")))
ALL = sorted(int(re.search(r"_(\d+)\.json", f).group(1))
             for f in glob.glob(os.path.join(TR, "bd_converged_*.json")))

arms = [("undertrained base @250", "bd_undertrained_250.json", "iid"),
        ("in-context @5000  · chain", "bd_incontext_5000.json", "seq"),
        ("in-context @5000  · iid", "bd_incontext_5000.json", "iid")]
arms += [(f"offline @{s:<6} · iid", f"bd_converged_{s}.json", "iid") for s in sorted(set(ALL))]

for split, d in (("HELD-OUT (200 captions)", TE), ("TRAIN (100 captions)", TR)):
    print(f"\n=== {split} — pass@k, exact scenes ===")
    print(f"{'arm':<28}" + "".join(f"{'k=' + str(k):>14}" for k in (1, 2, 4, 8)))
    for label, fname, kind in arms:
        blob = load(d, fname)
        print(f"{label:<28}" + "".join(cell(blob, kind, k) for k in (1, 2, 4, 8)))
    if d is TE and set(ALL) - set(STEPS):
        print(f"  (offline held-out evals still running: {sorted(set(ALL) - set(STEPS))})")
