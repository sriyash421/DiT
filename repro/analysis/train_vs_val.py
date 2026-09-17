"""In-context model @5000: train vs held-out, pass@k and perf@k side by side.

The question this answers: how much of the shortfall is FAILING TO FIT the 100 training captions
(which more capacity could buy back) versus FAILING TO TRANSFER to new ones (which it would not)?
"""
import json

import numpy as np

TR = "/gscratch/socialrl/sriyash/clevr_g6_bon/seqiid_train/bd_incontext_5000.json"
TE = "/gscratch/socialrl/sriyash/clevr_g6_bon/seqiid/bd_incontext_5000.json"
CEIL_TR = json.load(open("/gscratch/socialrl/sriyash/clevr_g6_bon/ceiling_g6_train.json"))["exact"]
CEIL_TE = json.load(open("/gscratch/socialrl/sriyash/clevr_g6_bon/ceiling_g6.json"))["exact"]


def mse(a):
    a = np.asarray(a, float)
    return a.mean(), a.std(ddof=1) / np.sqrt(len(a))


def passk(rows, k):
    return mse([1.0 if any(r["exact"][:k]) else 0.0 for r in rows])


def perfk(rows, k):
    return mse([1.0 if r["exact"][k - 1] else 0.0 for r in rows])


tr, te = json.load(open(TR))["seq"], json.load(open(TE))["seq"]
ks = [1, 2, 4, 8]

print("in-context model @5000, feedback chain            (mean +/- standard error)\n")
print(f"{'':10}{'':>18}" + "".join(f"{'k=' + str(k):>16}" for k in ks))
for name, fn in (("pass@k", passk), ("perf@k", perfk)):
    for split, rows, n in (("train", tr, 100), ("held-out", te, 200)):
        cells = "".join(f"{v[0]:>10.3f}+/-{v[1]:.3f}" for v in (fn(rows, k) for k in ks))
        print(f"{name:10}{split + ' (n=' + str(n) + ')':>18}{cells}")
    print()

print(f"verifier ceiling on exact:   train {CEIL_TR:.3f}   held-out {CEIL_TE:.3f}\n")

tr8, te8 = passk(tr, 8)[0], passk(te, 8)[0]
print("decomposing the shortfall at k=8 (pass@k):")
print(f"  ceiling on train                      {CEIL_TR:.3f}")
print(f"  what it achieves on TRAIN             {tr8:.3f}")
print(f"  -> FIT gap (capacity/optimisation)    {CEIL_TR - tr8:.3f}")
print(f"  what it achieves on HELD-OUT          {te8:.3f}")
print(f"  -> TRANSFER gap (generalisation)      {tr8 - te8:.3f}")
print(f"\n  transfer gap is {(tr8 - te8) / max(1e-9, CEIL_TR - te8):.0%} of the total shortfall")
