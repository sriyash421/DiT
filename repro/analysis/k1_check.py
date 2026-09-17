"""At k=1 the chain and iid must estimate the SAME thing: P(a fresh unconditioned sample is exact).

Attempt 1 of a chain has an empty feedback history and no attempt images, so it is an ordinary
sample. If the two numbers differ by more than noise, the eval is conditioning them differently and
that is a bug worth finding.

The two estimators are not the same statistic though:
  seq  pass@1 = exact rate of ONE sample per caption            (200 Bernoulli draws)
  iid  pass@1 = 1 - C(n-c,1)/C(n,1) = c/n = exact rate over ALL 8 samples per caption
so the iid one is an average over 8x more samples and is far less noisy.
"""
import json
from math import comb

import numpy as np

d = json.load(open("/gscratch/socialrl/sriyash/clevr_g6_bon/seqiid/bd_incontext_5000.json"))
seq, iid = d["seq"], d["iid"]


def mse(a):
    a = np.asarray(a, float)
    return a.mean(), a.std(ddof=1) / np.sqrt(len(a))


seq1 = [1.0 if r["exact"][0] else 0.0 for r in seq]
iid_first = [1.0 if r["exact"][0] else 0.0 for r in iid]          # same statistic as seq1
iid_all = [float(np.mean([bool(e) for e in r["exact"]])) for r in iid]  # what pass@1 reports
iid_est = [1.0 - (comb(len(r["exact"]) - int(sum(r["exact"])), 1) / comb(len(r["exact"]), 1))
           for r in iid]

print("apples to apples -- ONE sample per caption:")
print(f"  chain, attempt 1        {mse(seq1)[0]:.3f} +/- {mse(seq1)[1]:.3f}")
print(f"  iid, first sample only  {mse(iid_first)[0]:.3f} +/- {mse(iid_first)[1]:.3f}")
print(f"  difference              {mse(seq1)[0] - mse(iid_first)[0]:+.3f}")

print("\nwhat the figure actually plots at k=1:")
print(f"  iid pass@1 (all 8 samples averaged) {mse(iid_all)[0]:.3f} +/- {mse(iid_all)[1]:.3f}")
print(f"  matches the unbiased estimator?     {np.allclose(iid_all, iid_est)}")

diff = mse(seq1)[0] - mse(iid_all)[0]
se = (mse(seq1)[1] ** 2 + mse(iid_all)[1] ** 2) ** 0.5
print(f"\nchain@1 - iid@1 = {diff:+.3f}, combined standard error {se:.3f} -> {abs(diff) / se:.2f} SE")
print("same distribution, different sample size" if abs(diff) < 2 * se else "SYSTEMATIC -- investigate")
