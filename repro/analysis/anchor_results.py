"""The anchored arms against the baselines, same 200 held-out captions.

Two eval windows are present for each arm. That matters here: the anchored arms train at CHAIN
LENGTH 2, so a single prior attempt (window 1) is the context they were actually trained under,
while window 3 is what every earlier arm was reported at. Both are shown rather than picking one.
"""
import json
import os
from math import comb

import numpy as np

S = "/gscratch/socialrl/sriyash/clevr_g6_bon/seqiid"
ARMS = [
    ("offline full FT @2500 (baseline)", "bd_converged_2500.json", "iid"),
    ("no-feedback LoRA @5000", "bd_nofeedback_5000.json", "iid"),
    ("in-context @5000 · chain", "bd_incontext_5000.json", "seq"),
    ("in-context @5000 · best-of-k", "bd_incontext_5000.json", "iid"),
    ("anchor A (beta=1) @1750 · chain  [w3]", "bd_anchorA_1750.json", "seq"),
    ("anchor A (beta=1) @1750 · chain  [w1]", "bd_anchorA_1750_w1.json", "seq"),
    ("anchor B (beta=.2) @2750 · chain [w3]", "bd_anchorB_2750.json", "seq"),
    ("anchor B (beta=.2) @2750 · chain [w1]", "bd_anchorB_2750_w1.json", "seq"),
]


def mse(a):
    a = np.asarray(a, float)
    return a.mean(), a.std(ddof=1) / np.sqrt(len(a))


def passk(rows, k):
    return [1.0 if any(r["exact"][:k]) else 0.0 for r in rows]


def iidk(rows, k):
    out = []
    for r in rows:
        n, c = len(r["exact"]), int(sum(r["exact"]))
        out.append(1.0 - (comb(n - c, k) / comb(n, k) if n - c >= k else 0.0))
    return out


def perfk(rows, k):
    return [1.0 if r["exact"][k - 1] else 0.0 for r in rows]


ks = [1, 2, 4, 8]
ref = None
print(f"{'arm':<40}" + "".join(f"{'k=' + str(k):>15}" for k in ks))
for name, fname, kind in ARMS:
    p = os.path.join(S, fname)
    if not os.path.exists(p):
        print(f"{name:<40}  (missing)")
        continue
    b = json.load(open(p))
    if ref is None:
        ref = b["prompts"]
    elif b["prompts"] != ref:
        print(f"{name:<40}  !! DIFFERENT CAPTIONS -- not comparable")
        continue
    rows = b["seq"] if kind == "seq" else b["iid"]
    fn = passk if kind == "seq" else iidk
    cells = "".join(f"{m:>9.3f}+/-{s:.3f}" for m, s in (mse(fn(rows, k)) for k in ks))
    print(f"{name:<40}{cells}")

print("\nperf@k (attempt k alone) for the anchored arms -- does the repair step actually fire?")
for name, fname, kind in ARMS[4:]:
    p = os.path.join(S, fname)
    if not os.path.exists(p):
        continue
    rows = json.load(open(p))["seq"]
    cells = "".join(f"{m:>9.3f}+/-{s:.3f}" for m, s in (mse(perfk(rows, k)) for k in ks))
    print(f"{name:<40}{cells}")

print("\nrepair rate: of chains WRONG at attempt 1, how many were later fixed?")
for name, fname, kind in ARMS[2:]:
    p = os.path.join(S, fname)
    if not os.path.exists(p) or kind != "seq":
        continue
    rows = json.load(open(p))["seq"]
    wrong = [r for r in rows if not r["exact"][0]]
    fixed = [r for r in wrong if any(r["exact"][1:])]
    print(f"  {name:<38} {len(fixed):>3}/{len(wrong):<4} ({len(fixed)/max(1,len(wrong)):.1%})")
