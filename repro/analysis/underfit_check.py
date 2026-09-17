"""Is the chain's plateau underfitting, or failure to generalise?

If the model cannot ride its own chain even on the 100 captions it TRAINED on, the horizon really
is too hard to fit and more capacity (full FT) is the right lever. If the chain improves fine on
train and only stalls on held-out, capacity is not the binding constraint.
"""
import json

import numpy as np

TR = "/gscratch/socialrl/sriyash/clevr_g6_bon/seqiid_train/bd_incontext_5000.json"
TE = "/gscratch/socialrl/sriyash/clevr_g6_bon/seqiid/bd_incontext_5000.json"


def perf_at(rows, k):
    """Attempt k alone -- this is what has to rise if the chain is working."""
    return float(np.mean([1.0 if r["exact"][k - 1] else 0.0 for r in rows]))


def score_at(rows, k):
    return float(np.mean([r["bd"][k - 1]["score"] for r in rows]))


def passk(rows, k):
    return float(np.mean([1.0 if any(r["exact"][:k]) else 0.0 for r in rows]))


for name, path in (("TRAIN captions (the 100 it trained on)", TR),
                   ("HELD-OUT captions", TE)):
    d = json.load(open(path))
    rows = d["seq"]
    ks = list(range(1, d["n"] + 1))
    print(f"\n=== {name} — n={len(rows)} ===")
    print("  attempt k :", "".join(f"{k:>8}" for k in ks))
    print("  exact@k   :", "".join(f"{perf_at(rows, k):>8.3f}" for k in ks))
    print("  score@k   :", "".join(f"{score_at(rows, k):>8.3f}" for k in ks))
    print("  pass@k    :", "".join(f"{passk(rows, k):>8.3f}" for k in ks))
    print(f"  chain gain, attempt 1 -> best attempt : "
          f"{perf_at(rows, 1):.3f} -> {passk(rows, ks[-1]):.3f}")

    # How much headroom was there to begin with? A chain cannot repair what is already right.
    wrong_at_1 = [r for r in rows if not r["exact"][0]]
    repaired = [r for r in wrong_at_1 if any(r["exact"][1:])]
    print(f"  wrong at attempt 1: {len(wrong_at_1)}  ->  later repaired: {len(repaired)} "
          f"({len(repaired) / max(1, len(wrong_at_1)):.1%})")
