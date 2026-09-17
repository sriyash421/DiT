"""Why the component metrics sit near 0.9 while 'exact' sits near 0.5.

'exact' is a conjunction over the whole scene; the components are partial credit averaged over
objects. This checks that story against the actual eval records instead of asserting it.
"""
import json

import numpy as np

F = "/gscratch/socialrl/sriyash/clevr_g6_bon/seqiid/bd_incontext_5000.json"
d = json.load(open(F))

rows = [(a, e) for r in d["seq"] for a, e in zip(r["bd"], r["exact"])]
bd, ex = [a for a, _ in rows], np.array([bool(e) for _, e in rows])
print(f"{len(rows)} graded images (in-context model, feedback chains)\n")

print("mean of each metric, over ALL images:")
for k in ("score", "presence", "shape", "color", "precision"):
    print(f"  {k:<10} {np.mean([a[k] for a in bd]):.3f}")
print(f"  {'exact':<10} {ex.mean():.3f}   <- all-or-nothing")

s = np.array([a["score"] for a in bd])
print(f"\nis exact the same as score == 1.0?")
print(f"  score == 1.0            : {(s >= 1 - 1e-9).mean():.3f}")
print(f"  exact                   : {ex.mean():.3f}")
print(f"  agree                   : {((s >= 1 - 1e-9) == ex).mean():.3f}")

print(f"\nmean score, split by exactness:")
print(f"  exact images            : {s[ex].mean():.3f}")
print(f"  NOT exact images        : {s[~ex].mean():.3f}   <- still high: most of the scene is right")

print("\nhow close are the near-misses?")
for lo, hi in ((0.9, 1.0), (0.8, 0.9), (0.6, 0.8), (0.0, 0.6)):
    m = (s >= lo) & (s < hi)
    if m.sum():
        print(f"  score in [{lo:.1f},{hi:.1f}): {m.sum():5d} images, {ex[m].mean():.3f} of them exact")

# The conjunction: if a scene has n objects each independently right with prob p, the scene is
# exact with prob p^n. Check that the observed exact rate is near score^n for a typical n.
print("\nconjunction check (a scene is exact only if EVERY object is right):")
for n in (2, 3, 4, 5, 6):
    print(f"  if {n} objects each right with p={s.mean():.3f}: p^{n} = {s.mean() ** n:.3f}")
print(f"  observed exact rate: {ex.mean():.3f}")
