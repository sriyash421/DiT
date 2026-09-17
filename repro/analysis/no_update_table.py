"""How much of the control run's rollout buffer was a "no update" transition, by chain position?

A record stores the history that produced its attempt. The critique that CONDITIONS a record is the
last entry of its feedback_history; a record at chain position 0 has none. When that critique is
"no update" the row teaches the model to reproduce an image that was already correct -- a copy
lesson, not a repair lesson. As the draft improves, those rows crowd out the informative ones.

rows    = outer step (the rollout buffer collected at that step)
columns = chain position of the record (0 = draft, 1..3 = repairs)
cell    = % of records at that position whose CONDITIONING critique was "no update"

Position 0 is always blank: a draft has no critique conditioning it.
"""
import glob
import os

import numpy as np
import torch

ROOT = "/gscratch/scrubbed/sriyash/rollouts/g6_tiny_d250"


def is_no_update(text):
    return str(text).strip().lower().rstrip(".") == "no update"


steps = sorted(glob.glob(os.path.join(ROOT, "step_*")))
rows = []
for sd in steps:
    step = int(os.path.basename(sd).split("_")[1])
    by_pos = {}
    for shard in sorted(glob.glob(os.path.join(sd, "rank_*", "records.pt"))):
        try:
            blob = torch.load(shard, map_location="cpu", weights_only=False)
            recs = blob["records"] if isinstance(blob, dict) else blob
        except Exception as exc:
            print(f"  ! {shard}: {type(exc).__name__}")
            continue
        for r in recs:
            hist = r.get("feedback_history") or []
            pos = len(hist)                      # position in the chain == how many critiques seen
            if pos == 0:
                by_pos.setdefault(0, []).append(None)
                continue
            by_pos.setdefault(pos, []).append(1.0 if is_no_update(hist[-1]) else 0.0)
    rows.append((step, by_pos))

if not rows:
    raise SystemExit("no records found")

maxpos = max(p for _s, bp in rows for p in bp)
print(f"control run: {len(rows)} outer steps, chain positions 0..{maxpos}")
print("cell = % of records at that position conditioned on a \"no update\" critique\n")
hdr = "".join(f"{'pos ' + str(p):>10}" for p in range(maxpos + 1))
print(f"{'outer':>6}{hdr}{'buffer':>9}")

show = [r for i, r in enumerate(rows) if i % 5 == 0 or i == len(rows) - 1]
for step, bp in show:
    cells = ""
    total = 0
    for p in range(maxpos + 1):
        v = bp.get(p, [])
        total += len(v)
        if p == 0:
            cells += f"{'-':>10}"
        else:
            vals = [x for x in v if x is not None]
            cells += f"{np.mean(vals) * 100:>9.1f}%" if vals else f"{'-':>10}"
    print(f"{step:>6}{cells}{total:>9}")

# The headline: across the whole buffer, how much of it is a copy lesson?
print()
for step, bp in [rows[0], rows[len(rows) // 2], rows[-1]]:
    vals = [x for p, v in bp.items() if p > 0 for x in v if x is not None]
    n_all = sum(len(v) for v in bp.values())
    if vals:
        print(f"outer {step:>3}: {np.mean(vals)*100:5.1f}% of REPAIR rows are \"no update\" "
              f"({len(vals)} repair rows of {n_all} total in buffer)")
