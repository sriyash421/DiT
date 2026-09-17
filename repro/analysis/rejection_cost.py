"""What does it cost to keep sampling until the buffer has at most X% "no update" rows?

Rejection sampling keeps the buffer full and mostly informative, but the yield of informative rows
falls as the draft improves -- so the generation needed to fill one buffer RISES over training,
fastest exactly when the method is working. This prices that out against the control run's measured
rates rather than guessing.

Control run per outer loop: 100 chains x length 4 = 400 generations, 300 of them repair rows.
"""
import glob
import os

import numpy as np
import torch

ROOT = "/gscratch/scrubbed/sriyash/rollouts/g6_tiny_d250"
CHAINS = 100          # chains per outer loop in the control
LEN = 4               # chain length -> 3 repair rows per chain
REPAIR_PER_CHAIN = LEN - 1
TARGET_REPAIR = CHAINS * REPAIR_PER_CHAIN      # 300 repair rows to fill the buffer


def is_no_update(t):
    return str(t).strip().lower().rstrip(".") == "no update"


rates = []
for sd in sorted(glob.glob(os.path.join(ROOT, "step_*"))):
    step = int(os.path.basename(sd).split("_")[1])
    vals = []
    for shard in sorted(glob.glob(os.path.join(sd, "rank_*", "records.pt"))):
        blob = torch.load(shard, map_location="cpu", weights_only=False)
        for r in (blob["records"] if isinstance(blob, dict) else blob):
            h = r.get("feedback_history") or []
            if h:
                vals.append(1.0 if is_no_update(h[-1]) else 0.0)
    if vals:
        rates.append((step, float(np.mean(vals))))

print("cost of rejection sampling to a capped \"no update\" share")
print(f"baseline: {CHAINS} chains x {LEN} attempts = {CHAINS*LEN} generations per outer loop\n")
print(f"{'outer':>6}{'no-upd':>9}{'informative':>13}" + "".join(f"{'cap ' + str(c) + '%':>12}" for c in (25, 50)))

for step, nu in rates:
    if step % 10 and step != rates[-1][0]:
        continue
    p = 1.0 - nu                               # informative fraction of repair rows
    cells = ""
    for cap in (0.25, 0.50):
        need_info = TARGET_REPAIR * (1 - cap)  # informative rows the buffer must contain
        if p <= 1e-9:
            cells += f"{'inf':>12}"
            continue
        chains = need_info / (REPAIR_PER_CHAIN * p)
        # Capped share can also be filled with the no-update rows that arrive for free.
        chains = max(chains, CHAINS)
        cells += f"{chains * LEN / (CHAINS * LEN):>11.1f}x"
    print(f"{step:>6}{nu*100:>8.1f}%{p*100:>12.1f}%{cells}")

print("\n'x' = generations per outer loop relative to the control run.")
print("The factor rises with training: the better the draft, the rarer a real edit becomes.")
