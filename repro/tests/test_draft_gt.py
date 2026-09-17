"""Check the two anchor arms select different rows for the ground-truth term.

Pure logic test -- no model, no GPU. It mirrors update()'s row selection so a wiring mistake shows
up here rather than eight hours into a run.
"""
import sys

sys.path.insert(0, "/gscratch/scrubbed/sriyash/onpolicy-exp")
from algorithms.on_policy import _select_rows  # noqa: E402

# One rollout buffer batch: 4 chains of length 2 -> alternating draft/repair rows.
batch = {"chain_pos": [0, 1, 0, 1, 0, 1, 0, 1],
         "image": list(range(8)),
         "caption": [f"c{i}" for i in range(8)]}

repair = _select_rows(batch, [i for i, p in enumerate(batch["chain_pos"]) if int(p) > 0])
draft = _select_rows(batch, [i for i, p in enumerate(batch["chain_pos"]) if int(p) == 0])

print("all rows   :", batch["chain_pos"])
print("repair rows:", repair["chain_pos"])
print("draft rows :", draft["chain_pos"])
assert repair["chain_pos"] == [1, 1, 1, 1], repair["chain_pos"]
assert draft["chain_pos"] == [0, 0, 0, 0], draft["chain_pos"]

for draft_gt, name in ((False, "arm A (beta=1.0, draft pinned)"),
                       (True, "arm B (beta=0.2, expert loss at step 0)")):
    gt_rows = batch if draft_gt else repair
    n = len(gt_rows["chain_pos"])
    positions = sorted(set(gt_rows["chain_pos"]))
    print(f"{name}: ground-truth term covers {n} rows, chain_pos {positions}")
    if draft_gt:
        assert n == 8 and positions == [0, 1], "arm B must supervise BOTH positions"
    else:
        assert n == 4 and positions == [1], "arm A must supervise repair rows only"

print("\nboth arms select the rows they should")
