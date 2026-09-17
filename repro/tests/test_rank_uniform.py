"""Guard against the collective-mismatch bug that killed both anchor arms.

DDP broadcasts module buffers once per forward, so every rank must run the SAME number of
forwards. The anchored update path splits the batch by chain_pos; if either subset is empty on
some rank and that rank therefore skips a loss, the process group desynchronises and NCCL times
out ten minutes later with no Python traceback. That is expensive to diagnose and trivial to
prevent, so it is checked statically here.
"""
import ast
import sys

SRC = "/gscratch/scrubbed/sriyash/onpolicy-exp/algorithms/on_policy.py"
sys.path.insert(0, "/gscratch/scrubbed/sriyash/onpolicy-exp")
from algorithms.on_policy import _select_rows  # noqa: E402

fail = []

# --- 1. _select_rows still behaves as the branch assumes ----------------------------------------
batch = {"chain_pos": [0, 1, 0, 1], "image": [10, 11, 12, 13], "caption": list("abcd")}
assert _select_rows(batch, []) is None, "_select_rows must return None for an empty selection"
one = _select_rows(batch, [0])
assert one is not None and len(one["chain_pos"]) == 1, "placeholder slice must be a 1-row batch"
print("ok: _select_rows returns None on empty, a 1-row batch on [0]")

# --- 2. neither loss may be called conditionally in the anchored branch --------------------------
tree = ast.parse(open(SRC).read())
update = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "update")

for name in ("rollout_loss", "anchor_loss"):
    calls = [n for n in ast.walk(update)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == name]
    assert calls, f"no call to {name} found in update() -- did it get renamed?"
    # A call sitting inside an IfExp ("x if cond else zeros") is the exact shape of the bug.
    for call in calls:
        for node in ast.walk(update):
            if isinstance(node, ast.IfExp) and any(c is call for c in ast.walk(node)):
                fail.append(f"{name} is called inside a conditional expression -- a rank whose "
                            f"batch lacks those rows would skip a DDP forward")

# --- 3. the zero-scaling path must still exist --------------------------------------------------
src = open(SRC).read()
for token in ("rep_scale", "anc_scale"):
    if token not in src:
        fail.append(f"{token} missing -- the rank-uniform scaling was removed")

if fail:
    print("\nFAILED:")
    for f in fail:
        print("  -", f)
    sys.exit(1)
print("ok: both losses are evaluated unconditionally; empty subsets are zero-scaled, not skipped")
print("\nrank-uniform invariant holds")
