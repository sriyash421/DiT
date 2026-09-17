"""All three anchored arms, aligned on training step, over the range arm C has reached so far.

WHAT THESE NUMBERS ARE: rollout/draft_exact and rollout/repair_exact are the verifier's EXACT-SCENE
rate -- the fraction of attempts whose critique came back "no update" -- measured on the 100 TRAIN
captions during rollout collection. They are not a loss, and not the partial score. They overstate
held-out ability badly (the model has memorised these captions); their value is the TREND and the
GAP between the two positions.

The arms differ only in how the draft is treated:
  A  beta=1.0, draft gets NO ground truth  -> realised pull toward the frozen base: 100%
  B  beta=0.2, draft also regresses to GT  -> 2b/(1+2b) = 29%
  C  beta=0.5, draft also regresses to GT  -> 2b/(1+2b) = 50%
"""
import pathlib

import numpy as np
import wandb

RUNS = [
    ("A (beta=1.0, pinned)", "on_policy_g6_anchor"),
    ("B (beta=0.2, 29% pull)", "on_policy_g6_anchor_b02"),
    ("C (beta=0.5, 50% pull)", "on_policy_g6_anchor_b05"),
]
BASE = "/gscratch/scrubbed/sriyash/omni-clevr-g6-on-policy"
api = wandb.Api()

series = {}
for label, d in RUNS:
    f = pathlib.Path(BASE) / d / "wandb_run_id.txt"
    if not f.exists():
        print(f"{label}: no wandb id yet")
        continue
    run = api.run(f"sriyash-uw-team/clevr_g6/{f.read_text().strip()}")
    hist = [r for r in run.history(keys=["rollout/draft_exact", "rollout/repair_exact", "_step"],
                                   pandas=False)
            if r.get("rollout/draft_exact") is not None]
    if hist:
        series[label] = (run.state, hist)

if not series:
    raise SystemExit("no rollout history for any arm")

# Compare only where every arm has data -- arm C is the youngest, so it sets the ceiling.
last_common = min(h[-1]["_step"] for _, h in series.values())
print(f"comparing over steps 0 - {last_common} (limited by the youngest arm)\n")

grid = np.linspace(0, last_common, 6).astype(int)
for label, (state, hist) in series.items():
    steps = np.array([r["_step"] for r in hist])
    d = np.array([r["rollout/draft_exact"] for r in hist])
    rp = np.array([r["rollout/repair_exact"] for r in hist])
    pick = [int(np.argmin(np.abs(steps - g))) for g in grid]
    print(f"=== arm {label}   [{state}, latest step {steps[-1]}] ===")
    print("  step  :", "".join(f"{steps[i]:>8}" for i in pick))
    print("  draft :", "".join(f"{d[i]:>8.3f}" for i in pick))
    print("  repair:", "".join(f"{rp[i]:>8.3f}" for i in pick))
    print("  gap   :", "".join(f"{rp[i]-d[i]:>+8.3f}" for i in pick))
    print()

print("the gap (repair - draft) is what anchoring is for: the draft held down while repair rises.")
print("at the last common step:")
for label, (_s, hist) in series.items():
    steps = np.array([r["_step"] for r in hist])
    i = int(np.argmin(np.abs(steps - last_common)))
    d = hist[i]["rollout/draft_exact"]
    rp = hist[i]["rollout/repair_exact"]
    print(f"  arm {label:<24} draft {d:.3f}  repair {rp:.3f}  gap {rp-d:+.3f}")
