import wandb
from pathlib import Path
F = Path("/gscratch/socialrl/sriyash/clevr_g6_bon/figs")
run = wandb.init(project="clevr_g6", entity="sriyash-uw-team", id="m2hasqdr", resume="must")
wandb.log({f"report/{p.stem}": wandb.Image(str(p)) for p in sorted(F.glob("fig*.png"))})
print("logged:", *[p.stem for p in sorted(F.glob("fig*.png"))], flush=True)
wandb.finish()
