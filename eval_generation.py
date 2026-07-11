"""Evaluate generation quality: GT|prediction grids and Gemini distance-to-GT on train and val splits.

The model and dataset are rebuilt strictly from the training run's saved config.yaml (found one
level above the checkpoint's `checkpoints/` dir), so encoder freeze/LoRA/arch and the data source
always match the checkpoint being evaluated. Everything else is plain argparse."""
import argparse
from pathlib import Path

import hydra
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from omegaconf import OmegaConf
from tqdm import tqdm

import wandb
from algorithms.eval import save_gt_pred_grid, select_eval_batch
from algorithms.utils import write_json
from verifiers.eval_metrics import make_scorer, score


def resolve_run(run_dir, step):
    """Resolve a run directory + step into (training config, full checkpoint path).

    step=-1 picks the latest checkpoint. Checkpoints live at <run_dir>/checkpoints/<step:07d>.pt
    (the full checkpoint holding model/ema/context_encoder); the training config is <run_dir>/config.yaml.
    """
    run_dir = Path(run_dir)
    cfg_path = run_dir / "config.yaml"
    assert cfg_path.exists(), f"No training config at {cfg_path}; is {run_dir} a training run dir?"
    ckpt_dir = run_dir / "checkpoints"
    if step < 0:
        steps = sorted(int(p.stem) for p in ckpt_dir.glob("*.pt") if not p.stem.endswith("-ema"))
        assert steps, f"No checkpoints found in {ckpt_dir}."
        step = steps[-1]
    ckpt = ckpt_dir / f"{step:07d}.pt"
    assert ckpt.exists(), f"Checkpoint {ckpt} does not exist."
    print(f"Run {run_dir.name}: config={cfg_path.name}, ckpt={ckpt.name}")
    return OmegaConf.load(cfg_path), str(ckpt)


def evaluate_split(args, model, dataset, split, scorer, out_dir):
    seed = args.seed if split == "train" else args.seed + 1
    batch = select_eval_batch(dataset, seed, args.num_samples)

    predictions = []
    for start in tqdm(range(0, args.num_samples, args.batch_size), desc=f"generate {split}", unit="batch"):
        end = start + args.batch_size
        chunk = {"caption": batch["caption"][start:end]}
        if batch["context_tokens"] is not None:
            chunk["context_tokens"] = batch["context_tokens"][start:end]
            chunk["context_mask"] = batch["context_mask"][start:end]
        predictions.extend(
            model.generate(
                chunk,
                num_sampling_steps=args.num_sampling_steps,
                cfg_scale=args.cfg_scale,
                ddim_eta=args.ddim_eta,
                seed=args.sample_seed + start + (0 if split == "train" else 10_000),
            )
        )

    grid_path = out_dir / f"{split}_gt_pred.png"
    save_gt_pred_grid(grid_path, batch["gt_images"], predictions)
    distances = score(scorer, batch["caption"], batch["gt_images"], predictions)
    scored = [d for d in distances if d is not None]
    mean_distance = sum(scored) / len(scored) if scored else None
    results = {
        "split": split,
        "ckpt": args.ckpt,
        "num_samples": args.num_samples,
        "num_scored": len(scored),
        "mean_distance": mean_distance,
        "distances": distances,
        "captions": batch["caption"],
        "grid": str(grid_path),
    }
    write_json(out_dir / f"{split}_results.json", results)
    print(f"{split}: mean_distance={mean_distance} over {len(scored)}/{args.num_samples} scored samples")
    return grid_path, mean_distance


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True, help="Training run dir (holds config.yaml and checkpoints/).")
    p.add_argument("--step", type=int, default=-1, help="Checkpoint step to load; -1 for the latest.")
    p.add_argument("--out_dir", default=None,
                   help="Output dir; defaults to results/eval_<run_dir name>_ckpt<step>/.")
    p.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True,
                   help="Use the EMA weights inside the checkpoint (--no-use_ema for raw model).")
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=25)
    p.add_argument("--seed", type=int, default=0, help="Seed for selecting the eval batch.")
    p.add_argument("--sample_seed", type=int, default=1234, help="Seed for the sampling noise.")
    p.add_argument("--num_sampling_steps", type=int, default=100)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--ddim_eta", type=float, default=0.0)
    p.add_argument("--wandb_project", default="DiT-qwen-clevr")
    p.add_argument("--wandb_name", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = 0
    torch.cuda.set_device(device)

    train_cfg, ckpt = resolve_run(args.run_dir, args.step)
    args.ckpt = ckpt

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path("results") / f"eval_{Path(args.run_dir).name}_ckpt{Path(ckpt).stem}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to {out_dir}")
    build_dataset = lambda split: hydra.utils.instantiate(train_cfg.dataset, split=split)

    dataset = build_dataset(train_cfg.dataset.split)
    model = hydra.utils.instantiate(train_cfg.model, context_dim=dataset.context_dim, device=device)
    model.load(ckpt, use_ema=args.use_ema)
    model.net.eval()
    scorer = make_scorer()

    metrics = {}
    grids = {}
    for split in args.splits:
        split_dataset = dataset if split == train_cfg.dataset.split else build_dataset(split)
        grids[split], metrics[f"eval/{split}_mean_distance"] = evaluate_split(
            args, model, split_dataset, split, scorer, out_dir
        )

    if args.wandb_name is not None:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
        payload = dict(metrics)
        for split, grid_path in grids.items():
            payload[f"eval/{split}_grid"] = wandb.Image(str(grid_path))
        wandb.log(payload)
        wandb.finish()


if __name__ == "__main__":
    main()
