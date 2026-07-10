"""Evaluate generation quality: GT|prediction grids and Gemini distance-to-GT on train and val splits."""
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


def evaluate_split(cfg, model, dataset, split, scorer, out_dir):
    seed = cfg.seed if split == "train" else cfg.seed + 1
    batch = select_eval_batch(dataset, seed, cfg.num_samples)

    predictions = []
    for start in tqdm(range(0, cfg.num_samples, cfg.batch_size), desc=f"generate {split}", unit="batch"):
        end = start + cfg.batch_size
        chunk = {"caption": batch["caption"][start:end]}
        if batch["context_tokens"] is not None:
            chunk["context_tokens"] = batch["context_tokens"][start:end]
            chunk["context_mask"] = batch["context_mask"][start:end]
        predictions.extend(
            model.generate(
                chunk,
                num_sampling_steps=cfg.sampling.num_sampling_steps,
                cfg_scale=cfg.sampling.cfg_scale,
                ddim_eta=cfg.sampling.ddim_eta,
                seed=cfg.sample_seed + start + (0 if split == "train" else 10_000),
            )
        )

    grid_path = out_dir / f"{split}_gt_pred.png"
    save_gt_pred_grid(grid_path, batch["gt_images"], predictions)
    distances = score(scorer, batch["caption"], batch["gt_images"], predictions)
    scored = [d for d in distances if d is not None]
    mean_distance = sum(scored) / len(scored) if scored else None
    results = {
        "split": split,
        "ckpt": cfg.ckpt,
        "num_samples": cfg.num_samples,
        "num_scored": len(scored),
        "mean_distance": mean_distance,
        "distances": distances,
        "captions": batch["caption"],
        "grid": str(grid_path),
    }
    write_json(out_dir / f"{split}_results.json", results)
    print(f"{split}: mean_distance={mean_distance} over {len(scored)}/{cfg.num_samples} scored samples")
    return grid_path, mean_distance


@hydra.main(config_path="configs", config_name="eval_generation", version_base=None)
def main(cfg):
    device = 0
    torch.cuda.set_device(device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = hydra.utils.instantiate(cfg.dataset)
    model = hydra.utils.instantiate(cfg.model, context_dim=dataset.context_dim, device=device)
    model.load(cfg.ckpt, use_ema=cfg.use_ema)
    model.net.eval()
    scorer = make_scorer(**cfg.scorer)

    metrics = {}
    grids = {}
    for split in cfg.splits:
        split_dataset = dataset if split == cfg.dataset.split else hydra.utils.instantiate(cfg.dataset, split=split)
        grids[split], metrics[f"eval/{split}_mean_distance"] = evaluate_split(cfg, model, split_dataset, split, scorer, out_dir)

    if cfg.wandb.name is not None:
        wandb.init(project=cfg.wandb.project, name=cfg.wandb.name, config=OmegaConf.to_container(cfg, resolve=True))
        payload = dict(metrics)
        for split, grid_path in grids.items():
            payload[f"eval/{split}_grid"] = wandb.Image(str(grid_path))
        wandb.log(payload)
        wandb.finish()


if __name__ == "__main__":
    main()
