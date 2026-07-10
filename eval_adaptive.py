"""Evaluate the adaptive feedback rollout: sample, get verifier feedback, re-encode, and resample."""
from pathlib import Path

import hydra
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from omegaconf import OmegaConf

import wandb
from algorithms.eval import adaptive_rollout, distance_metrics, select_eval_batch
from algorithms.on_policy import PolicySampler
from algorithms.utils import save_trace_grid, write_json
from datasets.clevr.utils import load_metadata_for_zarr, metadata_by_index
from diffusion import create_diffusion
from verifiers.eval_metrics import make_scorer


@hydra.main(config_path="configs", config_name="eval_adaptive", version_base=None)
def main(cfg):
    device = 0
    torch.cuda.set_device(device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = hydra.utils.instantiate(cfg.dataset, split=cfg.split)
    model = hydra.utils.instantiate(cfg.model, context_dim=dataset.context_dim, device=device)
    model.load(cfg.ckpt, use_ema=cfg.use_ema)
    model.net.eval()
    verifier = hydra.utils.instantiate(cfg.verifier)
    # A model with its own (finetuned) encoder must be evaluated with that encoder,
    # not a fresh frozen one from the config.
    if model.encoder is not None:
        context_encoder = model.encoder
    else:
        context_encoder = hydra.utils.instantiate(cfg.context_encoder)
    scorer = make_scorer(**cfg.scorer)
    sampler = PolicySampler(
        create_diffusion(str(cfg.sampling.num_sampling_steps)),
        latent_size=model.latent_size,
        vae_scaling_factor=model.vae.config.scaling_factor,
        cfg_scale=cfg.sampling.cfg_scale,
        ddim_eta=cfg.sampling.ddim_eta,
    )

    metadata_rows = load_metadata_for_zarr(dataset.datasets[0].root)
    batch = select_eval_batch(dataset, cfg.caption_seed, cfg.num_captions)
    if model.encoder is not None:
        with torch.no_grad():
            batch["context_tokens"], batch["context_mask"] = model.encoder.encode_history(
                batch["caption"], [[] for _ in batch["caption"]], [[] for _ in batch["caption"]]
            )
    metadata = metadata_by_index(metadata_rows, batch["metadata_index"].tolist())

    traces, histories, token_count = adaptive_rollout(
        model.net,
        model.vae,
        sampler,
        verifier,
        context_encoder,
        batch,
        metadata,
        batch["gt_images"],
        steps=cfg.steps,
        seed=cfg.seed,
        scorer=scorer,
    )

    for idx, trace in enumerate(traces):
        if not trace:
            continue
        save_trace_grid(
            out_dir / f"trace_{idx:03d}.png",
            batch["gt_images"][idx],
            trace[0]["image"],
            "\n".join(histories[idx]),
            trace[-1]["image"],
        )
        for step, entry in enumerate(trace):
            entry["image"].save(out_dir / f"caption_{idx:03d}_step_{step:02d}.png")

    metrics = distance_metrics(traces)
    results = {
        "ckpt": cfg.ckpt,
        "split": cfg.split,
        "steps": cfg.steps,
        "num_captions": cfg.num_captions,
        "verifier_tokens": token_count,
        "metrics": metrics,
        "captions": batch["caption"],
        "feedback_histories": histories,
        "distances": [[entry["distance"] for entry in trace] for trace in traces],
    }
    write_json(out_dir / "results.json", results)
    print(f"Adaptive eval over {cfg.num_captions} captions x {cfg.steps} steps: {metrics}")

    if cfg.wandb.name is not None:
        wandb.init(project=cfg.wandb.project, name=cfg.wandb.name, config=OmegaConf.to_container(cfg, resolve=True))
        payload = {f"eval/{key}": value for key, value in metrics.items()}
        for idx in range(len(traces)):
            grid_path = out_dir / f"trace_{idx:03d}.png"
            if grid_path.exists():
                payload[f"eval/trace_{idx:03d}"] = wandb.Image(str(grid_path))
        wandb.log(payload)
        wandb.finish()


if __name__ == "__main__":
    main()
