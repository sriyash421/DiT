"""Single training entry point: builds the dataset, model, and trainer from the config and runs it."""
import random
from pathlib import Path

import hydra
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from omegaconf import OmegaConf

import wandb


def generate_context_cache(cfg, device):
    """Precompute stage-3 context tokens for frozen-encoder training when the dataset lacks them."""
    if "context_encoder" not in cfg.model:
        return
    encoder_cfg = cfg.model.context_encoder
    if not encoder_cfg.freeze_encoder or not cfg.dataset.load_context:
        return
    from datasets.clevr.encode_context import encode_context_tokens
    from datasets.clevr.utils import zarr_has_context

    if dist.get_rank() == 0:
        for entry in cfg.dataset.datasets:
            if zarr_has_context(entry.path):
                continue
            print(f"Precomputing context tokens for {entry.path} with {encoder_cfg.model_id}...")
            encode_context_tokens(
                entry.path,
                encoder_cfg.model_id,
                device,
                max_context_len=encoder_cfg.max_length,
                vlm_dtype=encoder_cfg.dtype,
            )
    dist.barrier()


def make_experiment_dir(cfg):
    experiment_dir = Path(cfg.results_dir) / cfg.experiment_name
    if dist.get_rank() == 0:
        (experiment_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, experiment_dir / "config.yaml")
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    return experiment_dir


@hydra.main(config_path="configs", config_name="train_base", version_base=None)
def main(cfg):
    assert torch.cuda.is_available(), "Training requires at least one GPU."
    dist.init_process_group("nccl")
    assert cfg.trainer.global_batch_size % dist.get_world_size() == 0, "Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    seed = cfg.trainer.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    random.seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    generate_context_cache(cfg, device)
    dataset = hydra.utils.instantiate(cfg.dataset)
    model = hydra.utils.instantiate(cfg.model, context_dim=dataset.context_dim, device=device)
    if cfg.ckpt is not None:
        missing, unexpected = model.load(cfg.ckpt)
        print(f"Loaded checkpoint {cfg.ckpt} (missing={len(missing)}, unexpected={len(unexpected)})")

    log_dir = make_experiment_dir(cfg)
    model.ddp(device)

    start_step = 0
    if "pretrain" in cfg:
        pretrainer = hydra.utils.instantiate(
            cfg.pretrain, model=model, dataset=dataset, device=device, log_dir=str(log_dir)
        )
        pretrainer.learn()
        start_step = pretrainer.train_steps

    trainer = hydra.utils.instantiate(
        cfg.trainer, model=model, dataset=dataset, device=device, log_dir=str(log_dir), start_step=start_step
    )
    trainer.learn()

    if rank == 0:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
