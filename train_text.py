#!/usr/bin/env python3
"""Train CLEVR DiT with cached frozen-Qwen context tokens."""
import os
from copy import deepcopy
from glob import glob
from time import time

import hydra
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import wandb
from clevr_eval_utils import build_eval_datasets, run_checkpoint_eval
from clevr_transforms import build_clevr_transform
from datasets_clevr import (
    ClevrContextMultiDataset,
    DistributedWeightedSampler,
    context_collate,
)
from diffusion import create_diffusion
from models import DiT_models
from train_utils import (
    create_logger,
    cfg_get,
    dataloader_kwargs,
    get_lr,
    rank_is_zero,
    requires_grad,
    save_checkpoint_atomic,
    set_optimizer_lr,
    update_ema,
)


def cleanup():
    dist.destroy_process_group()


def load_checkpoint(path):
    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    else:
        from download import find_model
        checkpoint = find_model(path)
    if isinstance(checkpoint, dict) and ("model" in checkpoint or "ema" in checkpoint):
        return checkpoint["model"] if "model" in checkpoint else checkpoint["ema"]
    return checkpoint


@hydra.main(config_path="configs", config_name="train_base", version_base=None)
def main(cfg):
    assert torch.cuda.is_available(), "Training requires at least one GPU."
    dist.init_process_group("nccl")
    assert cfg.train.global_batch_size % dist.get_world_size() == 0, "Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    seed = cfg.train.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    transform = build_clevr_transform(cfg.model.image_size)
    dataset = ClevrContextMultiDataset(
        cfg.data.dataset_config,
        transform=transform,
        split=cfg.data.split,
        use_disk=cfg_get(cfg.data, "use_disk", True),
        load_meta=cfg_get(cfg.data, "load_meta", True),
        load_images=cfg_get(cfg.data, "load_images", False),
        load_context=cfg_get(cfg.data, "load_context", False),
    )
    context_dim = dataset.context_dim
    eval_datasets, eval_names = build_eval_datasets(cfg, transform)

    if rank_is_zero():
        os.makedirs(cfg.train.results_dir, exist_ok=True)
        model_name = cfg.model.name.replace("/", "-")
        experiment_name = cfg.train.experiment_name
        if experiment_name is None:
            experiment_name = f"{len(glob(f'{cfg.train.results_dir}/*')):03d}-{model_name}-qwen"
        experiment_dir = f"{cfg.train.results_dir}/{experiment_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        OmegaConf.save(cfg, f"{experiment_dir}/config.yaml")
        logger.info(f"Experiment directory created at {experiment_dir}")
        wandb.init(project=cfg.wandb.project, name=os.path.basename(experiment_dir), config=resolved_cfg)
    else:
        checkpoint_dir = None
        logger = create_logger(None)

    latent_size = cfg.model.image_size // 8
    model = DiT_models[cfg.model.name](
        input_size=latent_size,
        num_classes=cfg.model.num_classes,
        text_conditioning=True,
        context_dim=context_dim,
        class_dropout_prob=cfg.model.context_dropout_prob,
    )
    if cfg.train.ckpt is not None:
        missing, unexpected = model.load_state_dict(load_checkpoint(cfg.train.ckpt), strict=False)
        logger.info(f"Loaded checkpoint {cfg.train.ckpt}")
        logger.info(f"Missing keys: {missing}")
        logger.info(f"Unexpected keys: {unexpected}")
    else:
        logger.info("Training from scratch.")
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=False)
    diffusion = create_diffusion(timestep_respacing="")
    vae = AutoencoderKL.from_pretrained(cfg.model.vae).to(device)
    vae.eval()
    logger.info(f"Loaded VAE {cfg.model.vae} with scaling_factor={vae.config.scaling_factor}")
    logger.info(f"Context dim: {context_dim}")
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.train.lr, weight_decay=0)
    sampler = DistributedWeightedSampler(
        dataset.sample_weights,
        num_replicas=dist.get_world_size(),
        rank=rank,
        seed=cfg.train.global_seed,
    )
    loader = DataLoader(
        dataset,
        **dataloader_kwargs(
            cfg.dataloader,
            batch_size=cfg.train.global_batch_size // dist.get_world_size(),
            shuffle=False,
            sampler=sampler,
            drop_last=cfg.dataloader.drop_last,
            collate_fn=context_collate,
        ),
    )
    desc = ", ".join(
        f"{entry.dataset_path} (n={len(ds)}, ratio={entry.sampling_ratio})"
        for entry, ds in zip(cfg.data.dataset_config, dataset.datasets)
    )
    logger.info(f"Dataset contains {len(dataset):,} rows from {desc}; split={cfg.data.split}")

    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    train_steps = 0
    log_steps = 0
    running = {key: 0.0 for key in ("loss", "grad_norm", "lr")}
    running_sources = torch.zeros(len(dataset.datasets), dtype=torch.float64, device=device)
    running_samples = 0
    start_time = time()
    done_training = False
    logger.info(f"Training for {cfg.train.epochs} epochs...")

    for epoch in range(cfg.train.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for batch in loader:
            x_img = batch["image"].to(device)
            context_tokens = batch["context_tokens"].to(device=device, dtype=next(model.parameters()).dtype)
            context_mask = batch["context_mask"].to(device)
            source_index = batch["source_index"].to(device)
            with torch.no_grad():
                x_latent = vae.encode(x_img).latent_dist.sample().mul_(vae.config.scaling_factor)
            t = torch.randint(0, diffusion.num_timesteps, (x_latent.shape[0],), device=device)
            loss_vec = diffusion.training_losses(
                model,
                x_latent,
                t,
                {"context_tokens": context_tokens, "context_mask": context_mask},
            )["loss"]
            loss = loss_vec.mean()

            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            lr = get_lr(cfg, train_steps)
            set_optimizer_lr(opt, lr)
            opt.step()
            update_ema(ema, model.module, decay=cfg.train.ema_decay)

            running["loss"] += loss.item()
            running["grad_norm"] += grad_norm.item()
            running["lr"] += lr
            running_sources += torch.bincount(source_index, minlength=len(dataset.datasets)).to(running_sources.dtype)
            running_samples += int(source_index.numel())
            log_steps += 1
            train_steps += 1

            if train_steps % cfg.train.log_every == 0:
                torch.cuda.synchronize()
                steps_per_sec = log_steps / (time() - start_time)
                avg_loss = torch.tensor(running["loss"] / log_steps, device=device)
                avg_grad = torch.tensor(running["grad_norm"] / log_steps, device=device)
                avg_lr = torch.tensor(running["lr"] / log_steps, device=device)
                source_counts = running_sources.clone()
                sample_count = torch.tensor(float(running_samples), device=device)
                for value in (avg_loss, avg_grad, avg_lr, source_counts, sample_count):
                    dist.all_reduce(value, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                avg_grad = avg_grad.item() / dist.get_world_size()
                avg_lr = avg_lr.item() / dist.get_world_size()
                source_fracs = (source_counts / sample_count.clamp_min(1)).detach().cpu().tolist()
                logger.info(
                    f"(step={train_steps:07d}) Loss: {avg_loss:.4f}, "
                    f"Grad Norm: {avg_grad:.4f}, LR: {avg_lr:.6g}, Steps/Sec: {steps_per_sec:.2f}"
                )
                if rank_is_zero():
                    log_payload = {
                        "train/loss": avg_loss,
                        "train/grad_norm": avg_grad,
                        "train/lr": avg_lr,
                        "train/steps_per_sec": steps_per_sec,
                        "train/epoch": epoch,
                    }
                    for idx, frac in enumerate(source_fracs):
                        log_payload[f"train/source_fraction/dataset_{idx}"] = frac
                        log_payload[f"train/source_fraction/{dataset.names[idx]}"] = frac
                    wandb.log(log_payload, step=train_steps)
                running = {key: 0.0 for key in running}
                running_sources.zero_()
                running_samples = 0
                log_steps = 0
                start_time = time()

            if train_steps % cfg.train.ckpt_every == 0 and train_steps > 0:
                run_checkpoint_eval(model, vae, diffusion, eval_datasets, eval_names, cfg, device, logger, train_steps)
                if rank_is_zero():
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "config": resolved_cfg,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    save_checkpoint_atomic(checkpoint, checkpoint_path)
                    save_checkpoint_atomic(ema.state_dict(), f"{checkpoint_dir}/{train_steps:07d}-ema.pt")
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

            if cfg.train.max_train_steps is not None and train_steps >= cfg.train.max_train_steps:
                done_training = True
                break
        if done_training:
            break

    logger.info("Done!")
    if rank_is_zero():
        wandb.finish()
    cleanup()


if __name__ == "__main__":
    main()
