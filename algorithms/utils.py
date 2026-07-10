"""Shared training utilities for all trainers."""
import json
import logging
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm.auto import tqdm


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for param in model.parameters():
        param.requires_grad = flag


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def build_optimizer_scheduler(params, lr, weight_decay, schedule="none", total_steps=0, warmup_steps=0, min_lr=0.0):
    """AdamW plus an optional cosine-with-warmup scheduler. schedule="none" returns scheduler=None."""
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if schedule == "none":
        return opt, None
    if schedule == "cosine":
        warmup_steps = int(warmup_steps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=max(1, int(total_steps) - warmup_steps),
            eta_min=min_lr,
        )
        if warmup_steps == 0:
            return opt, cosine
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return opt, torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_steps])
    raise ValueError(f"Unknown lr schedule: {schedule}")


def diffusion_loss(model, diffusion, x_latent, context_tokens, context_mask):
    """Mean diffusion training loss for one batch of latents and context tokens."""
    t = torch.randint(0, diffusion.num_timesteps, (x_latent.shape[0],), device=x_latent.device)
    model_dtype = next(unwrap_model(model).parameters()).dtype
    loss_vec = diffusion.training_losses(
        model,
        x_latent,
        t,
        {
            "context_tokens": context_tokens.to(dtype=model_dtype),
            "context_mask": context_mask,
        },
    )["loss"]
    return loss_vec.mean()


def tensor_to_pil(x):
    x = x.detach().float().cpu().clamp(-1, 1)
    x = (x + 1) / 2
    arr = (x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def normalized_tensor_to_pil(x):
    x = x.detach().float().cpu().clamp(-1, 1)
    x = ((x + 1) / 2 * 255).round().byte()
    return Image.fromarray(x.permute(1, 2, 0).numpy(), mode="RGB")


def save_trace_grid(path, gt_image, attempt1, feedback, attempt2):
    """Save a GT | first attempt | last attempt strip with the feedback text beside it."""
    from PIL import ImageDraw

    tile_w, tile_h = gt_image.size
    text_w = max(tile_w, 360)
    out = Image.new("RGB", (tile_w * 3 + text_w, tile_h), (255, 255, 255))
    out.paste(gt_image.convert("RGB").resize((tile_w, tile_h)), (0, 0))
    out.paste(attempt1.convert("RGB").resize((tile_w, tile_h)), (tile_w, 0))
    out.paste(attempt2.convert("RGB").resize((tile_w, tile_h)), (tile_w * 2, 0))
    draw = ImageDraw.Draw(out)
    draw.text((tile_w * 3 + 8, 8), feedback[:1000], fill=(20, 20, 20))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def create_logger(log_dir, rank):
    """Rank 0 logs to console and <log_dir>/log.txt; other ranks are silent."""
    logger = logging.getLogger("dit")
    logger.handlers.clear()
    if rank == 0:
        handlers = [logging.StreamHandler()]
        if log_dir is not None:
            handlers.append(logging.FileHandler(f"{log_dir}/log.txt"))
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=handlers,
            force=True,
        )
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def rank_is_zero():
    return dist.get_rank() == 0


def progress_bar(iterable=None, total=None, desc=""):
    """tqdm on rank 0, plain iterable elsewhere."""
    if dist.is_initialized() and dist.get_rank() != 0:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=False)


def save_checkpoint_atomic(checkpoint, checkpoint_path):
    tmp_path = f"{checkpoint_path}.tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, checkpoint_path)


def load_checkpoint(path, use_ema=False):
    """Load a checkpoint file and return the model (or EMA) state dict."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and ("model" in checkpoint or "ema" in checkpoint):
        if use_ema and "ema" in checkpoint:
            return checkpoint["ema"]
        return checkpoint["model"] if "model" in checkpoint else checkpoint["ema"]
    return checkpoint


def dataloader_kwargs(loader_cfg, **kwargs):
    out = dict(kwargs)
    num_workers = int(loader_cfg.num_workers)
    out.update({
        "num_workers": num_workers,
        "pin_memory": bool(loader_cfg.pin_memory),
    })
    if num_workers > 0:
        out["persistent_workers"] = bool(loader_cfg.persistent_workers)
        out["prefetch_factor"] = int(loader_cfg.prefetch_factor)
    return out


def load_jsonl(path):
    rows = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)
