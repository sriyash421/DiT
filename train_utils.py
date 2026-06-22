import logging
import math
import os
from collections import OrderedDict

import torch
import torch.distributed as dist


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for param in model.parameters():
        param.requires_grad = flag


def cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    return cfg.get(key, default)


def get_lr(cfg, step):
    train = cfg.train
    if train.lr_schedule == "constant":
        return train.lr
    warmup_steps = max(0, int(train.lr_warmup_steps))
    total_steps = max(1, train.max_train_steps if train.max_train_steps is not None else train.epochs)
    min_lr = train.min_lr if train.min_lr is not None else train.lr * train.min_lr_ratio
    if warmup_steps > 0 and step < warmup_steps:
        return train.lr * float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (train.lr - min_lr) * cosine


def set_optimizer_lr(opt, lr):
    for group in opt.param_groups:
        group["lr"] = lr


def create_logger(logging_dir):
    logger = logging.getLogger(__name__)
    logger.handlers.clear()
    if dist.get_rank() == 0:
        handlers = [logging.StreamHandler()]
        if logging_dir is not None:
            handlers.append(logging.FileHandler(f"{logging_dir}/log.txt"))
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


def save_checkpoint_atomic(checkpoint, checkpoint_path):
    tmp_path = f"{checkpoint_path}.tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, checkpoint_path)


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


def rank_is_zero():
    return dist.get_rank() == 0
