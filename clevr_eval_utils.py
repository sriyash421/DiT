import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import wandb
from datasets_clevr import ClevrContextDataset, context_collate
from train_utils import cfg_get, dataloader_kwargs, rank_is_zero


def build_eval_datasets(cfg, transform):
    eval_split = cfg.eval.split
    datasets = []
    names = []
    for idx, entry in enumerate(cfg.data.dataset_config):
        split = cfg_get(entry, "eval_split", eval_split)
        dataset = ClevrContextDataset(
            entry.dataset_path,
            transform=transform,
            split=split,
            use_disk=cfg_get(cfg.data, "use_disk", True),
            load_meta=cfg_get(cfg.data, "load_meta", True),
            load_images=cfg_get(cfg.data, "load_images", False),
            load_context=cfg_get(cfg.data, "load_context", False),
        )
        datasets.append(dataset)
        names.append(cfg_get(entry, "name", f"dataset_{idx}"))
    return datasets, names


@torch.no_grad()
def evaluate_dataset(model, vae, diffusion, dataset, cfg, device):
    if len(dataset) == 0:
        return None
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=False,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        **dataloader_kwargs(
            cfg.eval.dataloader,
            batch_size=cfg.eval.batch_size,
            shuffle=False,
            sampler=sampler,
            drop_last=False,
            collate_fn=context_collate,
        ),
    )
    total_loss = torch.tensor(0.0, device=device)
    total_count = torch.tensor(0.0, device=device)
    model_dtype = next(model.parameters()).dtype
    max_batches = cfg.eval.max_batches_per_dataset
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x_img = batch["image"].to(device)
        context_tokens = batch["context_tokens"].to(device=device, dtype=model_dtype)
        context_mask = batch["context_mask"].to(device)
        x_latent = vae.encode(x_img).latent_dist.sample().mul_(vae.config.scaling_factor)
        t = torch.randint(0, diffusion.num_timesteps, (x_latent.shape[0],), device=device)
        loss_vec = diffusion.training_losses(
            model,
            x_latent,
            t,
            {"context_tokens": context_tokens, "context_mask": context_mask},
        )["loss"]
        total_loss += loss_vec.sum()
        total_count += loss_vec.numel()
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
    if total_count.item() == 0:
        return None
    return (total_loss / total_count).item()


def run_checkpoint_eval(model, vae, diffusion, eval_datasets, eval_names, cfg, device, logger, train_steps):
    was_training = model.training
    model.eval()
    metrics = {}
    for idx, dataset in enumerate(eval_datasets):
        loss = evaluate_dataset(model, vae, diffusion, dataset, cfg, device)
        if loss is None:
            logger.info(f"Skipping empty eval dataset {idx}: {eval_names[idx]}")
            continue
        metrics[f"eval_loss/dataset_{idx}"] = loss
        metrics[f"eval_loss/{eval_names[idx]}"] = loss
    if rank_is_zero() and metrics:
        wandb.log(metrics, step=train_steps)
        logger.info("Eval losses: " + ", ".join(f"{key}={value:.4f}" for key, value in metrics.items()))
    if was_training:
        model.train()
