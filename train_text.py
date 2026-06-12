"""
Fine-tune DiT with cached FLAN-T5 caption embeddings.
"""
import argparse
import logging
import math
import os
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time

import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from datasets_clevr import ClevrTextEmbeddingDataset, WeightedDatasetList, adaptive_collate
from diffusion import create_diffusion
from download import find_model
from models import DiT_models

import wandb

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    dist.destroy_process_group()


def get_lr(args, step):
    if args.lr_schedule == "constant":
        return args.lr
    warmup_steps = max(0, args.lr_warmup_steps)
    total_steps = args.max_train_steps if args.max_train_steps is not None else args.epochs
    total_steps = max(1, total_steps)
    min_lr = args.min_lr if args.min_lr is not None else args.lr * args.min_lr_ratio
    if warmup_steps > 0 and step < warmup_steps:
        return args.lr * float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (args.lr - min_lr) * cosine


def set_optimizer_lr(opt, lr):
    for group in opt.param_groups:
        group["lr"] = lr


def create_logger(logging_dir):
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def save_checkpoint_atomic(checkpoint, checkpoint_path):
    tmp_path = f"{checkpoint_path}.tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, checkpoint_path)


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, "Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        model_string_name = args.model.replace("/", "-")
        if args.experiment_name is None:
            experiment_index = len(glob(f"{args.results_dir}/*"))
            experiment_name = f"{experiment_index:03d}-{model_string_name}-text"
        else:
            experiment_name = args.experiment_name
        experiment_dir = f"{args.results_dir}/{experiment_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        wandb.init(
            project=args.wandb_project,
            name=os.path.basename(experiment_dir),
            config=vars(args),
        )
    else:
        logger = create_logger(None)

    assert args.image_size % 8 == 0, "Image size must be divisible by 8."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        text_conditioning=True,
        text_embed_dim=args.text_embed_dim,
        max_text_len=args.max_text_len,
    )
    if not args.from_scratch:
        ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
        state_dict = find_model(ckpt_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded checkpoint {ckpt_path}")
        logger.info(f"Missing keys: {missing}")
        logger.info(f"Unexpected keys: {unexpected}")
    elif args.ckpt is not None:
        state_dict = find_model(args.ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded checkpoint {args.ckpt}")
        logger.info(f"Missing keys: {missing}")
        logger.info(f"Unexpected keys: {unexpected}")
    else:
        logger.info("Training from scratch.")
    requires_grad(model.y_embedder, False)

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)
    diffusion = create_diffusion(timestep_respacing="")
    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    vae_scaling_factor = vae.config.scaling_factor
    logger.info(f"Loaded VAE {args.vae} with scaling_factor={vae_scaling_factor}")
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0)

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    data_paths = args.data_paths if args.data_paths is not None else [args.data_path]
    if data_paths is None or any(path is None for path in data_paths):
        raise ValueError("Provide --data-path or --data-paths")
    sampling_ratios = args.sampling_ratios
    if sampling_ratios is None:
        sampling_ratios = [0.25, 0.75] if len(data_paths) == 2 else [1.0] * len(data_paths)
    if len(sampling_ratios) != len(data_paths):
        raise ValueError("--sampling-ratios must match --data-paths length")
    datasets = [ClevrTextEmbeddingDataset(path, transform=transform, split=args.split) for path in data_paths]
    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = WeightedDatasetList(
            datasets,
            sampling_ratios,
            virtual_epoch_size=args.virtual_epoch_size,
            seed=args.global_seed,
        )
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=args.global_seed)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=adaptive_collate,
    )
    dataset_desc = ", ".join(f"{path} (n={len(ds)}, ratio={ratio})" for path, ds, ratio in zip(data_paths, datasets, sampling_ratios))
    logger.info(f"Dataset contains {len(dataset):,} virtual rows from {dataset_desc}; split={args.split}")

    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    train_steps = 0
    log_steps = 0
    running_loss = 0
    running_base_loss = 0
    running_feedback_loss = 0
    running_base_batches = 0
    running_feedback_batches = 0
    running_feedback_fraction = 0
    running_grad_norm = 0
    running_lr = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    done_training = False
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for batch in loader:
            loss_terms = []
            base_loss_value = None
            feedback_loss_value = None
            base_count = 0
            feedback_count = 0

            def group_loss(group, adaptive=False):
                x_img = group["image"].to(device)
                with torch.no_grad():
                    x_latent = vae.encode(x_img).latent_dist.sample().mul_(vae_scaling_factor)
                t = torch.randint(0, diffusion.num_timesteps, (x_latent.shape[0],), device=device)
                kwargs = dict(
                    text_tokens=group["text_tokens"].to(device),
                    text_mask=group["text_mask"].to(device),
                    text_pooled=group["text_pooled"].to(device),
                )
                if adaptive:
                    kwargs.update(dict(
                        feedback_tokens=group["feedback_tokens"].to(device),
                        feedback_mask=group["feedback_mask"].to(device),
                        feedback_pooled=group["feedback_pooled"].to(device),
                        attempt_latent=group["attempt_latent"].to(device),
                    ))
                    if args.feedback_caption_dropout > 0:
                        kwargs["drop_caption"] = (
                            torch.rand(x_latent.shape[0], device=device) < args.feedback_caption_dropout
                        )
                return diffusion.training_losses(model, x_latent, t, kwargs)["loss"]

            if batch["base"] is not None:
                base_losses = group_loss(batch["base"], adaptive=False)
                loss_terms.append(base_losses)
                base_loss_value = base_losses.mean()
                base_count = base_losses.shape[0]
            if batch["feedback"] is not None:
                feedback_losses = group_loss(batch["feedback"], adaptive=True)
                loss_terms.append(feedback_losses)
                feedback_loss_value = feedback_losses.mean()
                feedback_count = feedback_losses.shape[0]
            if not loss_terms:
                continue
            loss = torch.cat(loss_terms).mean()

            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            lr = get_lr(args, train_steps)
            set_optimizer_lr(opt, lr)
            opt.step()
            update_ema(ema, model.module, decay=args.ema_decay)

            total_count = base_count + feedback_count
            running_loss += loss.item()
            if base_loss_value is not None:
                running_base_loss += base_loss_value.item()
                running_base_batches += 1
            if feedback_loss_value is not None:
                running_feedback_loss += feedback_loss_value.item()
                running_feedback_batches += 1
            running_feedback_fraction += feedback_count / max(1, total_count)
            running_grad_norm += grad_norm.item()
            running_lr += lr
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_base_loss = torch.tensor(running_base_loss / max(1, running_base_batches), device=device)
                avg_feedback_loss = torch.tensor(running_feedback_loss / max(1, running_feedback_batches), device=device)
                avg_feedback_fraction = torch.tensor(running_feedback_fraction / log_steps, device=device)
                avg_grad_norm = torch.tensor(running_grad_norm / log_steps, device=device)
                avg_lr = torch.tensor(running_lr / log_steps, device=device)
                for value in (avg_loss, avg_base_loss, avg_feedback_loss, avg_feedback_fraction, avg_grad_norm, avg_lr):
                    dist.all_reduce(value, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                avg_base_loss = avg_base_loss.item() / dist.get_world_size()
                avg_feedback_loss = avg_feedback_loss.item() / dist.get_world_size()
                avg_feedback_fraction = avg_feedback_fraction.item() / dist.get_world_size()
                avg_grad_norm = avg_grad_norm.item() / dist.get_world_size()
                avg_lr = avg_lr.item() / dist.get_world_size()
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, "
                    f"Base Loss: {avg_base_loss:.4f}, Feedback Loss: {avg_feedback_loss:.4f}, "
                    f"Feedback Frac: {avg_feedback_fraction:.2f}, Grad Norm: {avg_grad_norm:.4f}, "
                    f"LR: {avg_lr:.6g}, "
                    f"Train Steps/Sec: {steps_per_sec:.2f}"
                )
                if rank == 0:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/loss_base": avg_base_loss,
                        "train/loss_feedback": avg_feedback_loss,
                        "train/feedback_fraction": avg_feedback_fraction,
                        "train/image_context_fraction": avg_feedback_fraction,
                        "train/grad_norm": avg_grad_norm,
                        "train/lr": avg_lr,
                        "train/steps_per_sec": steps_per_sec,
                        "train/epoch": epoch,
                    }, step=train_steps)
                running_loss = 0
                running_base_loss = 0
                running_feedback_loss = 0
                running_base_batches = 0
                running_feedback_batches = 0
                running_feedback_fraction = 0
                running_grad_norm = 0
                running_lr = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    save_checkpoint_atomic(checkpoint, checkpoint_path)
                    sample_checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}-ema.pt"
                    save_checkpoint_atomic(ema.state_dict(), sample_checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    logger.info(f"Saved EMA checkpoint to {sample_checkpoint_path}")
                dist.barrier()

            if args.max_train_steps is not None and train_steps >= args.max_train_steps:
                done_training = True
                break
        if done_training:
            break

    logger.info("Done!")
    if rank == 0:
        wandb.finish()
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--data-paths", nargs="+", default=None)
    parser.add_argument("--sampling-ratios", nargs="+", type=float, default=None)
    parser.add_argument("--virtual-epoch-size", type=int, default=None)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, default="stabilityai/sdxl-vae")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10_000)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-schedule", type=str, default="constant", choices=["constant", "cosine"])
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--feedback-caption-dropout", type=float, default=0.0, help="For feedback rows only, force-drop caption tokens/pool with this probability while keeping feedback and image context.")
    parser.add_argument("--text-embed-dim", type=int, default=1024)
    parser.add_argument("--max-text-len", type=int, default=128)
    parser.add_argument("--wandb-project", type=str, default="DiT-text")
    main(parser.parse_args())
