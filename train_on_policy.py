#!/usr/bin/env python3
"""On-policy feedback distillation for Qwen-conditioned CLEVR DiT."""
import os
from copy import deepcopy
from glob import glob
from pathlib import Path
from time import time

import hydra
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

import wandb
from clevr_transforms import build_clevr_transform
from datasets_clevr import ClevrContextDataset, context_collate
from diffusion import create_diffusion
from feedback_verifiers import GeminiVerifier, build_feedback_verifier
from models import DiT_models
from on_policy import (
    OnPolicyTrainer,
    PolicySampler,
    QwenContextEncoder,
    RolloutBuffer,
    RolloutCollector,
    diffusion_loss,
    rollout_collate,
    load_metadata_for_zarr,
    metadata_by_index,
    save_trace_grid,
    unwrap_model,
)
from train_text import load_checkpoint
from train_utils import (
    create_logger,
    dataloader_kwargs,
    rank_is_zero,
    requires_grad,
    save_checkpoint_atomic,
    update_ema,
)


def cleanup():
    dist.destroy_process_group()


def build_verifier(cfg):
    name = cfg.verifier.get("name", "gemini")
    if name in {"gemini", "gemini-native"}:
        api_key = os.getenv(cfg.verifier.api_key_env)
        return GeminiVerifier(
            api_key=api_key,
            model=cfg.verifier.model,
            api_url=cfg.verifier.api_url,
            temperature=cfg.verifier.temperature,
            max_tokens=cfg.verifier.max_tokens,
            retries=cfg.verifier.retries,
            timeout=cfg.verifier.timeout,
            workers=cfg.verifier.workers,
        )
    if name in {"qwen-vllm", "openai-chat"}:
        return build_feedback_verifier(
            backend=name,
            model=cfg.verifier.model,
            api_url=cfg.verifier.api_url,
            api_key=cfg.verifier.get("api_key", "EMPTY"),
            temperature=cfg.verifier.temperature,
            max_tokens=cfg.verifier.max_tokens,
            retries=cfg.verifier.retries,
            timeout=cfg.verifier.timeout,
            workers=cfg.verifier.workers,
            enable_thinking=cfg.verifier.get("enable_thinking", False),
            include_caption=cfg.verifier.get("include_caption", True),
            include_metadata=cfg.verifier.get("include_metadata", False),
            image_size=cfg.verifier.get("image_size", None),
        )
    raise ValueError(f"Unsupported verifier.name: {name}")


def samples_for_rank(global_count, rank, world_size):
    base = int(global_count) // int(world_size)
    remainder = int(global_count) % int(world_size)
    return base + int(rank < remainder)


def all_reduce_rollout_stats(stats, device):
    keys = ("attempted", "success", "failed", "gemini_tokens")
    stats_tensor = torch.tensor([stats[key] for key in keys], device=device, dtype=torch.long)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
    return {key: int(value.item()) for key, value in zip(keys, stats_tensor)}


def all_reduce_scalar(value, device, op=dist.ReduceOp.SUM):
    tensor = torch.tensor([float(value)], device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=op)
    return float(tensor.item())


def exact_update_batches(loader, sampler, updates_per_step):
    updates = 0
    epoch = 0
    while updates < int(updates_per_step):
        sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
            updates += 1
            if updates == int(updates_per_step):
                break
        epoch += 1


def get_optimizer_scheduler(trainable_params, opt_cfg, total_steps):
    opt = torch.optim.AdamW(
        trainable_params,
        lr=opt_cfg.lr,
        weight_decay=opt_cfg.weight_decay,
    )
    if opt_cfg.lr_schedule == "constant":
        scheduler = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=1)
    elif opt_cfg.lr_schedule == "cosine":
        warmup_steps = int(opt_cfg.lr_warmup_steps)
        min_lr = opt_cfg.min_lr if opt_cfg.min_lr is not None else opt_cfg.lr * opt_cfg.min_lr_ratio
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=max(1, int(total_steps) - warmup_steps),
            eta_min=min_lr,
        )
        if warmup_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                opt,
                start_factor=1.0 / float(warmup_steps),
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                opt,
                schedulers=[warmup, cosine],
                milestones=[warmup_steps],
            )
        else:
            scheduler = cosine
    else:
        raise ValueError(f"Unsupported lr_schedule: {opt_cfg.lr_schedule}")
    return opt, scheduler


def log_rollout_samples(dataset, cfg, outer_step, log_step):
    if not rank_is_zero():
        return log_step
    max_samples = int(cfg.rollout.log_samples)
    if max_samples <= 0 or len(dataset) == 0:
        return log_step

    import matplotlib.pyplot as plt

    sample_dir = Path(dataset.path) / "logged_samples"
    rows = []
    for row_idx in range(min(max_samples, len(dataset))):
        record = dataset[row_idx]
        feedback = record["feedback"]
        gt_path = record["gt_path"]
        attempt_path = record["attempt_path"]
        if not (gt_path and attempt_path and Path(gt_path).exists() and Path(attempt_path).exists()):
            continue
        rows.append((row_idx, gt_path, attempt_path, feedback))
    if rows:
        sample_dir.mkdir(parents=True, exist_ok=True)
        combined_path = sample_dir / f"outer_{outer_step:06d}_rollout_samples.png"
        fig, axes = plt.subplots(len(rows), 2, figsize=(8, 3.4 * len(rows)), squeeze=False)
        for plot_row, (row_idx, gt_path, attempt_path, feedback) in enumerate(rows):
            axes[plot_row, 0].imshow(plt.imread(gt_path))
            axes[plot_row, 0].set_title(f"sample {row_idx:02d} GT", fontsize=9)
            axes[plot_row, 0].axis("off")
            axes[plot_row, 1].imshow(plt.imread(attempt_path))
            axes[plot_row, 1].set_title(f"Attempt\n{feedback}", fontsize=9, wrap=True)
            axes[plot_row, 1].axis("off")
        fig.tight_layout()
        fig.savefig(combined_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        wandb.log({"rollout/sample_grid": wandb.Image(str(combined_path))}, step=log_step)
        log_step += 1
    return log_step


def progress_bar(iterable=None, total=None, desc=""):
    if not rank_is_zero():
        return iterable if iterable is not None else None
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=False)


def save_training_checkpoint(cfg, model, ema, opt, scheduler, checkpoint_dir, train_steps, logger):
    if not rank_is_zero():
        return
    checkpoint = {
        "model": model.module.state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "scheduler": scheduler.state_dict(),
        "train_steps": train_steps,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    save_checkpoint_atomic(checkpoint, checkpoint_dir / f"{train_steps:07d}.pt")
    save_checkpoint_atomic(ema.state_dict(), checkpoint_dir / f"{train_steps:07d}-ema.pt")
    logger.info(f"Saved checkpoint at step={train_steps}")


def validate_startup_config(cfg):
    assert cfg.train.ckpt is not None or int(cfg.train.pretraining_steps) > 0, (
        "Set train.ckpt to a pretrained checkpoint, or set train.pretraining_steps > 0."
    )


def run_pretraining_loop(
    cfg,
    model,
    ema,
    vae,
    train_diffusion,
    loader,
    train_sampler,
    opt,
    scheduler,
    device,
    logger,
    log_step,
):
    pretraining_steps = int(cfg.train.pretraining_steps)
    if pretraining_steps <= 0:
        return log_step

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    step = 0
    epoch = 0
    log_steps = 0
    running = {key: 0.0 for key in ("loss", "grad_norm", "lr")}
    log_start = time()
    if rank_is_zero():
        logger.info(f"Starting base-loss pretraining for {pretraining_steps} optimizer steps.")

    while step < pretraining_steps:
        train_sampler.set_epoch(epoch)
        for batch in loader:
            x_img = batch["image"].to(device)
            context_tokens = batch["context_tokens"].to(device)
            context_mask = batch["context_mask"].to(device)
            with torch.no_grad():
                x_latent = vae.encode(x_img).latent_dist.sample().mul_(vae.config.scaling_factor)

            opt.zero_grad()
            loss = diffusion_loss(model, train_diffusion, x_latent, context_tokens, context_mask)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            grad_norm_avg = all_reduce_scalar(float(grad_norm.item()), device) / world_size
            lr = opt.param_groups[0]["lr"]
            opt.step()
            scheduler.step()
            update_ema(ema, model.module, decay=cfg.train.ema_decay)

            running["loss"] += float(loss.item())
            running["grad_norm"] += float(grad_norm_avg)
            running["lr"] += float(lr)
            log_steps += 1
            step += 1

            if step % cfg.train.log_every == 0 or step == pretraining_steps:
                keys = list(running.keys())
                values = torch.tensor([running[key] for key in keys], device=device, dtype=torch.float64)
                dist.all_reduce(values, op=dist.ReduceOp.SUM)
                denom = max(log_steps * world_size, 1)
                avg = {key: float(value.item()) / denom for key, value in zip(keys, values)}
                elapsed = max(time() - log_start, 1e-6)
                if rank_is_zero():
                    wandb.log({
                        "pretrain/loss": avg["loss"],
                        "pretrain/grad_norm": avg["grad_norm"],
                        "pretrain/lr": avg["lr"],
                        "pretrain/steps_per_sec": log_steps / elapsed,
                        "pretrain/global_step": step,
                    }, step=log_step)
                    log_step += 1
                    logger.info(
                        f"Pretrain step={step:07d}/{pretraining_steps:07d}: "
                        f"loss={avg['loss']:.4f}, grad_norm={avg['grad_norm']:.4f}, "
                        f"lr={avg['lr']:.6g}, steps_per_sec={log_steps / elapsed:.2f}"
                    )
                running = {key: 0.0 for key in running}
                log_steps = 0
                log_start = time()

            if step >= pretraining_steps:
                break
        epoch += 1

    if rank_is_zero():
        logger.info(f"Finished base-loss pretraining after {pretraining_steps} optimizer steps.")
    return log_step


def run_adaptive_eval(
    cfg,
    model,
    vae,
    sampler,
    verifier,
    context_encoder,
    val_dataset,
    metadata_rows,
    device,
    out_dir,
    step,
    log_step,
):
    if not rank_is_zero() or val_dataset is None or len(val_dataset) == 0:
        return 0.0, log_step
    module = unwrap_model(model)
    start_idx = int(cfg.eval.caption_index)
    batch_size = max(1, int(cfg.eval.batch_size))
    indices = [(start_idx + offset) % len(val_dataset) for offset in range(batch_size)]
    items = [val_dataset[idx] for idx in indices]
    batch = context_collate(items)
    metadata = metadata_by_index(metadata_rows, batch["metadata_index"].tolist())
    gt_images = [val_dataset.image_for_row(val_dataset.indices[idx]) for idx in indices]
    eval_steps = max(1, int(cfg.eval.get("steps", cfg.rollout.get("length", 1))))
    histories = [[] for _ in indices]
    history_images = [[] for _ in indices]
    active = list(range(len(indices)))
    current_tokens = batch["context_tokens"]
    current_mask = batch["context_mask"]
    traces = [[] for _ in indices]
    eval_tokens = 0
    distance_rows = []

    for eval_step in range(eval_steps):
        if not active:
            break
        _, attempt_images = sampler.sample(
            module,
            vae,
            current_tokens,
            current_mask,
            device,
            seed=cfg.eval.seed + step * 1000 + eval_step,
        )
        active_captions = [batch["caption"][idx] for idx in active]
        active_metadata = [metadata[idx] for idx in active]
        active_gt_images = [gt_images[idx] for idx in active]
        score_results = verifier.score_distance_batch(
            active_captions,
            active_metadata,
            active_gt_images,
            attempt_images,
        )
        if score_results:
            eval_tokens += sum(int(result.token_count) for result in score_results)
        for pos, batch_idx in enumerate(active):
            score = None
            if score_results and pos < len(score_results) and score_results[pos].ok:
                score = score_results[pos].score
            traces[batch_idx].append({
                "image": attempt_images[pos],
                "feedback_used": histories[batch_idx][-1] if histories[batch_idx] else "",
                "distance": score,
            })
            if score is not None:
                distance_rows.append((eval_step, float(score)))

        if eval_step + 1 >= eval_steps:
            break
        active_histories = [list(histories[idx]) for idx in active]
        results = verifier.verify_history_batch(
            active_captions,
            active_metadata,
            active_gt_images,
            attempt_images,
            active_histories,
        )
        eval_tokens += sum(int(result.token_count) for result in results)
        success_positions = [idx for idx, result in enumerate(results) if result.ok]
        if not success_positions:
            break
        next_active = []
        next_captions = []
        next_histories = []
        next_history_images = []
        for pos in success_positions:
            batch_idx = active[pos]
            histories[batch_idx].append(results[pos].feedback)
            history_images[batch_idx].append(attempt_images[pos])
            next_active.append(batch_idx)
            next_captions.append(batch["caption"][batch_idx])
            next_histories.append(list(histories[batch_idx]))
            next_history_images.append(list(history_images[batch_idx]))
        current_tokens, current_mask = context_encoder.encode_history(
            next_captions,
            next_histories,
            next_history_images,
        )
        active = next_active

    trace_dir = Path(out_dir) / "adaptive_eval"
    table = wandb.Table(columns=["step", "eval_index", "caption", "feedback", "trace"])
    for batch_idx, eval_index in enumerate(indices):
        if not traces[batch_idx]:
            continue
        grid_path = trace_dir / f"step_{step:07d}_idx_{indices[batch_idx]:06d}.png"
        first_attempt = traces[batch_idx][0]["image"]
        last_attempt = traces[batch_idx][-1]["image"]
        feedback_text = "\n".join(histories[batch_idx])
        save_trace_grid(grid_path, gt_images[batch_idx], first_attempt, feedback_text, last_attempt)
        table.add_data(
            step,
            eval_index,
            batch["caption"][batch_idx],
            feedback_text,
            wandb.Image(str(grid_path)),
        )
    metrics = {"eval/adaptive_trace": table}
    if distance_rows:
        by_step = {}
        for eval_step, score in distance_rows:
            by_step.setdefault(eval_step, []).append(score)
        for eval_step, values in by_step.items():
            metrics[f"eval/distance_step_{eval_step}"] = sum(values) / len(values)
        best_scores = []
        for row_trace in traces:
            scores = [entry["distance"] for entry in row_trace if entry["distance"] is not None]
            if scores:
                best_scores.append(min(scores))
        if best_scores:
            metrics["eval/best_distance"] = sum(best_scores) / len(best_scores)
            metrics["eval/best_aligned_score"] = sum(1.0 / (1.0 + score) for score in best_scores) / len(best_scores)
    wandb.log(metrics, step=log_step)
    log_step += 1
    return eval_tokens, log_step


def run_on_policy_loop(
    cfg,
    model,
    ema,
    vae,
    sampler,
    verifier,
    context_encoder,
    train_diffusion,
    metadata_rows,
    loader,
    train_sampler,
    opt,
    scheduler,
    val_dataset,
    device,
    experiment_dir,
    checkpoint_dir,
    logger,
    log_step,
):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_batch_size = cfg.train.global_batch_size // world_size
    rollout_samples = int(cfg.rollout.samples)
    updates_per_rollout = int(cfg.train.updates_per_rollout)
    local_samples = samples_for_rank(rollout_samples, rank, world_size)
    rollout_root = Path(cfg.rollout.storage_dir) if cfg.rollout.storage_dir is not None else Path(experiment_dir) / "rollouts"
    collector = RolloutCollector(
        model=ema if cfg.rollout.use_ema else model,
        vae=vae,
        sampler=sampler,
        verifier=verifier,
        context_encoder=context_encoder,
        metadata_rows=metadata_rows,
        rollout_length=cfg.rollout.get("length", 1),
    )
    trainer = OnPolicyTrainer(
        model=model,
        train_diffusion=train_diffusion,
        feedback_weight=cfg.loss.feedback_weight,
        base_weight=cfg.loss.base_weight,
    )

    train_steps = 0
    outer_step = 0
    cumulative_gemini_tokens = 0
    max_train_steps = cfg.train.max_train_steps
    if max_train_steps is None:
        raise ValueError("train.max_train_steps must be set for on-policy training.")
    outer_total = (int(max_train_steps) + updates_per_rollout - 1) // updates_per_rollout
    outer_iter = range(outer_total)
    outer_progress = progress_bar(outer_iter, total=outer_total, desc="on-policy steps")
    if outer_progress is None:
        outer_progress = outer_iter

    for _ in outer_progress:
        outer_step += 1
        step_dir = rollout_root / f"step_{outer_step:06d}"
        shard_dir = step_dir / f"rank_{rank:03d}"
        collect_progress = progress_bar(total=local_samples, desc=f"collect {outer_step:06d}")
        rollout_start = time()
        stats = collector.collect(
            loader,
            shard_dir,
            sample_count=local_samples,
            device=device,
            seed=cfg.train.global_seed + outer_step * 1_000_000 + rank,
            progress=collect_progress,
            data_sampler=train_sampler,
            epoch=outer_step,
        )
        if collect_progress is not None:
            collect_progress.close()
        rollout_seconds = time() - rollout_start

        global_stats = all_reduce_rollout_stats(stats, device)
        max_rollout_seconds = all_reduce_scalar(rollout_seconds, device, op=dist.ReduceOp.MAX)
        cumulative_gemini_tokens += global_stats["gemini_tokens"]
        global_success = global_stats["success"]
        global_attempted = global_stats["attempted"]
        global_failed = global_stats["failed"]
        fail_rate = global_failed / max(global_attempted, 1)
        success_rate = global_success / max(global_attempted, 1)
        rollout_samples_per_sec = global_attempted / max(max_rollout_seconds, 1e-6)

        if rank_is_zero():
            wandb.log({
                "rollout/outer_step": outer_step,
                "rollout/attempted": global_attempted,
                "rollout/success": global_success,
                "rollout/failed": global_failed,
                "rollout/success_rate": success_rate,
                "rollout/failure_rate": fail_rate,
                "rollout/seconds": max_rollout_seconds,
                "rollout/samples_per_sec": rollout_samples_per_sec,
                "rollout/local_samples_per_rank": local_samples,
                "rollout/samples": rollout_samples,
                "rollout/batch_size_per_rank": int(loader.batch_size or 0),
                "train/updates_per_rollout": updates_per_rollout,
                "rollout/use_ema": float(bool(cfg.rollout.use_ema)),
                "generation/ddim_steps": int(cfg.sampler.num_sampling_steps),
                "generation/cfg_scale": float(cfg.sampler.cfg_scale),
                "generation/ddim_eta": float(cfg.sampler.ddim_eta),
                "gemini/total_tokens": cumulative_gemini_tokens,
                "gemini/tokens_this_rollout": global_stats["gemini_tokens"],
            }, step=log_step)
            log_step += 1
            logger.info(
                f"Rollout outer={outer_step:06d}: generated {global_attempted} attempts "
                f"across {world_size} ranks in {max_rollout_seconds:.1f}s "
                f"({rollout_samples_per_sec:.2f} attempts/s); "
                f"accepted {global_success}, failed {global_failed}, success_rate={success_rate:.3f}; "
                f"gemini_tokens_this_rollout={global_stats['gemini_tokens']}, "
                f"gemini_tokens_total={cumulative_gemini_tokens}"
            )

        dist.barrier()
        if global_success == 0:
            logger.info(f"Skipping outer={outer_step}: no successful rollout rows.")
            dist.barrier()
            continue

        buffer_dataset = RolloutBuffer(step_dir)
        update_batch_size = min(local_batch_size, len(buffer_dataset))
        log_step = log_rollout_samples(buffer_dataset, cfg, outer_step, log_step)
        if rank_is_zero():
            wandb.log({
                "rollout/buffer_rows": len(buffer_dataset),
                "train/update_batch_size_per_rank": update_batch_size,
                "train/global_batch_size": int(cfg.train.global_batch_size),
            }, step=log_step)
            log_step += 1
            logger.info(
                f"Training outer={outer_step:06d}: loaded {len(buffer_dataset)} merged rollout rows; "
                f"DDP batch size is {update_batch_size} per rank ({int(cfg.train.global_batch_size)} global); "
                f"running exactly {updates_per_rollout} optimizer updates."
            )
        buffer_sampler = DistributedSampler(
            buffer_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=cfg.train.global_seed + outer_step,
            drop_last=False,
        )
        buffer_loader = DataLoader(
            buffer_dataset,
            batch_size=update_batch_size,
            shuffle=False,
            sampler=buffer_sampler,
            num_workers=0,
            pin_memory=cfg.dataloader.pin_memory,
            collate_fn=rollout_collate,
            drop_last=False,
        )
        running = {key: 0.0 for key in ("loss", "feedback_loss", "base_loss", "grad_norm")}
        log_steps = 0
        log_start = time()
        update_iter = exact_update_batches(buffer_loader, buffer_sampler, updates_per_rollout)
        inner_progress = progress_bar(update_iter, total=updates_per_rollout, desc=f"updates {outer_step:06d}")
        if inner_progress is None:
            inner_progress = update_iter

        train_outer_start = time()
        for outer_update, batch in enumerate(inner_progress, start=1):
            opt.zero_grad()
            loss, step_stats = trainer.compute_loss(batch, device)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            grad_norm_avg = all_reduce_scalar(float(grad_norm.item()), device) / world_size
            lr = opt.param_groups[0]["lr"]
            opt.step()
            scheduler.step()
            update_ema(ema, model.module, decay=cfg.train.ema_decay)

            for key in ("loss", "feedback_loss", "base_loss"):
                running[key] += step_stats[key]
            running["grad_norm"] += float(grad_norm_avg)
            log_steps += 1
            train_steps += 1

            update_keys = ("loss", "feedback_loss", "base_loss")
            update_values = torch.tensor([step_stats[key] for key in update_keys], device=device, dtype=torch.float64)
            dist.all_reduce(update_values, op=dist.ReduceOp.SUM)
            update_avg = {
                key: float(value.item()) / world_size
                for key, value in zip(update_keys, update_values)
            }
            if rank_is_zero():
                wandb.log({
                    "train_update/loss": update_avg["loss"],
                    "train_update/feedback_loss": update_avg["feedback_loss"],
                    "train_update/base_loss": update_avg["base_loss"],
                    "train_update/grad_norm": grad_norm_avg,
                    "train_update/lr": lr,
                    "train/global_step": train_steps,
                    "train_update/outer_step": outer_step,
                    "train_update/outer_update": outer_update,
                    "train_update/batch_size_per_rank": int(batch["x_latent"].shape[0]),
                }, step=log_step)
                log_step += 1

            if train_steps % cfg.train.log_every == 0:
                keys = list(running.keys())
                values = torch.tensor([running[key] for key in keys], device=device, dtype=torch.float64)
                dist.all_reduce(values, op=dist.ReduceOp.SUM)
                denom = max(log_steps * world_size, 1)
                avg = {key: float(value.item()) / denom for key, value in zip(keys, values)}
                elapsed = max(time() - log_start, 1e-6)
                if rank_is_zero():
                    wandb.log({
                        "train/loss": avg["loss"],
                        "train/feedback_loss": avg["feedback_loss"],
                        "train/base_loss": avg["base_loss"],
                        "train/grad_norm": avg["grad_norm"],
                        "train/lr": lr,
                        "train/steps_per_sec": log_steps / elapsed,
                        "train/global_step": train_steps,
                        "train/outer_step": outer_step,
                    }, step=log_step)
                    log_step += 1
                    logger.info(
                        f"Train step={train_steps:07d} outer={outer_step:06d} "
                        f"update={outer_update:04d}/{updates_per_rollout}: "
                        f"avg_loss={avg['loss']:.4f}, avg_feedback_loss={avg['feedback_loss']:.4f}, "
                        f"avg_base_loss={avg['base_loss']:.4f}, avg_grad_norm={avg['grad_norm']:.4f}, "
                        f"lr={lr:.6g}, steps_per_sec={log_steps / elapsed:.2f}"
                    )
                running = {key: 0.0 for key in running}
                log_steps = 0
                log_start = time()

            if train_steps % cfg.train.ckpt_every == 0:
                eval_tokens, log_step = run_adaptive_eval(
                    cfg,
                    ema,
                    vae,
                    sampler,
                    verifier,
                    context_encoder,
                    val_dataset,
                    metadata_rows,
                    device,
                    experiment_dir,
                    train_steps,
                    log_step=log_step,
                )
                if rank_is_zero() and eval_tokens:
                    cumulative_gemini_tokens += int(eval_tokens)
                    wandb.log({"gemini/total_tokens": cumulative_gemini_tokens}, step=log_step)
                    log_step += 1
                save_training_checkpoint(cfg, model, ema, opt, scheduler, checkpoint_dir, train_steps, logger)
                dist.barrier()

            if max_train_steps is not None and train_steps >= int(max_train_steps):
                break

        if hasattr(inner_progress, "close"):
            inner_progress.close()
        train_outer_seconds = time() - train_outer_start
        if rank_is_zero():
            logger.info(
                f"Finished training outer={outer_step:06d}: completed up to global step {train_steps} "
                f"in {train_outer_seconds:.1f}s."
            )
            wandb.log({
                "train/outer_seconds": train_outer_seconds,
                "train/global_step": train_steps,
                "train/outer_step": outer_step,
            }, step=log_step)
            log_step += 1
        if max_train_steps is not None and train_steps >= int(max_train_steps):
            break

    if hasattr(outer_progress, "close"):
        outer_progress.close()
    return log_step


@hydra.main(config_path="configs", config_name="train_on_policy", version_base=None)
def main(cfg):
    assert torch.cuda.is_available(), "On-policy training requires at least one GPU."
    validate_startup_config(cfg)
    dist.init_process_group("nccl")
    assert cfg.train.global_batch_size % dist.get_world_size() == 0, "Batch size must be divisible by world size."
    rollout_global_batch_size = int(cfg.rollout.get("batch_size", cfg.train.global_batch_size))
    assert rollout_global_batch_size % dist.get_world_size() == 0, "Rollout batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    torch.manual_seed(cfg.train.global_seed * dist.get_world_size() + rank)

    transform = build_clevr_transform(cfg.model.image_size)
    dataset = ClevrContextDataset(
        cfg.data.dataset_path,
        transform=transform,
        split=cfg.data.split,
        use_disk=cfg.data.use_disk,
        load_meta=True,
        load_images=cfg.data.load_images,
        load_context=cfg.data.load_context,
        max_dataset_size=cfg.data.max_dataset_size,
    )
    val_dataset = ClevrContextDataset(
        cfg.data.dataset_path,
        transform=transform,
        split=cfg.eval.split,
        use_disk=True,
        load_meta=True,
        load_images=False,
        load_context=False,
    ) if cfg.eval.enabled else None
    metadata_rows = load_metadata_for_zarr(cfg.data.dataset_path)

    os.makedirs(cfg.train.results_dir, exist_ok=True)
    experiment_name = cfg.train.experiment_name
    if experiment_name is None:
        model_name = cfg.model.name.replace("/", "-")
        experiment_name = f"{len(glob(f'{cfg.train.results_dir}/*')):03d}-{model_name}-on-policy"
    experiment_dir = Path(cfg.train.results_dir) / experiment_name
    checkpoint_dir = experiment_dir / "checkpoints"
    if rank_is_zero():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger = create_logger(str(experiment_dir))
        OmegaConf.save(cfg, experiment_dir / "config.yaml")
        wandb.init(project=cfg.wandb.project, name=experiment_name, config=OmegaConf.to_container(cfg, resolve=True))
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    model = DiT_models[cfg.model.name](
        input_size=cfg.model.image_size // 8,
        num_classes=cfg.model.num_classes,
        text_conditioning=True,
        context_dim=dataset.context_dim,
        class_dropout_prob=cfg.model.context_dropout_prob,
    )
    if cfg.train.ckpt is not None:
        missing, unexpected = model.load_state_dict(load_checkpoint(cfg.train.ckpt), strict=False)
        logger.info(f"Loaded checkpoint {cfg.train.ckpt}")
        logger.info(f"Missing keys: {missing}")
        logger.info(f"Unexpected keys: {unexpected}")
    else:
        logger.info("No train.ckpt provided; starting from random initialization before pretraining.")
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=False)
    vae = AutoencoderKL.from_pretrained(cfg.model.vae).to(device)
    vae.eval()
    train_diffusion = create_diffusion(timestep_respacing="")
    sample_diffusion = create_diffusion(str(cfg.sampler.num_sampling_steps))
    sampler = PolicySampler(
        sample_diffusion,
        latent_size=cfg.model.image_size // 8,
        vae_scaling_factor=vae.config.scaling_factor,
        cfg_scale=cfg.sampler.cfg_scale,
        sampler=cfg.sampler.type,
        ddim_eta=cfg.sampler.ddim_eta,
    )
    verifier = build_verifier(cfg)
    context_encoder = QwenContextEncoder(
        cfg.context_encoder.model,
        device=f"cuda:{device}",
        vlm_dtype=cfg.context_encoder.dtype,
        device_map=cfg.context_encoder.device_map,
        max_length=cfg.context_encoder.max_length,
        out_dtype=torch.float16,
        include_metadata=cfg.context_encoder.get("include_metadata", True),
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    pretrain_opt, pretrain_scheduler = get_optimizer_scheduler(
        trainable_params,
        cfg.train.pretrain_optimizer,
        cfg.train.pretraining_steps,
    )
    finetune_opt, finetune_scheduler = get_optimizer_scheduler(
        trainable_params,
        cfg.train.finetune_optimizer,
        cfg.train.max_train_steps,
    )
    train_sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=cfg.train.global_seed,
        drop_last=cfg.dataloader.drop_last,
    )
    rollout_sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=cfg.train.global_seed,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        **dataloader_kwargs(
            cfg.dataloader,
            batch_size=cfg.train.global_batch_size // dist.get_world_size(),
            shuffle=False,
            sampler=train_sampler,
            drop_last=cfg.dataloader.drop_last,
            collate_fn=context_collate,
        ),
    )
    rollout_loader = DataLoader(
        dataset,
        **dataloader_kwargs(
            cfg.dataloader,
            batch_size=rollout_global_batch_size // dist.get_world_size(),
            shuffle=False,
            sampler=rollout_sampler,
            drop_last=False,
            collate_fn=context_collate,
        ),
    )
    pretraining_data_ratio = float(cfg.train.pretraining_data_ratio)
    if not (0.0 < pretraining_data_ratio <= 1.0):
        raise ValueError("train.pretraining_data_ratio must be in (0, 1].")
    pretraining_size = max(1, int(round(len(dataset) * pretraining_data_ratio)))
    generator = torch.Generator().manual_seed(int(cfg.train.global_seed))
    pretraining_indices = torch.randperm(len(dataset), generator=generator)[:pretraining_size].tolist()
    pretraining_dataset = Subset(dataset, pretraining_indices)
    pretraining_sampler = DistributedSampler(
        pretraining_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=cfg.train.global_seed,
        drop_last=cfg.dataloader.drop_last,
    )
    pretraining_loader = DataLoader(
        pretraining_dataset,
        **dataloader_kwargs(
            cfg.dataloader,
            batch_size=cfg.train.global_batch_size // dist.get_world_size(),
            shuffle=False,
            sampler=pretraining_sampler,
            drop_last=cfg.dataloader.drop_last,
            collate_fn=context_collate,
        ),
    )
    logger.info(
        f"Dataset contains {len(dataset):,} rows; pretraining uses {len(pretraining_dataset):,} rows "
        f"({pretraining_data_ratio:.3f}); split={cfg.data.split}; context_dim={dataset.context_dim}"
    )

    update_ema(ema, model.module, decay=0)
    model.train()
    log_step = 0
    log_step = run_pretraining_loop(
        cfg=cfg,
        model=model,
        ema=ema,
        vae=vae,
        train_diffusion=train_diffusion,
        loader=pretraining_loader,
        train_sampler=pretraining_sampler,
        opt=pretrain_opt,
        scheduler=pretrain_scheduler,
        device=device,
        logger=logger,
        log_step=log_step,
    )
    log_step = run_on_policy_loop(
        cfg=cfg,
        model=model,
        ema=ema,
        vae=vae,
        sampler=sampler,
        verifier=verifier,
        context_encoder=context_encoder,
        train_diffusion=train_diffusion,
        metadata_rows=metadata_rows,
        loader=rollout_loader,
        train_sampler=rollout_sampler,
        opt=finetune_opt,
        scheduler=finetune_scheduler,
        val_dataset=val_dataset,
        device=device,
        experiment_dir=experiment_dir,
        checkpoint_dir=checkpoint_dir,
        logger=logger,
        log_step=log_step,
    )

    if rank_is_zero():
        wandb.finish()
    cleanup()


if __name__ == "__main__":
    main()
