"""On-policy feedback distillation: rollout generation, verification, and training."""
from copy import deepcopy
from pathlib import Path
from time import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import wandb
from algorithms.utils import (
    build_optimizer_scheduler,
    create_logger,
    dataloader_kwargs,
    diffusion_loss,
    normalized_tensor_to_pil,
    progress_bar,
    rank_is_zero,
    requires_grad,
    save_checkpoint_atomic,
    save_trace_grid,
    tensor_to_pil,
    unwrap_model,
    update_ema,
)
from datasets.clevr.dataset import context_collate
from datasets.rollouts import RolloutBuffer, rollout_collate
from diffusion import create_diffusion


class PolicySampler:
    """Samples images from a DiT with DDIM/DDPM, optionally with classifier-free guidance."""

    def __init__(
        self,
        diffusion,
        latent_size,
        vae_scaling_factor,
        cfg_scale=1.0,
        sampler="ddim",
        ddim_eta=0.0,
    ):
        self.diffusion = diffusion
        self.latent_size = int(latent_size)
        self.vae_scaling_factor = float(vae_scaling_factor)
        self.cfg_scale = float(cfg_scale)
        self.sampler = sampler
        self.ddim_eta = float(ddim_eta)

    @torch.no_grad()
    def sample(self, model, vae, context_tokens, context_mask, device, seed=None):
        module = unwrap_model(model)
        was_training = module.training
        module.eval()
        batch_size = int(context_tokens.shape[0])
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(seed))
        z = torch.randn(
            batch_size,
            4,
            self.latent_size,
            self.latent_size,
            device=device,
            generator=generator,
        )
        model_dtype = next(module.parameters()).dtype
        model_kwargs = {
            "context_tokens": context_tokens.to(device=device, dtype=model_dtype),
            "context_mask": context_mask.to(device=device),
        }
        if self.cfg_scale > 1:
            z = torch.cat([z, z], dim=0)
            model_kwargs = {
                "context_tokens": model_kwargs["context_tokens"].repeat(2, 1, 1),
                "context_mask": model_kwargs["context_mask"].repeat(2, 1),
                "cfg_scale": self.cfg_scale,
            }
            forward_fn = module.forward_with_cfg
        else:
            forward_fn = module.forward

        sample_loop = self.diffusion.ddim_sample_loop if self.sampler == "ddim" else self.diffusion.p_sample_loop
        samples = sample_loop(
            forward_fn,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=device,
            **({"eta": self.ddim_eta} if self.sampler == "ddim" else {}),
        )
        if self.cfg_scale > 1:
            samples, _ = samples.chunk(2, dim=0)
        decoded = vae.decode(samples / self.vae_scaling_factor).sample
        if was_training:
            module.train()
        return decoded, [tensor_to_pil(image) for image in decoded]


def _slice_batch(batch, count):
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value) or isinstance(value, list):
            out[key] = value[:count]
        else:
            out[key] = value
    return out


class RolloutCollector:
    """Samples attempts, verifies them, and writes accepted (latent, context) records to disk.

    A rollout of length K makes K predictions and verifies only the first K-1: feedback on
    the final attempt is never used to make another prediction. A depth-k record trains
    C_k -> GT, where C_k is the interleaved history that generated attempt k (caption-only
    at depth 0). When preprocess_context is True (frozen encoder) each record also stores
    its C_k tokens so training reads them straight from the buffer.
    """

    def __init__(self, model, vae, sampler, verifier, encoder, preprocess_context, rollout_length=1):
        self.model = model
        self.vae = vae
        self.sampler = sampler
        self.verifier = verifier
        self.encoder = encoder
        self.preprocess_context = bool(preprocess_context)
        self.rollout_length = max(1, int(rollout_length))

    @torch.no_grad()
    def collect(self, loader, output_dir, sample_count, device, seed=None, progress=None, data_sampler=None, epoch=0):
        output_dir = Path(output_dir)
        attempts_dir = output_dir / "attempts"
        gt_dir = output_dir / "gt"
        attempts_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        records = []
        base_attempted = 0
        sampled_attempts = 0
        failed = 0
        token_count = 0
        current_epoch = int(epoch)
        if data_sampler is not None:
            data_sampler.set_epoch(current_epoch)
        iterator = iter(loader)

        while base_attempted < sample_count:
            try:
                batch = next(iterator)
            except StopIteration:
                current_epoch += 1
                if data_sampler is not None:
                    data_sampler.set_epoch(current_epoch)
                iterator = iter(loader)
                batch = next(iterator)
            remaining = int(sample_count) - base_attempted
            batch = _slice_batch(batch, min(remaining, int(batch["image"].shape[0])))
            batch_size = int(batch["image"].shape[0])
            if batch_size == 0:
                continue

            x_img = batch["image"].to(device)
            captions = batch["caption"]
            gt_images = [normalized_tensor_to_pil(image) for image in batch["image"]]
            x_latents = self.vae.encode(x_img).latent_dist.sample().mul_(self.vae.config.scaling_factor)

            active = list(range(batch_size))
            current_tokens, current_mask = self.encoder.encode_history(
                captions, [[] for _ in captions], [[] for _ in captions]
            )
            histories = [[] for _ in range(batch_size)]
            history_paths = [[] for _ in range(batch_size)]
            history_images = [[] for _ in range(batch_size)]

            for step_idx in range(self.rollout_length):
                if not active:
                    break
                _, attempt_images = self.sampler.sample(
                    self.model,
                    self.vae,
                    current_tokens,
                    current_mask,
                    device,
                    seed=None if seed is None else int(seed) + base_attempted * self.rollout_length + step_idx,
                )
                sampled_attempts += len(active)
                is_last = step_idx + 1 >= self.rollout_length
                if is_last:
                    results = None
                    success_positions = list(range(len(active)))
                else:
                    results = self.verifier.verify(
                        [captions[idx] for idx in active],
                        [gt_images[idx] for idx in active],
                        attempt_images,
                        [list(histories[idx]) for idx in active],
                    )
                    token_count += sum(int(result.token_count) for result in results)
                    success_positions = [idx for idx, result in enumerate(results) if result.ok]
                    failed += len(results) - len(success_positions)

                next_active = []
                next_captions = []
                next_histories = []
                next_history_images = []
                for pos in success_positions:
                    batch_idx = active[pos]
                    feedback = results[pos].feedback if results is not None else ""
                    record_id = len(records)
                    attempt_path = attempts_dir / f"{record_id:06d}_step_{step_idx:02d}.png"
                    gt_path = gt_dir / f"{record_id:06d}.png"
                    attempt_images[pos].save(attempt_path)
                    if not gt_path.exists():
                        gt_images[batch_idx].save(gt_path)

                    record = {
                        "x_latent": x_latents[batch_idx].detach().cpu().float(),
                        "caption": captions[batch_idx],
                        "feedback": feedback,
                        "feedback_history": list(histories[batch_idx]),
                        "history_attempt_paths": list(history_paths[batch_idx]),
                        "step_index": int(step_idx),
                        "gt_path": str(gt_path),
                        "attempt_path": str(attempt_path),
                    }
                    if self.preprocess_context:
                        context_valid = current_mask[pos].detach().cpu().bool()
                        record["context_tokens"] = current_tokens[pos].detach().cpu()[context_valid].to(torch.float16)
                    records.append(record)

                    if not is_last:
                        histories[batch_idx].append(feedback)
                        history_paths[batch_idx].append(str(attempt_path))
                        history_images[batch_idx].append(attempt_images[pos])
                        next_active.append(batch_idx)
                        next_captions.append(captions[batch_idx])
                        next_histories.append(list(histories[batch_idx]))
                        next_history_images.append(list(history_images[batch_idx]))

                active = next_active
                if is_last or not active:
                    break
                current_tokens, current_mask = self.encoder.encode_history(
                    next_captions,
                    next_histories,
                    next_history_images,
                )

            base_attempted += batch_size
            if progress is not None:
                progress.update(batch_size)

        stats = {
            "attempted": int(sampled_attempts),
            "success": int(len(records)),
            "failed": int(failed),
            "gemini_tokens": int(token_count),
        }
        torch.save({"records": records, "stats": stats}, output_dir / "records.pt")
        return stats


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


class OnPolicyTrainer:
    """Rollout -> verify -> distill loop with fixed-LR optimizers."""

    def __init__(
        self,
        model,
        dataset,
        verifier,
        device,
        log_dir,
        sampler,
        rollout,
        lr,
        weight_decay,
        global_batch_size,
        global_seed,
        updates_per_rollout,
        max_train_steps,
        grad_clip,
        ema_decay,
        log_every,
        ckpt_every,
        dataloader,
        eval,
        val_dataset=None,
        start_step=0,
    ):
        from verifiers.eval_metrics import make_scorer

        self.model = model
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.verifier = verifier
        self.device = device
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = self.log_dir / "checkpoints"
        self.sampler_cfg = sampler
        self.rollout_cfg = rollout
        self.global_seed = int(global_seed)
        self.updates_per_rollout = int(updates_per_rollout)
        self.max_train_steps = int(max_train_steps)
        self.start_step = int(start_step)
        self.total_steps = self.start_step + self.max_train_steps
        self.grad_clip = float(grad_clip)
        self.ema_decay = float(ema_decay)
        self.log_every = int(log_every)
        self.ckpt_every = int(ckpt_every)
        self.eval_cfg = eval
        self.train_steps = self.start_step
        self.cumulative_verifier_tokens = 0
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_batch_size = int(global_batch_size) // self.world_size
        self.logger = create_logger(str(self.log_dir), self.rank)

        assert int(rollout.batch_size) % self.world_size == 0, "Rollout batch size must be divisible by world size."
        self.rollout_root = Path(rollout.storage_dir) if rollout.storage_dir is not None else self.log_dir / "rollouts"

        self.encoder = model.get_encoder()
        self.scorer = make_scorer() if self.rank == 0 else None

        self.ema = deepcopy(unwrap_model(model.net))
        requires_grad(self.ema, False)
        self.ema.eval()
        model.net.train()

        self.policy_sampler = PolicySampler(
            create_diffusion(str(sampler.num_sampling_steps)),
            latent_size=model.latent_size,
            vae_scaling_factor=model.vae.config.scaling_factor,
            cfg_scale=sampler.cfg_scale,
            sampler=sampler.type,
            ddim_eta=sampler.ddim_eta,
        )
        self.collector = RolloutCollector(
            model=self.ema if rollout.use_ema else model.net,
            vae=model.vae,
            sampler=self.policy_sampler,
            verifier=verifier,
            encoder=self.encoder,
            preprocess_context=self.encoder.freeze,
            rollout_length=rollout.length,
        )

        trainable_params = model.trainable_parameters()
        self.params = trainable_params
        self.opt, _ = build_optimizer_scheduler(trainable_params, lr=lr, weight_decay=weight_decay)

        self.rollout_sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            seed=self.global_seed,
            drop_last=False,
        )
        self.rollout_loader = DataLoader(
            dataset,
            **dataloader_kwargs(
                dataloader,
                batch_size=int(rollout.batch_size) // self.world_size,
                shuffle=False,
                sampler=self.rollout_sampler,
                drop_last=False,
                collate_fn=context_collate,
            ),
        )
        self.pin_memory = bool(dataloader.pin_memory)
        self.logger.info(f"Dataset contains {len(dataset):,} rows.")

    def learn(self):
        outer_total = (self.max_train_steps + self.updates_per_rollout - 1) // self.updates_per_rollout
        for outer_step in progress_bar(range(1, outer_total + 1), total=outer_total, desc="on-policy steps"):
            stats, step_dir = self.collect(outer_step)
            dist.barrier()
            if stats["success"] == 0:
                self.logger.info(f"Skipping outer={outer_step}: no successful rollout rows.")
                dist.barrier()
                continue
            self.update(outer_step, step_dir)
            if self.train_steps >= self.total_steps:
                break
        self.logger.info("Done!")

    def collect(self, outer_step):
        step_dir = self.rollout_root / f"step_{outer_step:06d}"
        shard_dir = step_dir / f"rank_{self.rank:03d}"
        local_samples = samples_for_rank(int(self.rollout_cfg.samples), self.rank, self.world_size)
        collect_progress = progress_bar(total=local_samples, desc=f"collect {outer_step:06d}")
        rollout_start = time()
        self.encoder.eval()
        stats = self.collector.collect(
            self.rollout_loader,
            shard_dir,
            sample_count=local_samples,
            device=self.device,
            seed=self.global_seed + outer_step * 1_000_000 + self.rank,
            progress=collect_progress if hasattr(collect_progress, "update") else None,
            data_sampler=self.rollout_sampler,
            epoch=outer_step,
        )
        if hasattr(collect_progress, "close"):
            collect_progress.close()
        rollout_seconds = time() - rollout_start

        global_stats = all_reduce_rollout_stats(stats, self.device)
        max_rollout_seconds = all_reduce_scalar(rollout_seconds, self.device, op=dist.ReduceOp.MAX)
        self.cumulative_verifier_tokens += global_stats["gemini_tokens"]
        success_rate = global_stats["success"] / max(global_stats["attempted"], 1)
        if rank_is_zero():
            wandb.log({
                "rollout/outer_step": outer_step,
                "rollout/attempted": global_stats["attempted"],
                "rollout/success": global_stats["success"],
                "rollout/failed": global_stats["failed"],
                "rollout/success_rate": success_rate,
                "rollout/seconds": max_rollout_seconds,
                "rollout/samples_per_sec": global_stats["attempted"] / max(max_rollout_seconds, 1e-6),
                "verifier/total_tokens": self.cumulative_verifier_tokens,
                "verifier/tokens_this_rollout": global_stats["gemini_tokens"],
            }, step=self.train_steps)
        self.logger.info(
            f"Rollout outer={outer_step:06d}: generated {global_stats['attempted']} attempts "
            f"across {self.world_size} ranks in {max_rollout_seconds:.1f}s; "
            f"accepted {global_stats['success']}, failed {global_stats['failed']}, "
            f"success_rate={success_rate:.3f}; verifier_tokens={global_stats['gemini_tokens']}"
        )
        return global_stats, step_dir

    def update(self, outer_step, step_dir):
        prepare_fn = None if self.encoder.freeze else self.encoder.prepare_row
        buffer_dataset = RolloutBuffer(step_dir, prepare_fn=prepare_fn)
        update_batch_size = min(self.local_batch_size, len(buffer_dataset))
        self.log_rollout_samples(buffer_dataset, outer_step)
        if rank_is_zero():
            wandb.log({"rollout/buffer_rows": len(buffer_dataset)}, step=self.train_steps)
        self.logger.info(
            f"Training outer={outer_step:06d}: {len(buffer_dataset)} rollout rows, "
            f"batch size {update_batch_size} per rank, {self.updates_per_rollout} updates."
        )
        buffer_sampler = DistributedSampler(
            buffer_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            seed=self.global_seed + outer_step,
            drop_last=False,
        )
        buffer_loader = DataLoader(
            buffer_dataset,
            batch_size=update_batch_size,
            shuffle=False,
            sampler=buffer_sampler,
            num_workers=0,
            pin_memory=self.pin_memory,
            collate_fn=rollout_collate,
            drop_last=False,
        )
        running = {key: 0.0 for key in ("loss", "grad_norm")}
        log_steps = 0
        log_start = time()
        self.encoder.train()
        batches = exact_update_batches(buffer_loader, buffer_sampler, self.updates_per_rollout)
        for batch in progress_bar(batches, total=self.updates_per_rollout, desc=f"updates {outer_step:06d}"):
            self.opt.zero_grad()
            loss, step_stats = self.compute_loss(batch)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip)
            grad_norm_avg = all_reduce_scalar(float(grad_norm.item()), self.device) / self.world_size
            self.opt.step()
            update_ema(self.ema, unwrap_model(self.model.net), decay=self.ema_decay)

            running["loss"] += step_stats["loss"]
            running["grad_norm"] += float(grad_norm_avg)
            log_steps += 1
            self.train_steps += 1

            if self.train_steps % self.log_every == 0:
                keys = list(running.keys())
                values = torch.tensor([running[key] for key in keys], device=self.device, dtype=torch.float64)
                dist.all_reduce(values, op=dist.ReduceOp.SUM)
                denom = max(log_steps * self.world_size, 1)
                avg = {key: float(value.item()) / denom for key, value in zip(keys, values)}
                elapsed = max(time() - log_start, 1e-6)
                if rank_is_zero():
                    wandb.log({
                        "train/loss": avg["loss"],
                        "train/grad_norm": avg["grad_norm"],
                        "train/lr": self.opt.param_groups[0]["lr"],
                        "train/steps_per_sec": log_steps / elapsed,
                        "train/outer_step": outer_step,
                    }, step=self.train_steps)
                running = {key: 0.0 for key in running}
                log_steps = 0
                log_start = time()

            if self.train_steps % self.ckpt_every == 0:
                eval_tokens = self.eval_step()
                if rank_is_zero() and eval_tokens:
                    self.cumulative_verifier_tokens += int(eval_tokens)
                    wandb.log({"verifier/total_tokens": self.cumulative_verifier_tokens}, step=self.train_steps)
                self.save()
                dist.barrier()

            if self.train_steps >= self.total_steps:
                break

    def compute_loss(self, batch):
        x_latent = batch["x_latent"].to(self.device, non_blocking=True)
        if self.encoder.freeze:
            context_tokens = batch["context_tokens"].to(self.device, non_blocking=True)
            context_mask = batch["context_mask"].to(self.device, non_blocking=True)
        else:
            # Forward the tokenized (once) rollout contexts so gradients reach the encoder
            # and the context tracks its current weights, not rollout-time snapshots.
            context_tokens, context_mask = self.encoder.forward(batch["context_inputs"])
        loss = diffusion_loss(self.model.net, self.model.diffusion, x_latent, context_tokens, context_mask)
        return loss, {"loss": float(loss.item())}

    def log_rollout_samples(self, buffer_dataset, outer_step):
        if not rank_is_zero():
            return
        max_samples = int(self.rollout_cfg.log_samples)
        if max_samples <= 0 or len(buffer_dataset) == 0:
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = []
        for row_idx in range(min(max_samples, len(buffer_dataset))):
            record = buffer_dataset[row_idx]
            if Path(record["gt_path"]).exists() and Path(record["attempt_path"]).exists():
                rows.append((row_idx, record["gt_path"], record["attempt_path"], record["feedback"]))
        if not rows:
            return
        sample_dir = Path(buffer_dataset.path) / "logged_samples"
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
        wandb.log({"rollout/sample_grid": wandb.Image(str(combined_path))}, step=self.train_steps)

    @torch.no_grad()
    def eval_step(self):
        from algorithms.eval import adaptive_rollout, distance_metrics

        if not rank_is_zero() or self.val_dataset is None or len(self.val_dataset) == 0:
            return 0
        self.encoder.eval()
        start_idx = int(self.eval_cfg.caption_index)
        batch_size = max(1, int(self.eval_cfg.batch_size))
        indices = [(start_idx + offset) % len(self.val_dataset) for offset in range(batch_size)]
        items = [self.val_dataset[idx] for idx in indices]
        batch = context_collate(items)
        batch["context_tokens"], batch["context_mask"] = self.encoder.encode_history(
            batch["caption"], [[] for _ in batch["caption"]], [[] for _ in batch["caption"]]
        )
        gt_images = [self.val_dataset.image_for_index(idx) for idx in indices]

        traces, histories, eval_tokens = adaptive_rollout(
            self.ema,
            self.model.vae,
            self.policy_sampler,
            self.verifier,
            self.encoder,
            batch,
            gt_images,
            steps=max(1, int(self.eval_cfg.steps)),
            seed=int(self.eval_cfg.seed) + self.train_steps * 1000,
            scorer=self.scorer,
        )
        self.encoder.train()

        trace_dir = self.log_dir / "adaptive_eval"
        trace_images = []
        for batch_idx, eval_index in enumerate(indices):
            if not traces[batch_idx]:
                continue
            grid_path = trace_dir / f"step_{self.train_steps:07d}_idx_{eval_index:06d}.png"
            feedback_text = "\n".join(histories[batch_idx])
            save_trace_grid(
                grid_path,
                gt_images[batch_idx],
                traces[batch_idx][0]["image"],
                feedback_text,
                traces[batch_idx][-1]["image"],
            )
            trace_images.append(wandb.Image(str(grid_path), caption=batch["caption"][batch_idx]))
        metrics = {"eval/adaptive_traces": trace_images}
        for key, value in distance_metrics(traces).items():
            metrics[f"eval/{key}"] = value
        wandb.log(metrics, step=self.train_steps)
        return eval_tokens

    def save(self):
        if not rank_is_zero():
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model.checkpoint_state(),
            "ema": self.ema.state_dict(),
            "opt": self.opt.state_dict(),
            "train_steps": self.train_steps,
        }
        encoder_state = self.model.encoder_state()
        if encoder_state is not None:
            payload["context_encoder"] = encoder_state
        checkpoint_path = self.checkpoint_dir / f"{self.train_steps:07d}.pt"
        save_checkpoint_atomic(payload, checkpoint_path)
        save_checkpoint_atomic(self.ema.state_dict(), self.checkpoint_dir / f"{self.train_steps:07d}-ema.pt")
        self.logger.info(f"Saved checkpoint at step={self.train_steps}")
