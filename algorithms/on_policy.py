"""On-policy feedback distillation: rollout generation, verification, and training.

Model-agnostic: the trainer orchestrates rollout -> verify -> update but delegates every
model-specific operation to the model (`generate` for sampling attempts, `rollout_loss` for the
update). It never touches an encoder, sampler, VAE, or diffusion, so it trains both QwenDiT and
OmniGen. Attempts condition on the full interleaved history; records store that raw history.
"""
from contextlib import contextmanager
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
    normalized_tensor_to_pil,
    progress_bar,
    rank_is_zero,
    requires_grad,
    save_checkpoint_atomic,
    unwrap_model,
    update_ema,
)
from datasets.clevr.dataset import context_collate
from datasets.rollouts import RolloutBuffer, rollout_collate


def _slice_batch(batch, count):
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value) or isinstance(value, list):
            out[key] = value[:count]
        else:
            out[key] = value
    return out


class RolloutCollector:
    """Samples attempts via model.generate, verifies them, and writes accepted records to disk.

    A rollout of length K makes K predictions and verifies only the first K-1: feedback on the
    final attempt is never used to make another prediction. Each attempt conditions on the FULL
    prior interleaved history (caption, attempt_0, feedback_0, ...). A record stores the raw
    history that generated its attempt (caption + prior feedbacks + prior attempt-image paths) plus
    the GT path; the model turns that history into a loss at update time.
    """

    def __init__(self, model, sampler_cfg, verifier, rollout_length=1):
        self.model = model
        self.sampler_cfg = sampler_cfg
        self.verifier = verifier
        self.rollout_length = max(1, int(rollout_length))

    def _generate(self, active, captions, feedback_history, attempt_image_history, attempt_path_history, seed):
        context_batch = {
            "caption": [captions[idx] for idx in active],
            "feedback_history": [list(feedback_history[idx]) for idx in active],
            "attempt_images": [list(attempt_image_history[idx]) for idx in active],
            "attempt_paths": [list(attempt_path_history[idx]) for idx in active],
        }
        return self.model.generate(
            context_batch,
            num_sampling_steps=int(self.sampler_cfg.num_sampling_steps),
            cfg_scale=float(self.sampler_cfg.cfg_scale),
            ddim_eta=float(self.sampler_cfg.ddim_eta),
            seed=seed,
        )

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

            captions = batch["caption"]
            gt_images = [normalized_tensor_to_pil(image) for image in batch["image"]]
            gt_paths = []
            for local_idx in range(batch_size):
                gt_path = gt_dir / f"{base_attempted + local_idx:06d}.png"
                gt_images[local_idx].save(gt_path)
                gt_paths.append(str(gt_path))

            active = list(range(batch_size))
            feedback_history = [[] for _ in range(batch_size)]
            attempt_image_history = [[] for _ in range(batch_size)]
            attempt_path_history = [[] for _ in range(batch_size)]

            for step_idx in range(self.rollout_length):
                if not active:
                    break
                step_seed = None if seed is None else int(seed) + base_attempted * self.rollout_length + step_idx
                attempt_images = self._generate(
                    active, captions, feedback_history, attempt_image_history, attempt_path_history, step_seed
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
                        [list(feedback_history[idx]) for idx in active],
                    )
                    token_count += sum(int(result.token_count) for result in results)
                    success_positions = [pos for pos, result in enumerate(results) if result.ok]
                    failed += len(results) - len(success_positions)

                next_active = []
                for pos in success_positions:
                    batch_idx = active[pos]
                    feedback = results[pos].feedback if results is not None else ""
                    record_id = len(records)
                    attempt_path = attempts_dir / f"{record_id:06d}_step_{step_idx:02d}.png"
                    attempt_images[pos].save(attempt_path)
                    records.append({
                        "gt_path": gt_paths[batch_idx],
                        "caption": captions[batch_idx],
                        "feedback_history": list(feedback_history[batch_idx]),
                        "attempt_paths": list(attempt_path_history[batch_idx]),
                        "attempt_path": str(attempt_path),
                        "feedback": feedback,
                    })
                    if not is_last:
                        feedback_history[batch_idx].append(feedback)
                        attempt_image_history[batch_idx].append(attempt_images[pos])
                        attempt_path_history[batch_idx].append(str(attempt_path))
                        next_active.append(batch_idx)

                active = next_active
                if is_last or not active:
                    break

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

        self.scorer = make_scorer() if self.rank == 0 else None

        self.ema = deepcopy(unwrap_model(model.net))
        requires_grad(self.ema, False)
        self.ema.eval()
        model.net.train()

        self.collector = RolloutCollector(
            model=model,
            sampler_cfg=sampler,
            verifier=verifier,
            rollout_length=rollout.length,
        )

        self.params = model.trainable_parameters()
        self.opt, _ = build_optimizer_scheduler(self.params, lr=lr, weight_decay=weight_decay)

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

    @contextmanager
    def _ema_weights(self):
        """Swap EMA weights into the live net for the duration of the block, then restore.

        Used so rollouts generate from the EMA policy without a separate model copy. A no-op when
        rollout.use_ema is False.
        """
        if not self.rollout_cfg.use_ema:
            yield
            return
        module = unwrap_model(self.model.net)
        saved_weights = {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}
        module.load_state_dict(self.ema.state_dict())
        try:
            yield
        finally:
            module.load_state_dict(saved_weights)

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
        with self._ema_weights():
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
        # Feedback-verifier cost is per-rank (sum); the eval scorer runs on rank 0 only.
        verifier_cost = all_reduce_scalar(getattr(self.verifier, "session_cost", 0.0), self.device)
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
                "verifier/total_session_cost": verifier_cost + getattr(self.scorer, "session_cost", 0.0),
            }, step=self.train_steps)
        self.logger.info(
            f"Rollout outer={outer_step:06d}: generated {global_stats['attempted']} attempts "
            f"across {self.world_size} ranks in {max_rollout_seconds:.1f}s; "
            f"accepted {global_stats['success']}, failed {global_stats['failed']}, "
            f"success_rate={success_rate:.3f}; verifier_tokens={global_stats['gemini_tokens']}"
        )
        return global_stats, step_dir

    def update(self, outer_step, step_dir):
        buffer_dataset = RolloutBuffer(step_dir)
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
        self.model.net.train()
        batches = exact_update_batches(buffer_loader, buffer_sampler, self.updates_per_rollout)
        for batch in progress_bar(batches, total=self.updates_per_rollout, desc=f"updates {outer_step:06d}"):
            self.opt.zero_grad()
            loss = self.model.rollout_loss(batch)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip)
            grad_norm_avg = all_reduce_scalar(float(grad_norm.item()), self.device) / self.world_size
            self.opt.step()
            update_ema(self.ema, unwrap_model(self.model.net), decay=self.ema_decay)

            running["loss"] += float(loss.item())
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
        from algorithms.eval import adaptive_eval, distance_metrics, save_adaptive_trace_grid

        if not rank_is_zero() or self.val_dataset is None or len(self.val_dataset) == 0:
            return 0
        start_idx = int(self.eval_cfg.caption_index)
        batch_size = max(1, int(self.eval_cfg.batch_size))
        indices = [(start_idx + offset) % len(self.val_dataset) for offset in range(batch_size)]
        captions = [self.val_dataset[idx]["caption"] for idx in indices]
        gt_images = [self.val_dataset.image_for_index(idx) for idx in indices]

        traces, _histories, eval_tokens = adaptive_eval(
            self.model,
            self.verifier,
            captions,
            gt_images,
            steps=max(1, int(self.eval_cfg.steps)),
            seed=int(self.eval_cfg.seed) + self.train_steps * 1000,
            scorer=self.scorer,
            sampler_cfg=self.sampler_cfg,
        )

        grid_path = self.log_dir / "adaptive_eval" / f"step_{self.train_steps:07d}.png"
        metrics = {}
        if save_adaptive_trace_grid(grid_path, traces, gt_images, captions) is not None:
            metrics["eval/adaptive_traces"] = wandb.Image(str(grid_path))
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
