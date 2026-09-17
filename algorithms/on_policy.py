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

    def __init__(self, model, sampler_cfg, verifier, rollout_length=1, verify_last=False,
                 max_no_update_frac=None, max_oversample=8.0):
        self.model = model
        self.sampler_cfg = sampler_cfg
        self.verifier = verifier
        self.rollout_length = max(1, int(rollout_length))
        # Cap on the share of DEGENERATE chains (first attempt already exact, so every later row is
        # conditioned on "no update" and teaches copy-the-image). None disables rejection entirely
        # and reproduces the original collector. max_oversample bounds the extra generation, which
        # the control run's own rates say reaches ~6x by the end of training at a 25% cap.
        self.max_no_update_frac = (None if max_no_update_frac is None
                                   else float(max_no_update_frac))
        self.max_oversample = float(max_oversample)
        self.last_rollout_composition = {}
        # The final attempt is normally left ungraded because nothing consumes its critique.
        # Grading it costs one extra verifier call per chain and yields the per-position exact
        # rates, which are the headline diagnostic for anchored training.
        self.verify_last = bool(verify_last)

    def _generate(self, captions, feedback_history, attempt_image_history,
                  attempt_path_history, seed, init_latents=None):
        context_batch = {
            "caption": list(captions),
            "feedback_history": [list(history) for history in feedback_history],
            "attempt_images": [list(images) for images in attempt_image_history],
            "attempt_paths": [list(paths) for paths in attempt_path_history],
        }
        return self.model.generate(
            context_batch,
            num_sampling_steps=int(self.sampler_cfg.num_sampling_steps),
            cfg_scale=float(self.sampler_cfg.cfg_scale),
            ddim_eta=float(self.sampler_cfg.ddim_eta),
            seed=seed,
            return_latents=True,      # store x_T so training can reuse the exact sampling noise
            init_latents=init_latents,
        )

    @torch.no_grad()
    def collect(self, loader, output_dir, sample_count, device, seed=None, progress=None, data_sampler=None, epoch=0):
        output_dir = Path(output_dir)
        attempts_dir = output_dir / "attempts"
        gt_dir = output_dir / "gt"
        attempts_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        records = []
        exact_by_pos = {}
        base_attempted = 0
        sampled_attempts = 0
        failed = 0
        token_count = 0
        current_epoch = int(epoch)
        if data_sampler is not None:
            data_sampler.set_epoch(current_epoch)
        iterator = iter(loader)

        # Rejection sampling over CHAINS. A chain is "degenerate" when its first attempt was
        # already exact: every later row in it is conditioned on "no update" and teaches
        # copy-the-image, not repair. Keep sampling until the buffer is FULL of `sample_count`
        # chains with at most `max_degenerate` of them degenerate.
        max_degenerate = (int(round(sample_count * self.max_no_update_frac))
                          if self.max_no_update_frac is not None else None)
        gen_budget = (int(sample_count * self.max_oversample)
                      if max_degenerate is not None else sample_count)
        accepted_chains = 0
        degenerate_accepted = 0
        degenerate_rejected = 0
        generated_chains = 0
        written = 0

        while accepted_chains < sample_count:
            if generated_chains >= gen_budget:
                # Budget spent. Take what we have rather than spin: the realised composition is
                # logged, so a run that could not hit its cap is visible instead of silent.
                break
            try:
                batch = next(iterator)
            except StopIteration:
                current_epoch += 1
                if data_sampler is not None:
                    data_sampler.set_epoch(current_epoch)
                iterator = iter(loader)
                batch = next(iterator)
            batch_size = int(batch["image"].shape[0])
            if batch_size == 0:
                continue

            captions = batch["caption"]
            gt_images = [normalized_tensor_to_pil(image) for image in batch["image"]]
            gt_paths = []
            for local_idx in range(batch_size):
                gt_path = gt_dir / f"{generated_chains + local_idx:06d}.png"
                gt_images[local_idx].save(gt_path)
                gt_paths.append(str(gt_path))

            feedback_history = [[] for _ in range(batch_size)]
            attempt_image_history = [[] for _ in range(batch_size)]
            attempt_path_history = [[] for _ in range(batch_size)]
            # Records are buffered per chain: acceptance is only decidable once the first attempt
            # has been graded, and a chain is accepted or rejected whole.
            pending = [[] for _ in range(batch_size)]
            degenerate = [False] * batch_size
            episode_noise = None

            for step_idx in range(self.rollout_length):
                step_seed = None if seed is None else int(seed) + generated_chains
                attempt_images, attempt_latents = self._generate(
                    captions, feedback_history, attempt_image_history, attempt_path_history,
                    step_seed, init_latents=episode_noise,
                )
                if episode_noise is None:
                    episode_noise = torch.stack([latent.detach() for latent in attempt_latents])
                sampled_attempts += batch_size

                is_last = step_idx + 1 >= self.rollout_length
                results = None
                if not is_last or self.verify_last:
                    results = self.verifier.verify(
                        captions, gt_images, attempt_images,
                        [list(history) for history in feedback_history],
                    )
                    token_count += sum(int(result.token_count) for result in results)
                    failed += sum(1 for result in results if not result.ok)
                    for local_idx, result in enumerate(results):
                        ok = (result.ok and
                              str(result.feedback).strip().lower().rstrip(".") == "no update")
                        # Rates are logged over everything GENERATED, not everything accepted, so
                        # they stay an honest measure of the policy rather than of the filter.
                        exact_by_pos.setdefault(step_idx, []).append(1.0 if ok else 0.0)
                        if step_idx == 0:
                            degenerate[local_idx] = bool(ok)

                for idx in range(batch_size):
                    feedback = ("" if is_last else
                                (results[idx].feedback if results is not None and results[idx].ok else ""))
                    attempt_path = attempts_dir / f"{written:06d}_step_{step_idx:02d}.png"
                    attempt_images[idx].save(attempt_path)
                    latent_path = attempts_dir / f"{written:06d}_step_{step_idx:02d}.pt"
                    torch.save(attempt_latents[idx].to(torch.float16), latent_path)
                    written += 1
                    pending[idx].append({
                        "gt_path": gt_paths[idx],
                        "caption": captions[idx],
                        "feedback_history": list(feedback_history[idx]),
                        "attempt_paths": list(attempt_path_history[idx]),
                        "attempt_path": str(attempt_path),
                        "latent_path": str(latent_path),
                        "feedback": feedback,
                    })
                    if not is_last:
                        feedback_history[idx].append(feedback)
                        attempt_image_history[idx].append(attempt_images[idx])
                        attempt_path_history[idx].append(str(attempt_path))

            for idx in range(batch_size):
                if accepted_chains >= sample_count:
                    reject = True
                elif degenerate[idx] and max_degenerate is not None \
                        and degenerate_accepted >= max_degenerate:
                    reject = True
                    degenerate_rejected += 1
                else:
                    reject = False
                if reject:
                    # Drop the whole chain AND its files: at high oversampling the step directory
                    # would otherwise grow with the rejection factor.
                    for rec in pending[idx]:
                        for key in ("attempt_path", "latent_path"):
                            try:
                                Path(rec[key]).unlink(missing_ok=True)
                            except OSError:
                                pass
                    continue
                records.extend(pending[idx])
                accepted_chains += 1
                degenerate_accepted += int(degenerate[idx])

            generated_chains += batch_size
            base_attempted = accepted_chains
            if progress is not None:
                progress.update(batch_size)

        self.last_rollout_composition = {
            "accepted_chains": int(accepted_chains),
            "generated_chains": int(generated_chains),
            "degenerate_accepted": int(degenerate_accepted),
            "degenerate_rejected": int(degenerate_rejected),
            "oversample": float(generated_chains / max(1, accepted_chains)),
            "degenerate_frac": float(degenerate_accepted / max(1, accepted_chains)),
        }

        stats = {
            "attempted": int(sampled_attempts),
            "success": int(len(records)),
            "failed": int(failed),
            "gemini_tokens": int(token_count),
            # raw counts, not rates: all_reduce_rollout_stats only carries the integer keys above,
            # so the trainer reduces these itself and forms the rate after summing across ranks.
            "exact_pos_sum": {k: float(sum(v)) for k, v in exact_by_pos.items()},
            "exact_pos_n": {k: float(len(v)) for k, v in exact_by_pos.items()},
        }
        torch.save({"records": records, "stats": stats}, output_dir / "records.pt")
        return stats


def samples_for_rank(global_count, rank, world_size):
    base = int(global_count) // int(world_size)
    remainder = int(global_count) % int(world_size)
    return base + int(rank < remainder)


def _select_rows(batch, idx):
    """Sub-batch of a rollout_collate dict, or None when no row qualifies.

    Anchored training runs two objectives over one batch, so the rows must be separable: repair
    rows regress to ground truth, draft rows are distilled toward the frozen base.
    """
    if not idx:
        return None
    out = {}
    for key, value in batch.items():
        if isinstance(value, list) and len(value) == len(batch["caption"]):
            out[key] = [value[i] for i in idx]
        else:
            out[key] = value
    return out


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


class RepeatSampler(torch.utils.data.Sampler):
    """Yield every index of a base sampler `repeats` times: several chains from one prompt.

    Running N independent chains per caption is what gives the loss many DIFFERENT sources mapping
    to the SAME target, which is what forces the critique to carry information rather than letting
    the model reach the target from the caption alone. The repeats land next to each other in a
    batch, and `generate` draws noise for the whole batch at once, so each replica still gets its
    own x_T while the noise stays fixed WITHIN each chain.
    """

    def __init__(self, base, repeats):
        self.base = base
        self.repeats = max(int(repeats), 1)

    def __iter__(self):
        for index in self.base:
            for _ in range(self.repeats):
                yield index

    def __len__(self):
        return len(self.base) * self.repeats

    def set_epoch(self, epoch):
        # collect() calls set_epoch on whatever sampler it is given; forward it so reshuffling
        # across outer iterations still happens.
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)


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
        resume=None,
        eval_every=None,
        curriculum=None,
        caption_dropout_prob=0.0,
        anchor=None,
    ):
        from verifiers.eval_metrics import scorer_from_eval_cfg

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
        # defaults to ckpt_every so existing configs behave exactly as before
        self.eval_every = int(eval_every) if eval_every else int(ckpt_every)
        self.eval_cfg = eval
        self.train_steps = self.start_step
        # Index of the last COMPLETED outer iteration. Restored on resume because it seeds
        # the rollout noise and the sampler epoch -- see learn().
        self.outer_step = 0
        # Curriculum: a list of {loops, length, weights, updates}. The active level is a pure
        # function of outer_step, which save_backup() already persists and _restore() restores, so
        # resuming mid-curriculum needs no extra state. None => the original fixed-length behaviour.
        self.curriculum = [dict(level) for level in curriculum] if curriculum else None
        self.caption_dropout_prob = float(caption_dropout_prob)
        # Anchored training: draft rows (no critique in context) are NOT supervised to ground
        # truth. They are held at the frozen base by a velocity-space distillation term, so the
        # supply of errors the repair step learns from stays stationary instead of drying up as
        # the draft improves -- which is what stalled the unanchored run at 83% draft accuracy.
        self.anchor_beta = float(anchor["beta"]) if anchor else 0.0
        # When true the draft ALSO gets ground truth and the anchor is only a brake; when false
        # the draft is held at the frozen base with no ground truth at all.
        self.anchor_draft_gt = bool(anchor.get("draft_gt", False)) if anchor else False
        # The final attempt normally goes ungraded (nothing consumes its critique). The anchored
        # run needs its exact-rate as the headline diagnostic, so grade it too.
        self.verify_last = bool(getattr(rollout, "verify_last", False))
        self.cumulative_verifier_tokens = 0
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_batch_size = int(global_batch_size) // self.world_size
        self.logger = create_logger(str(self.log_dir), self.rank)

        assert int(rollout.batch_size) % self.world_size == 0, "Rollout batch size must be divisible by world size."
        self.rollout_root = Path(rollout.storage_dir) if rollout.storage_dir is not None else self.log_dir / "rollouts"

        self.scorer = scorer_from_eval_cfg(self.eval_cfg) if self.rank == 0 else None

        self.ema = deepcopy(unwrap_model(model.net))
        requires_grad(self.ema, False)
        self.ema.eval()
        model.net.train()

        self.collector = RolloutCollector(
            model=model,
            sampler_cfg=sampler,
            verifier=verifier,
            rollout_length=rollout.length,
            verify_last=self.verify_last,
            max_no_update_frac=getattr(rollout, "max_no_update_frac", None),
            max_oversample=float(getattr(rollout, "max_oversample", 8.0)),
        )

        self.params = model.trainable_parameters()
        self.opt, _ = build_optimizer_scheduler(self.params, lr=lr, weight_decay=weight_decay)
        if resume:
            self._restore(resume)

        base_rollout_sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            seed=self.global_seed,
            drop_last=False,
        )
        # chains_per_prompt > 1 collects several independent chains per caption (see RepeatSampler).
        # rollout.samples counts CHAINS, so `samples: 200` with chains_per_prompt 4 is 50 captions.
        chains_per_prompt = int(getattr(rollout, "chains_per_prompt", 1) or 1)
        self.chains_per_prompt = chains_per_prompt
        self.rollout_sampler = (
            RepeatSampler(base_rollout_sampler, chains_per_prompt)
            if chains_per_prompt > 1 else base_rollout_sampler
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
        self.num_workers = int(dataloader.num_workers)
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

    def level_for(self, outer_step):
        """The curriculum level a 1-based outer iteration falls in, or None without a curriculum.

        Derived from outer_step alone, so a resumed run lands back in the right level with no extra
        state to save or restore.
        """
        if not self.curriculum:
            return None
        seen = 0
        for level in self.curriculum:
            seen += int(level["loops"])
            if outer_step <= seen:
                return level
        return self.curriculum[-1]

    def learn(self):
        # outer_step CONTINUES across a resume rather than restarting at 1. It is not just a label:
        # collect() derives the rollout noise seed from it (global_seed + outer_step * 1e6 + rank)
        # and passes it as the sampler epoch. Restarting it would make every requeue replay the
        # same x_T draws and the same prompt order as the start of the run -- on a preemptable
        # partition that silently collapses the noise distribution the run actually sees.
        if self.curriculum:
            # The curriculum fixes the loop count outright; max_train_steps is only a backstop.
            last_outer = sum(int(level["loops"]) for level in self.curriculum)
            outer_range = range(self.outer_step + 1, last_outer + 1)
        else:
            remaining = max(self.total_steps - self.train_steps, 0)
            outer_total = (remaining + self.updates_per_rollout - 1) // self.updates_per_rollout
            first = self.outer_step + 1
            outer_range = range(first, first + outer_total)
        for outer_step in progress_bar(outer_range, total=len(outer_range),
                                       desc="on-policy steps"):
            level = self.level_for(outer_step)
            if level is not None:
                # Growing chain length is the curriculum: short chains first, so the model learns a
                # single repair before being asked to compose three.
                self.collector.rollout_length = int(level["length"])
            stats, step_dir = self.collect(outer_step)
            dist.barrier()
            self.outer_step = outer_step
            if stats["success"] == 0:
                self.logger.info(f"Skipping outer={outer_step}: no successful rollout rows.")
                dist.barrier()
                continue
            self.update(outer_step, step_dir)
            self.save_backup()
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
            self.log_on_policy_grid(outer_step)
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
        # Per-position exact rates, summed across ranks before forming the rate. Under anchoring
        # these are the whole diagnostic: the draft must stay FLAT while the repair RISES.
        pos_rates = {}
        for pos in sorted(stats.get("exact_pos_n", {})):
            hits = all_reduce_scalar(stats["exact_pos_sum"][pos], self.device)
            total = all_reduce_scalar(stats["exact_pos_n"][pos], self.device)
            pos_rates[int(pos)] = hits / max(total, 1.0)
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
                **({"rollout/draft_exact": pos_rates[0]} if 0 in pos_rates else {}),
                **({"rollout/repair_exact": pos_rates[max(pos_rates)]}
                   if pos_rates and max(pos_rates) > 0 else {}),
                **{f"rollout/exact_pos_{p}": r for p, r in pos_rates.items()},
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
        level = self.level_for(outer_step)
        # Per-level update count holds the epochs-per-rollout constant as chains (and so record
        # counts) grow across levels.
        n_updates = int(level["updates"]) if level else self.updates_per_rollout
        # Weight by chain position: later attempts, the ones we actually want correct, count more.
        pos_weights = [float(w) for w in level["weights"]] if level else None
        self.log_rollout_samples(buffer_dataset, outer_step)
        if rank_is_zero():
            wandb.log({"rollout/buffer_rows": len(buffer_dataset)}, step=self.train_steps)
        self.logger.info(
            f"Training outer={outer_step:06d}: {len(buffer_dataset)} rollout rows, "
            f"batch size {update_batch_size} per rank, {n_updates} updates."
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
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=rollout_collate,
            drop_last=False,
            **({"prefetch_factor": 4, "persistent_workers": True} if self.num_workers > 0 else {}),
        )
        running = {key: 0.0 for key in ("loss", "grad_norm", "mean_weight", "caption_dropped",
                                        "loss_repair", "loss_anchor")}
        log_steps = 0
        log_start = time()
        self.model.net.train()
        batches = exact_update_batches(buffer_loader, buffer_sampler, n_updates)
        for batch in progress_bar(batches, total=n_updates, desc=f"updates {outer_step:06d}"):
            self.opt.zero_grad()
            weights = None
            if pos_weights is not None:
                # chain_pos can exceed the level's chain length only if a buffer from a longer
                # level were reused; clamp rather than crash.
                weights = [pos_weights[min(int(p), len(pos_weights) - 1)]
                           for p in batch["chain_pos"]]
            if self.anchor_beta > 0.0:
                # Draft rows (no critique yet) are pulled toward the frozen base; repair rows get
                # the usual flow-matching target. Splitting the batch is what lets one batch carry
                # two different objectives.
                #
                # anchor.draft_gt selects between the two arms:
                #   false -- the draft gets NO ground truth, so it is fully pinned to the base.
                #   true  -- the draft ALSO regresses to ground truth ("expert loss at step 0"),
                #            and the anchor is only a brake. Both terms are quadratic in the
                #            predicted velocity, so the draft settles at a blend,
                #            (w*v_gt + beta*v_base)/(w+beta): beta is the fraction pulled back
                #            toward the base, not an on/off switch.
                repair = _select_rows(batch, [i for i, p in enumerate(batch["chain_pos"]) if int(p) > 0])
                draft = _select_rows(batch, [i for i, p in enumerate(batch["chain_pos"]) if int(p) == 0])
                gt_rows = batch if self.anchor_draft_gt else repair
                # EVERY RANK MUST RUN THE SAME NUMBER OF FORWARDS. DDP broadcasts module buffers
                # once per forward, so a rank whose local batch happens to hold only drafts (or
                # only repairs) would skip a collective and desynchronise the process group --
                # NCCL then times out with rank 0 in an ALLREDUCE while its peers sit in a
                # BROADCAST. At 4 ranks the local batch is 8 rows from a 50/50 buffer, so a
                # single-class batch arrives roughly every 32 optimizer steps: frequent enough to
                # kill every run. Both terms are therefore always evaluated, and whichever has no
                # real rows is scaled to zero instead of being skipped.
                rep_scale, anc_scale = 1.0, 1.0
                if gt_rows is None:
                    gt_rows, rep_scale = _select_rows(batch, [0]), 0.0
                if draft is None:
                    draft, anc_scale = _select_rows(batch, [0]), 0.0
                l_rep = rep_scale * self.model.rollout_loss(
                    gt_rows, weights=weights if self.anchor_draft_gt else None,
                    caption_dropout_prob=self.caption_dropout_prob)
                l_anc = anc_scale * self.model.anchor_loss(draft)
                loss = l_rep + self.anchor_beta * l_anc
                running["loss_repair"] += float(l_rep.item())
                running["loss_anchor"] += float(l_anc.item())
            else:
                loss = self.model.rollout_loss(
                    batch, weights=weights, caption_dropout_prob=self.caption_dropout_prob)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip)
            grad_norm_avg = all_reduce_scalar(float(grad_norm.item()), self.device) / self.world_size
            self.opt.step()
            update_ema(self.ema, unwrap_model(self.model.net), decay=self.ema_decay)

            running["loss"] += float(loss.item())
            running["grad_norm"] += float(grad_norm_avg)
            running["mean_weight"] += (sum(weights) / len(weights)) if weights else 1.0
            n_feedback_rows = sum(1 for p in batch["chain_pos"] if int(p) > 0)
            running["caption_dropped"] += (
                getattr(self.model, "last_caption_dropped", 0) / max(n_feedback_rows, 1))
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
                    payload = {
                        "train/loss": avg["loss"],
                        "train/grad_norm": avg["grad_norm"],
                        "train/lr": self.opt.param_groups[0]["lr"],
                        "train/steps_per_sec": log_steps / elapsed,
                        "train/outer_step": outer_step,
                    }
                    if level is not None:
                        # mean_weight should read ~1.0 at every level -- it is the direct check that
                        # the per-level weights were rescaled correctly and that the curriculum is
                        # not silently changing the gradient scale between levels.
                        payload["train/level"] = self.curriculum.index(level) + 1
                        payload["train/chain_length"] = int(level["length"])
                        payload["train/mean_weight"] = avg["mean_weight"]
                    if self.caption_dropout_prob > 0.0:
                        payload["train/caption_dropped_frac"] = avg["caption_dropped"]
                    if self.anchor_beta > 0.0:
                        # loss_anchor is 0 at init (LoRA B=0 => the adapter is the identity);
                        # it growing means the draft is drifting away from the frozen base.
                        payload["train/loss_repair"] = avg["loss_repair"]
                        payload["train/loss_anchor"] = avg["loss_anchor"]
                        payload["train/anchor_beta"] = self.anchor_beta
                    wandb.log(payload, step=self.train_steps)
                running = {key: 0.0 for key in running}
                log_steps = 0
                log_start = time()

            # eval and checkpointing are decoupled: eval every rollout (cheap, gives the TTS curve),
            # save far less often (each checkpoint is ~1GB of LoRA weights).
            if self.train_steps % self.eval_every == 0:
                eval_tokens = self.eval_step()
                if rank_is_zero() and eval_tokens:
                    self.cumulative_verifier_tokens += int(eval_tokens)
                    wandb.log({"verifier/total_tokens": self.cumulative_verifier_tokens}, step=self.train_steps)
                dist.barrier()
            if self.train_steps % self.ckpt_every == 0:
                self.save()
                dist.barrier()

            if self.train_steps >= self.total_steps:
                break

    @torch.no_grad()
    def log_on_policy_grid(self, outer_step, history=5):
        """Drift probe, run at the start of every rollout: render the SAME val prompts from the SAME
        noise with the current rollout policy, then plot prompts x last-`history` rollouts so policy
        drift is visible at a glance. Caller holds the EMA context, matching what rollouts sample."""
        if not rank_is_zero() or self.val_dataset is None or len(self.val_dataset) == 0:
            return
        count = min(int(self.rollout_cfg.log_samples), 8, len(self.val_dataset))
        if count <= 0:
            return
        captions = [self.val_dataset[idx]["caption"] for idx in range(count)]
        gt_images = [self.val_dataset.image_for_index(idx) for idx in range(count)]
        images = self.model.generate(
            {
                "caption": captions,
                "feedback_history": [[] for _ in captions],
                "attempt_images": [[] for _ in captions],
                "attempt_paths": [[] for _ in captions],
            },
            num_sampling_steps=int(self.sampler_cfg.num_sampling_steps),
            cfg_scale=float(self.sampler_cfg.cfg_scale),
            ddim_eta=float(self.sampler_cfg.ddim_eta),
            seed=self.global_seed,  # fixed noise: differences across columns are pure policy drift
        )
        grid_dir = self.log_dir / "on_policy_grid"
        grid_dir.mkdir(parents=True, exist_ok=True)
        for idx, image in enumerate(images):
            image.save(grid_dir / f"outer_{outer_step:06d}_p{idx:02d}.png")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        outers = sorted({int(p.name.split("_")[1]) for p in grid_dir.glob("outer_*_p00.png")})[-int(history):]
        cols = 1 + len(outers)  # col 0 = GT render of the prompt, then the last `history` rollout policies
        fig, axes = plt.subplots(count, cols, figsize=(2.4 * cols, 2.6 * count), squeeze=False)
        for row in range(count):
            axes[row][0].imshow(gt_images[row])
            if row == 0:
                axes[row][0].set_title("GT", fontsize=9)
            for col, outer in enumerate(outers, start=1):
                ax = axes[row][col]
                path = grid_dir / f"outer_{outer:06d}_p{row:02d}.png"
                if path.exists():
                    ax.imshow(plt.imread(path))
                if row == 0:
                    ax.set_title(f"outer {outer}", fontsize=9)
            for col in range(cols):
                axes[row][col].set_xticks([])
                axes[row][col].set_yticks([])
        fig.suptitle("on-policy drift: fixed prompts / fixed noise, first-step generations", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(grid_dir / "on_policy_grid.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        wandb.log({"rollout/on_policy_grid": wandb.Image(str(grid_dir / "on_policy_grid.png"))},
                  step=self.train_steps)

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
        # batch_size <= 0 evaluates the ENTIRE val split (test-time-scaling curve over all prompts).
        batch_size = int(self.eval_cfg.batch_size)
        batch_size = len(self.val_dataset) if batch_size <= 0 else max(1, batch_size)
        indices = [(start_idx + offset) % len(self.val_dataset) for offset in range(batch_size)]
        captions = [self.val_dataset[idx]["caption"] for idx in indices]
        gt_images = [self.val_dataset.image_for_index(idx) for idx in indices]

        # Chunked: adaptive_eval generates every prompt in ONE batch, so large eval sets must be
        # split to stay inside generation memory (64/rank is the proven limit).
        chunk = int(getattr(self.eval_cfg, "gen_batch", 32))
        traces = []
        eval_tokens = 0
        for lo in range(0, len(captions), chunk):
            chunk_traces, _histories, chunk_tokens = adaptive_eval(
                self.model,
                self.verifier,
                captions[lo:lo + chunk],
                gt_images[lo:lo + chunk],
                steps=max(1, int(self.eval_cfg.steps)),
                seed=int(self.eval_cfg.seed) + self.train_steps * 1000 + lo,
                scorer=self.scorer,
                sampler_cfg=self.sampler_cfg,
            )
            traces.extend(chunk_traces)
            eval_tokens += chunk_tokens

        grid_path = self.log_dir / "adaptive_eval" / f"step_{self.train_steps:07d}.png"
        metrics = {}
        grid_n = min(len(traces), 16)  # the metrics use every trace; the image grid only needs a sample
        if save_adaptive_trace_grid(grid_path, traces[:grid_n], gt_images[:grid_n], captions[:grid_n]) is not None:
            metrics["eval/adaptive_traces"] = wandb.Image(str(grid_path))
        for key, value in distance_metrics(traces).items():
            metrics[f"eval/{key}"] = value

        # Test-time-scaling curve: mean eval score (+/- standard error) at each rollout step, over the
        # evaluated prompts. Shows whether extra feedback steps keep buying image quality.
        by_step = {}
        for trace in traces:
            for step, entry in enumerate(trace):
                if entry["distance"] is not None:
                    by_step.setdefault(step, []).append(float(entry["distance"]))
        if by_step:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            steps_axis = sorted(by_step)
            means = [sum(by_step[s]) / len(by_step[s]) for s in steps_axis]
            stderrs = [
                (sum((v - m) ** 2 for v in by_step[s]) / max(len(by_step[s]) - 1, 1)) ** 0.5
                / max(len(by_step[s]), 1) ** 0.5
                for s, m in zip(steps_axis, means)
            ]
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.errorbar(steps_axis, means, yerr=stderrs, marker="o", capsize=3)
            ax.set_xlabel("rollout step")
            ax.set_ylabel("mean score (higher = better)")
            ax.set_xticks(steps_axis)
            ax.set_title(f"test-time scaling (n={len(traces)} prompts)")
            ax.grid(alpha=0.3)
            tts_path = self.log_dir / "adaptive_eval" / f"tts_{self.train_steps:07d}.png"
            fig.tight_layout()
            fig.savefig(tts_path, dpi=140)
            plt.close(fig)
            metrics["eval/test_time_scaling"] = wandb.Image(str(tts_path))
            for s, m, se in zip(steps_axis, means, stderrs):
                metrics[f"eval/tts_step_{s}_stderr"] = se

        wandb.log(metrics, step=self.train_steps)
        return eval_tokens

    def _restore(self, path):
        """Resume optimizer/EMA/step from a checkpoint written by save().

        train.py already restores the *model* weights from the same file; without this the
        optimizer moments and EMA silently cold-start on every preemption, and train_steps resets.
        `max_train_steps` stays an absolute target, so a resumed run stops where it would have.
        """
        payload = torch.load(path, map_location="cpu")
        if "opt" in payload:
            self.opt.load_state_dict(payload["opt"])
        if "ema" in payload:
            if self.model.lora_finetune:
                from peft import set_peft_model_state_dict
                set_peft_model_state_dict(self.ema, payload["ema"])
            else:
                self.ema.load_state_dict(payload["ema"], strict=False)
        if "train_steps" in payload:
            self.train_steps = int(payload["train_steps"])
            self.start_step = self.train_steps
        # Older checkpoints predate these keys; defaulting to 0 reproduces the previous behaviour.
        self.outer_step = int(payload.get("outer_step", 0))
        self.cumulative_verifier_tokens = int(payload.get("verifier_tokens", 0))
        rng = payload.get("torch_rng")
        if rng is not None:
            torch.set_rng_state(rng.to(torch.uint8) if hasattr(rng, "to") else rng)
        self.logger.info(
            f"Resumed from {path}: train_steps={self.train_steps} outer_step={self.outer_step} "
            f"(opt={'opt' in payload}, ema={'ema' in payload}, rng={rng is not None}), "
            f"target={self.total_steps}")

    def save_backup(self):
        """Rolling latest-only resume point, written after every outer iteration.

        Numbered checkpoints are milestones (kept, every ckpt_every steps); this is the thing a
        requeue actually resumes from, so it is written far more often and overwritten in place --
        one file, constant disk. It carries everything `learn()` needs to continue where it stopped:
        LoRA weights, EMA, optimizer moments, train_steps, the outer-iteration counter (which seeds
        the rollout noise), the verifier token tally, and the torch RNG state.

        The wandb run id travels in a separate one-line sidecar so train.py can re-attach to the
        same run BEFORE building the trainer, without loading a multi-GB file to read one string.
        """
        if not rank_is_zero():
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model.checkpoint_state(),
            "ema": self.model.lora_state_of(self.ema),
            "opt": self.opt.state_dict(),
            "train_steps": self.train_steps,
            "outer_step": self.outer_step,
            "verifier_tokens": self.cumulative_verifier_tokens,
            "torch_rng": torch.get_rng_state(),
        }
        encoder_state = self.model.encoder_state()
        if encoder_state is not None:
            payload["context_encoder"] = encoder_state
        save_checkpoint_atomic(payload, self.log_dir / "resume.bkp")
        run_id = getattr(getattr(wandb, "run", None), "id", None)
        if run_id:
            (self.log_dir / "wandb_run_id.txt").write_text(str(run_id))
        self.logger.info(
            f"Backup written at step={self.train_steps} outer={self.outer_step}")

    def save(self):
        if not rank_is_zero():
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model.checkpoint_state(),
            "ema": self.model.lora_state_of(self.ema),
            "opt": self.opt.state_dict(),
            "train_steps": self.train_steps,
            "outer_step": self.outer_step,
            "verifier_tokens": self.cumulative_verifier_tokens,
            "torch_rng": torch.get_rng_state(),
        }
        encoder_state = self.model.encoder_state()
        if encoder_state is not None:
            payload["context_encoder"] = encoder_state
        checkpoint_path = self.checkpoint_dir / f"{self.train_steps:07d}.pt"
        save_checkpoint_atomic(payload, checkpoint_path)
        save_checkpoint_atomic(self.model.lora_state_of(self.ema),
                               self.checkpoint_dir / f"{self.train_steps:07d}-ema.pt")
        self.logger.info(f"Saved checkpoint at step={self.train_steps}")
