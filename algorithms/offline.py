"""Supervised diffusion training on a fixed dataset."""
from copy import deepcopy
from pathlib import Path
from time import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

import wandb
from algorithms.eval import build_eval_datasets, run_checkpoint_eval
from algorithms.utils import (
    build_optimizer_scheduler,
    create_logger,
    dataloader_kwargs,
    progress_bar,
    rank_is_zero,
    requires_grad,
    save_checkpoint_atomic,
    unwrap_model,
    update_ema,
)
from datasets.clevr.dataset import DistributedWeightedSampler


class OfflineTrainer:
    """Trains or finetunes a model on a fixed dataset with DDP, EMA, and a cosine LR schedule."""

    def __init__(
        self,
        model,
        dataset,
        device,
        log_dir,
        epochs,
        max_train_steps,
        global_batch_size,
        global_seed,
        lr,
        lr_schedule,
        lr_warmup_steps,
        min_lr,
        weight_decay,
        grad_clip,
        ema_decay,
        log_every,
        ckpt_every,
        dataloader,
        eval,
        start_step=0,
    ):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = self.log_dir / "checkpoints"
        self.epochs = int(epochs)
        self.max_train_steps = int(max_train_steps)
        self.global_seed = int(global_seed)
        self.grad_clip = float(grad_clip)
        self.ema_decay = float(ema_decay)
        self.log_every = int(log_every)
        self.ckpt_every = int(ckpt_every)
        self.eval_cfg = eval
        self.train_steps = int(start_step)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.logger = create_logger(str(self.log_dir), self.rank)

        self.ema = None
        if self.ema_decay > 0:
            self.ema = deepcopy(unwrap_model(model.net))
            requires_grad(self.ema, False)
            self.ema.eval()
        model.net.train()

        self.params = model.trainable_parameters()
        self.opt, self.scheduler = build_optimizer_scheduler(
            self.params,
            lr=lr,
            weight_decay=weight_decay,
            schedule=lr_schedule,
            total_steps=self.max_train_steps,
            warmup_steps=lr_warmup_steps,
            min_lr=min_lr,
        )

        train_dataset, self.collate = model.prepare_dataset(dataset)
        self.sampler = DistributedWeightedSampler(
            dataset.sample_weights,
            num_replicas=self.world_size,
            rank=self.rank,
            seed=self.global_seed,
        )
        self.loader = DataLoader(
            train_dataset,
            **dataloader_kwargs(
                dataloader,
                batch_size=int(global_batch_size) // self.world_size,
                shuffle=False,
                sampler=self.sampler,
                drop_last=dataloader.drop_last,
                collate_fn=self.collate,
            ),
        )
        self.eval_datasets, self.eval_names = build_eval_datasets(dataset, eval.split)
        self.logger.info(
            f"Dataset contains {len(dataset):,} rows from {len(dataset.datasets)} sources; "
            f"eval datasets: {self.eval_names}"
        )
        self.logger.info(f"Trainable parameters: {sum(p.numel() for p in model.net.parameters() if p.requires_grad):,}")

    def learn(self):
        running = {key: 0.0 for key in ("loss", "grad_norm", "lr")}
        running_sources = torch.zeros(len(self.dataset.datasets), dtype=torch.float64, device=self.device)
        running_samples = 0
        log_steps = 0
        start_time = time()

        self.logger.info(f"Training for up to {self.max_train_steps} steps ({self.epochs} epochs max)...")
        for epoch in range(self.epochs):
            self.sampler.set_epoch(epoch)
            for batch in progress_bar(self.loader, desc=f"epoch {epoch}"):
                loss = self.model.loss(batch)
                self.opt.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip)
                lr = self.opt.param_groups[0]["lr"]
                self.opt.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                if self.ema is not None:
                    update_ema(self.ema, unwrap_model(self.model.net), decay=self.ema_decay)

                source_index = batch["source_index"].to(self.device)
                running["loss"] += loss.item()
                running["grad_norm"] += float(grad_norm)
                running["lr"] += lr
                running_sources += torch.bincount(source_index, minlength=len(self.dataset.datasets)).to(running_sources.dtype)
                running_samples += int(source_index.numel())
                log_steps += 1
                self.train_steps += 1

                if self.train_steps % self.log_every == 0:
                    torch.cuda.synchronize()
                    steps_per_sec = log_steps / (time() - start_time)
                    values = torch.tensor(
                        [running["loss"], running["grad_norm"], running["lr"], float(running_samples)],
                        device=self.device,
                        dtype=torch.float64,
                    )
                    source_counts = running_sources.clone()
                    dist.all_reduce(values, op=dist.ReduceOp.SUM)
                    dist.all_reduce(source_counts, op=dist.ReduceOp.SUM)
                    avg_loss = values[0].item() / (log_steps * self.world_size)
                    avg_grad = values[1].item() / (log_steps * self.world_size)
                    avg_lr = values[2].item() / (log_steps * self.world_size)
                    source_fracs = (source_counts / max(values[3].item(), 1)).cpu().tolist()
                    self.logger.info(
                        f"(step={self.train_steps:07d}) Loss: {avg_loss:.4f}, "
                        f"Grad Norm: {avg_grad:.4f}, LR: {avg_lr:.6g}, Steps/Sec: {steps_per_sec:.2f}"
                    )
                    if rank_is_zero():
                        payload = {
                            "train/loss": avg_loss,
                            "train/grad_norm": avg_grad,
                            "train/lr": avg_lr,
                            "train/steps_per_sec": steps_per_sec,
                            "train/epoch": epoch,
                        }
                        for name, frac in zip(self.dataset.names, source_fracs):
                            payload[f"train/source_fraction/{name}"] = frac
                        wandb.log(payload, step=self.train_steps)
                    running = {key: 0.0 for key in running}
                    running_sources.zero_()
                    running_samples = 0
                    log_steps = 0
                    start_time = time()

                if self.train_steps % self.ckpt_every == 0:
                    self.eval_step()
                    self.save()
                    dist.barrier()

                if self.train_steps >= self.max_train_steps:
                    self.logger.info("Done!")
                    return
        self.logger.info("Done!")

    def eval_step(self):
        run_checkpoint_eval(
            self.model,
            self.eval_datasets,
            self.eval_names,
            self.eval_cfg,
            self.device,
            self.logger,
            self.train_steps,
        )

    def save(self):
        if not rank_is_zero():
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {"model": self.model.checkpoint_state(), "opt": self.opt.state_dict()}
        if self.ema is not None:
            payload["ema"] = self.ema.state_dict()
        if self.scheduler is not None:
            payload["scheduler"] = self.scheduler.state_dict()
        encoder_state = self.model.encoder_state()
        if encoder_state is not None:
            payload["context_encoder"] = encoder_state
        checkpoint_path = self.checkpoint_dir / f"{self.train_steps:07d}.pt"
        save_checkpoint_atomic(payload, checkpoint_path)
        if self.ema is not None:
            save_checkpoint_atomic(self.ema.state_dict(), self.checkpoint_dir / f"{self.train_steps:07d}-ema.pt")
        self.model.save_extras(self.checkpoint_dir)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
