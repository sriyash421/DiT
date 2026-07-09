import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets_clevr import pad_contexts
from vlm_utils import build_context_text, build_history_context_text, encode_contexts, load_vlm


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def tensor_to_pil(x):
    x = x.detach().float().cpu().clamp(-1, 1)
    x = (x + 1) / 2
    arr = (x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def normalized_tensor_to_pil(x):
    x = x.detach().float().cpu().clamp(-1, 1)
    x = ((x + 1) / 2 * 255).round().byte()
    return Image.fromarray(x.permute(1, 2, 0).numpy(), mode="RGB")


def load_metadata_for_zarr(dataset_path):
    path = Path(dataset_path)
    candidates = []
    if path.name == "data.zarr":
        candidates.append(path.parent / "metadata.jsonl")
    candidates.append(path / "metadata.jsonl")
    for candidate in candidates:
        if candidate.exists():
            rows = []
            with candidate.open() as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
            return rows
    return []


def metadata_by_index(metadata_rows, metadata_indices):
    out = []
    for idx in metadata_indices:
        idx = int(idx)
        out.append(metadata_rows[idx] if 0 <= idx < len(metadata_rows) else None)
    return out


def _torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class ContextEncoder:
    def encode_feedback(self, captions, metadata, feedback, attempt_images):
        raise NotImplementedError


class QwenContextEncoder(ContextEncoder):
    def __init__(
        self,
        model_id,
        device,
        vlm_dtype="bfloat16",
        device_map="auto",
        max_length=1024,
        out_dtype=torch.float16,
        include_metadata=True,
    ):
        self.device = torch.device(device)
        self.max_length = int(max_length)
        self.out_dtype = out_dtype
        self.include_metadata = bool(include_metadata)
        self.processor, self.model = load_vlm(model_id, self.device, dtype=vlm_dtype, device_map=device_map)

    @torch.no_grad()
    def encode_feedback(self, captions, metadata, feedback, attempt_images):
        texts = [
            build_context_text(caption, meta, fb, include_metadata=self.include_metadata)
            for caption, meta, fb in zip(captions, metadata, feedback)
        ]
        tokens, masks = encode_contexts(
            self.processor,
            self.model,
            texts,
            self.device,
            images=attempt_images,
            max_length=self.max_length,
            out_dtype=self.out_dtype,
        )
        contexts = [token[mask.bool()].contiguous() for token, mask in zip(tokens, masks)]
        return pad_contexts(contexts)

    @torch.no_grad()
    def encode_history(self, captions, feedback_histories, attempt_image_histories):
        texts = [
            build_history_context_text(caption, history)
            for caption, history in zip(captions, feedback_histories)
        ]
        tokens, masks = encode_contexts(
            self.processor,
            self.model,
            texts,
            self.device,
            images=attempt_image_histories,
            max_length=self.max_length,
            out_dtype=self.out_dtype,
        )
        contexts = [token[mask.bool()].contiguous() for token, mask in zip(tokens, masks)]
        return pad_contexts(contexts)


class PolicySampler:
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


def diffusion_loss(model, diffusion, x_latent, context_tokens, context_mask):
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


class RolloutBuffer(Dataset):
    def __init__(self, path):
        self.path = Path(path)
        payloads = self._load_payloads(self.path)
        self.records = []
        self.stats = {"attempted": 0, "success": 0, "failed": 0, "gemini_tokens": 0}
        for payload in payloads:
            self.records.extend(payload["records"])
            for key, value in payload.get("stats", {}).items():
                if key in self.stats:
                    self.stats[key] += value
        self.stats["success"] = len(self.records)

    @staticmethod
    def _load_payloads(path):
        if (path / "records.pt").exists():
            return [_torch_load(path / "records.pt")]
        record_paths = sorted(path.glob("rank_*/records.pt"))
        if not record_paths:
            raise FileNotFoundError(f"No rollout records found under {path}")
        return [_torch_load(record_path) for record_path in record_paths]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[int(idx)]


def rollout_collate(batch):
    base_tokens, base_mask = pad_contexts([item["base_context_tokens"] for item in batch])
    adaptive_tokens, adaptive_mask = pad_contexts([item["adaptive_context_tokens"] for item in batch])
    return {
        "x_latent": torch.stack([item["x_latent"] for item in batch]),
        "base_context_tokens": base_tokens,
        "base_context_mask": base_mask,
        "adaptive_context_tokens": adaptive_tokens,
        "adaptive_context_mask": adaptive_mask,
        "caption": [item["caption"] for item in batch],
        "feedback": [item["feedback"] for item in batch],
        "metadata_index": torch.tensor([item["metadata_index"] for item in batch], dtype=torch.long),
        "attempt_path": [item["attempt_path"] for item in batch],
    }


def _slice_batch(batch, count):
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value[:count]
        elif isinstance(value, list):
            out[key] = value[:count]
        else:
            out[key] = value
    return out


class RolloutCollector:
    def __init__(
        self,
        model,
        vae,
        sampler,
        verifier,
        context_encoder,
        metadata_rows,
        rollout_length=1,
    ):
        self.model = model
        self.vae = vae
        self.sampler = sampler
        self.verifier = verifier
        self.context_encoder = context_encoder
        self.metadata_rows = metadata_rows
        self.rollout_length = max(1, int(rollout_length))

    @torch.no_grad()
    def collect(self, loader, output_dir, sample_count, device, seed=None, progress=None, data_sampler=None, epoch=0):
        if self.rollout_length <= 1:
            return self._collect_one_step(loader, output_dir, sample_count, device, seed, progress, data_sampler, epoch)
        return self._collect_multi_step(loader, output_dir, sample_count, device, seed, progress, data_sampler, epoch)

    @torch.no_grad()
    def _collect_one_step(self, loader, output_dir, sample_count, device, seed=None, progress=None, data_sampler=None, epoch=0):
        output_dir = Path(output_dir)
        attempts_dir = output_dir / "attempts"
        gt_dir = output_dir / "gt"
        attempts_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        records = []
        attempted = 0
        failed = 0
        token_count = 0
        current_epoch = int(epoch)
        if data_sampler is not None:
            data_sampler.set_epoch(current_epoch)
        iterator = iter(loader)

        while attempted < sample_count:
            try:
                batch = next(iterator)
            except StopIteration:
                current_epoch += 1
                if data_sampler is not None:
                    data_sampler.set_epoch(current_epoch)
                iterator = iter(loader)
                batch = next(iterator)
            remaining = int(sample_count) - attempted
            batch = _slice_batch(batch, min(remaining, int(batch["image"].shape[0])))
            batch_size = int(batch["image"].shape[0])
            if batch_size == 0:
                continue

            x_img = batch["image"].to(device)
            base_tokens = batch["context_tokens"].to(device)
            base_mask = batch["context_mask"].to(device)
            captions = batch["caption"]
            metadata = metadata_by_index(self.metadata_rows, batch["metadata_index"].tolist())
            gt_images = [normalized_tensor_to_pil(image) for image in batch["image"]]

            _, attempt_images = self.sampler.sample(
                self.model,
                self.vae,
                base_tokens,
                base_mask,
                device,
                seed=None if seed is None else int(seed) + attempted,
            )
            results = self.verifier.verify_batch(captions, metadata, gt_images, attempt_images)
            success_indices = [idx for idx, result in enumerate(results) if result.ok]
            token_count += sum(int(result.token_count) for result in results)
            failed += len(results) - len(success_indices)
            attempted += batch_size

            if not success_indices:
                if progress is not None:
                    progress.update(batch_size)
                continue

            x_latent = self.vae.encode(x_img[success_indices]).latent_dist.sample().mul_(self.vae.config.scaling_factor)
            feedbacks = [results[idx].feedback for idx in success_indices]
            adaptive_tokens, adaptive_mask = self.context_encoder.encode_feedback(
                [captions[idx] for idx in success_indices],
                [metadata[idx] for idx in success_indices],
                feedbacks,
                [attempt_images[idx] for idx in success_indices],
            )

            for local_idx, batch_idx in enumerate(success_indices):
                record_id = len(records)
                attempt_path = attempts_dir / f"{record_id:06d}.png"
                gt_path = gt_dir / f"{record_id:06d}.png"
                attempt_images[batch_idx].save(attempt_path)
                gt_images[batch_idx].save(gt_path)
                base_valid = base_mask[batch_idx].detach().cpu().bool()
                adaptive_valid = adaptive_mask[local_idx].detach().cpu().bool()
                records.append({
                    "x_latent": x_latent[local_idx].detach().cpu().float(),
                    "base_context_tokens": base_tokens[batch_idx].detach().cpu()[base_valid].to(torch.float16),
                    "adaptive_context_tokens": adaptive_tokens[local_idx].detach().cpu()[adaptive_valid].to(torch.float16),
                    "caption": captions[batch_idx],
                    "feedback": feedbacks[local_idx],
                    "metadata_index": int(batch["metadata_index"][batch_idx].item()),
                    "gt_path": str(gt_path),
                    "attempt_path": str(attempt_path),
                })

            if progress is not None:
                progress.update(batch_size)

        stats = {
            "attempted": int(attempted),
            "success": int(len(records)),
            "failed": int(failed),
            "gemini_tokens": int(token_count),
        }
        torch.save({"records": records, "stats": stats}, output_dir / "records.pt")
        return stats

    @torch.no_grad()
    def _collect_multi_step(self, loader, output_dir, sample_count, device, seed=None, progress=None, data_sampler=None, epoch=0):
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
            base_tokens = batch["context_tokens"].to(device)
            base_mask = batch["context_mask"].to(device)
            captions = batch["caption"]
            metadata = metadata_by_index(self.metadata_rows, batch["metadata_index"].tolist())
            gt_images = [normalized_tensor_to_pil(image) for image in batch["image"]]
            x_latents = self.vae.encode(x_img).latent_dist.sample().mul_(self.vae.config.scaling_factor)

            active = list(range(batch_size))
            current_tokens = base_tokens
            current_mask = base_mask
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
                active_captions = [captions[idx] for idx in active]
                active_metadata = [metadata[idx] for idx in active]
                active_gt_images = [gt_images[idx] for idx in active]
                active_histories = [list(histories[idx]) for idx in active]
                results = self.verifier.verify_history_batch(
                    active_captions,
                    active_metadata,
                    active_gt_images,
                    attempt_images,
                    active_histories,
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
                    result = results[pos]
                    record_id = len(records)
                    attempt_path = attempts_dir / f"{record_id:06d}_step_{step_idx:02d}.png"
                    gt_path = gt_dir / f"{record_id:06d}.png"
                    attempt_images[pos].save(attempt_path)
                    if not gt_path.exists():
                        gt_images[batch_idx].save(gt_path)

                    base_valid = base_mask[batch_idx].detach().cpu().bool()
                    context_valid = current_mask[pos].detach().cpu().bool()
                    records.append({
                        "x_latent": x_latents[batch_idx].detach().cpu().float(),
                        "base_context_tokens": base_tokens[batch_idx].detach().cpu()[base_valid].to(torch.float16),
                        "adaptive_context_tokens": current_tokens[pos].detach().cpu()[context_valid].to(torch.float16),
                        "caption": captions[batch_idx],
                        "feedback": result.feedback,
                        "feedback_history": list(histories[batch_idx]),
                        "history_attempt_paths": list(history_paths[batch_idx]),
                        "step_index": int(step_idx),
                        "metadata_index": int(batch["metadata_index"][batch_idx].item()),
                        "gt_path": str(gt_path),
                        "attempt_path": str(attempt_path),
                    })

                    histories[batch_idx].append(result.feedback)
                    history_paths[batch_idx].append(str(attempt_path))
                    history_images[batch_idx].append(attempt_images[pos])
                    next_active.append(batch_idx)
                    next_captions.append(captions[batch_idx])
                    next_histories.append(list(histories[batch_idx]))
                    next_history_images.append(list(history_images[batch_idx]))

                if step_idx + 1 >= self.rollout_length or not next_active:
                    active = next_active
                    break
                current_tokens, current_mask = self.context_encoder.encode_history(
                    next_captions,
                    next_histories,
                    next_history_images,
                )
                active = next_active

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


class OnPolicyTrainer:
    def __init__(
        self,
        model,
        train_diffusion,
        feedback_weight=1.0,
        base_weight=1.0,
    ):
        self.model = model
        self.train_diffusion = train_diffusion
        self.feedback_weight = float(feedback_weight)
        self.base_weight = float(base_weight)

    def compute_loss(self, batch, device):
        x_latent = batch["x_latent"].to(device)
        base_tokens = batch["base_context_tokens"].to(device)
        base_mask = batch["base_context_mask"].to(device)
        adaptive_tokens = batch["adaptive_context_tokens"].to(device)
        adaptive_mask = batch["adaptive_context_mask"].to(device)
        if self.base_weight:
            base_loss = diffusion_loss(
                self.model,
                self.train_diffusion,
                x_latent,
                base_tokens,
                base_mask,
            )
        else:
            base_loss = x_latent.new_tensor(0.0)
        if self.feedback_weight:
            feedback_loss = diffusion_loss(
                self.model,
                self.train_diffusion,
                x_latent,
                adaptive_tokens,
                adaptive_mask,
            )
        else:
            feedback_loss = x_latent.new_tensor(0.0)
        loss = self.feedback_weight * feedback_loss + self.base_weight * base_loss
        return loss, {
            "loss": float(loss.item()),
            "feedback_loss": float(feedback_loss.item()),
            "base_loss": float(base_loss.item()),
        }


def save_trace_grid(path, gt_image, attempt1, feedback, attempt2):
    tile_w, tile_h = gt_image.size
    text_w = max(tile_w, 360)
    out = Image.new("RGB", (tile_w * 3 + text_w, tile_h), (255, 255, 255))
    out.paste(gt_image.convert("RGB").resize((tile_w, tile_h)), (0, 0))
    out.paste(attempt1.convert("RGB").resize((tile_w, tile_h)), (tile_w, 0))
    out.paste(attempt2.convert("RGB").resize((tile_w, tile_h)), (tile_w * 2, 0))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(out)
    draw.text((tile_w * 3 + 8, 8), feedback[:1000], fill=(20, 20, 20))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
