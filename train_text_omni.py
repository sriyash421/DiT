#!/usr/bin/env python3
"""Fine-tune OmniGen on CLEVR zarr rows.

This mirrors train_text.py's CLEVR/DDP training loop, but replaces the
repo-local Qwen-conditioned DiT with OmniGen's own processor, transformer, VAE,
and flow-matching training loss.
"""
import os
import random
from copy import deepcopy
from glob import glob
from time import time

import hydra
import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from diffusers.models import AutoencoderKL
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

import wandb
from datasets_clevr import ClevrContextMultiDataset, DistributedWeightedSampler
from train_utils import (
    cfg_get,
    create_logger,
    dataloader_kwargs,
    get_lr,
    rank_is_zero,
    requires_grad,
    save_checkpoint_atomic,
    set_optimizer_lr,
    update_ema,
)
from torchvision import transforms

try:
    from OmniGen import OmniGen, OmniGenProcessor
    from OmniGen.train_helper import TrainDataCollator, training_losses
    from OmniGen.utils import vae_encode, vae_encode_list
except ImportError as exc:  # pragma: no cover - depends on external OmniGen install.
    raise ImportError(
        "train_text_omni.py requires the OmniGen package. Install it first, e.g. "
        "`git clone https://github.com/VectorSpaceLab/OmniGen.git && cd OmniGen && pip install -e .`"
    ) from exc

try:
    from peft import LoraConfig, get_peft_model
    from peft.utils import get_peft_model_state_dict
except ImportError:
    LoraConfig = None
    get_peft_model = None
    get_peft_model_state_dict = None


def cleanup():
    dist.destroy_process_group()


def resolve_model_path(model_name_or_path, logger):
    if os.path.exists(model_name_or_path):
        return model_name_or_path
    cache_folder = os.getenv("HF_HUB_CACHE")
    local_path = snapshot_download(
        repo_id=model_name_or_path,
        cache_dir=cache_folder,
        ignore_patterns=["flax_model.msgpack", "rust_model.ot", "tf_model.h5", "model.pt"],
    )
    logger.info(f"Downloaded OmniGen model to {local_path}")
    return local_path


def get_model_resolution(model_cfg):
    height = cfg_get(model_cfg, "image_height", None)
    width = cfg_get(model_cfg, "image_width", None)
    if height is None and width is None:
        size = int(cfg_get(model_cfg, "image_size", 256))
        return size, size
    if height is None or width is None:
        raise ValueError("Set both model.image_height and model.image_width, or neither.")
    return int(height), int(width)


def build_omni_clevr_transform(model_cfg):
    height, width = get_model_resolution(model_cfg)
    return transforms.Compose([
        transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])


class OmniClevrDataset(Dataset):
    """Adapts ClevrContextMultiDataset rows to OmniGen training examples."""

    def __init__(
        self,
        dataset,
        processor,
        image_transform,
        condition_dropout_prob=0.0,
        max_input_length_limit=18000,
        use_feedback_images=True,
    ):
        self.dataset = dataset
        self.processor = processor
        self.image_transform = image_transform
        self.condition_dropout_prob = float(condition_dropout_prob)
        self.max_input_length_limit = int(max_input_length_limit)
        self.use_feedback_images = bool(use_feedback_images)

    def __len__(self):
        return len(self.dataset)

    def _generated_image_for_global_index(self, idx):
        dataset_idx, local_idx = self.dataset.global_to_local(idx)
        source = self.dataset.datasets[dataset_idx]
        row_idx = int(source.indices[int(local_idx)])
        generated_idx = int(source._data["generated_image_index"][row_idx])
        if generated_idx < 0:
            return None
        if source._generated_image_cache is not None and generated_idx in source._generated_image_cache:
            arr = np.asarray(source._generated_image_cache[generated_idx])
        else:
            arr = np.asarray(source._data["generated_images"][generated_idx])
        return Image.fromarray(arr, mode="RGB")

    @staticmethod
    def _build_base_instruction(item):
        return item["caption"]

    @staticmethod
    def _build_feedback_instruction(item):
        caption = item["caption"].strip()
        feedback = item["feedback"].strip()
        if feedback:
            return (
                "<|image_1|> Edit the input image so it matches this CLEVR description: "
                f"{caption}. Apply this feedback: {feedback}"
            )
        return f"<|image_1|> Edit the input image so it matches this CLEVR description: {caption}."

    def _make_example(self, idx):
        item = self.dataset[idx]
        output_image = item["image"]
        input_images = None
        instruction = self._build_base_instruction(item)

        if self.use_feedback_images and item["is_feedback"]:
            generated = self._generated_image_for_global_index(idx)
            if generated is not None:
                instruction = self._build_feedback_instruction(item)
                input_images = [self.image_transform(generated)]

        if random.random() < self.condition_dropout_prob:
            instruction = ""
            input_images = None

        mllm_input = self.processor.process_multi_modal_prompt(instruction, input_images)
        if len(mllm_input["input_ids"]) > self.max_input_length_limit:
            raise RuntimeError(
                f"input token count {len(mllm_input['input_ids'])} exceeds "
                f"max_input_length_limit={self.max_input_length_limit}"
            )
        return {
            "mllm_input": mllm_input,
            "output_image": output_image,
            "source_index": item.get("source_index", 0),
            "is_feedback": item["is_feedback"],
        }

    def __getitem__(self, idx):
        for _ in range(8):
            try:
                return self._make_example(int(idx))
            except Exception as exc:
                print(f"error when loading OmniGen CLEVR row {idx}: {exc}")
                idx = random.randint(0, len(self) - 1)
        raise RuntimeError("Too many bad OmniGen CLEVR rows.")


class OmniClevrCollator:
    def __init__(self, processor, hidden_size, keep_raw_resolution=False):
        self.inner = TrainDataCollator(
            pad_token_id=processor.text_tokenizer.eos_token_id,
            hidden_size=hidden_size,
            keep_raw_resolution=keep_raw_resolution,
        )

    def __call__(self, features):
        data = self.inner([(item["mllm_input"], item["output_image"]) for item in features])
        data["source_index"] = torch.tensor([item["source_index"] for item in features], dtype=torch.long)
        data["is_feedback"] = torch.tensor([item["is_feedback"] for item in features], dtype=torch.bool)
        return data


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    return value


def move_model_kwargs_to_device(model_kwargs, device):
    return {key: move_to_device(value, device) for key, value in model_kwargs.items()}


def encode_images(vae, output_images, input_pixel_values, weight_dtype):
    with torch.no_grad():
        if isinstance(output_images, list):
            output_latents = vae_encode_list(vae, output_images, weight_dtype)
            input_latents = vae_encode_list(vae, input_pixel_values, weight_dtype) if input_pixel_values is not None else None
        else:
            output_latents = vae_encode(vae, output_images, weight_dtype)
            input_latents = vae_encode(vae, input_pixel_values, weight_dtype) if input_pixel_values is not None else None
    return output_latents, input_latents


def maybe_apply_lora(model, cfg, logger):
    omni_cfg = cfg_get(cfg, "omni", {})
    use_lora = bool(cfg_get(omni_cfg, "use_lora", True))
    if not use_lora:
        return model
    if LoraConfig is None or get_peft_model is None:
        raise ImportError("omni.use_lora=true requires `peft` to be installed.")
    requires_grad(model, False)
    target_modules = list(cfg_get(omni_cfg, "lora_target_modules", ["qkv_proj", "o_proj"]))
    rank = int(cfg_get(omni_cfg, "lora_rank", 8))
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=int(cfg_get(omni_cfg, "lora_alpha", rank)),
        init_lora_weights=cfg_get(omni_cfg, "lora_init", "gaussian"),
        target_modules=target_modules,
    )
    model.llm.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    logger.info(f"Enabled OmniGen LoRA rank={rank}, target_modules={target_modules}")
    return model


@hydra.main(config_path="configs", config_name="finetune_omni", version_base=None)
def main(cfg):
    assert torch.cuda.is_available(), "Training requires at least one GPU."
    dist.init_process_group("nccl")
    assert cfg.train.global_batch_size % dist.get_world_size() == 0, "Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    seed = cfg.train.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    random.seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    omni_cfg = cfg_get(cfg, "omni", {})
    model_name = cfg_get(omni_cfg, "model_name_or_path", "Shitao/OmniGen-v1")

    if rank_is_zero():
        os.makedirs(cfg.train.results_dir, exist_ok=True)
        experiment_name = cfg.train.experiment_name
        if experiment_name is None:
            clean_model_name = os.path.basename(str(model_name).rstrip("/")).replace("/", "-")
            experiment_name = f"{len(glob(f'{cfg.train.results_dir}/*')):03d}-{clean_model_name}-omni"
        experiment_dir = f"{cfg.train.results_dir}/{experiment_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        OmegaConf.save(cfg, f"{experiment_dir}/config.yaml")
        logger.info(f"Experiment directory created at {experiment_dir}")
        wandb.init(project=cfg.wandb.project, name=os.path.basename(experiment_dir), config=resolved_cfg)
    else:
        checkpoint_dir = None
        logger = create_logger(None)

    model_path = resolve_model_path(model_name, logger)
    processor = OmniGenProcessor.from_pretrained(model_path)
    model = OmniGen.from_pretrained(model_path)
    model.llm.config.use_cache = False
    if bool(cfg_get(omni_cfg, "gradient_checkpointing", True)):
        model.llm.gradient_checkpointing_enable()

    model = maybe_apply_lora(model, cfg, logger)

    weight_dtype = torch.float32
    mixed_precision = cfg_get(omni_cfg, "mixed_precision", "bf16")
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    model = model.to(device=device, dtype=weight_dtype)

    vae_path = cfg_get(omni_cfg, "vae", None)
    if vae_path is None:
        local_vae_path = os.path.join(model_path, "vae")
        vae_path = local_vae_path if os.path.exists(local_vae_path) else cfg.model.vae
    vae = AutoencoderKL.from_pretrained(vae_path).to(device=device, dtype=torch.float32)
    requires_grad(vae, False)
    vae.eval()
    logger.info(f"Loaded OmniGen model {model_name}")
    logger.info(f"Loaded VAE {vae_path}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    ema = None
    if bool(cfg_get(omni_cfg, "use_ema", False)):
        ema = deepcopy(model).to(device)
        requires_grad(ema, False)
        ema.eval()

    image_height, image_width = get_model_resolution(cfg.model)
    transform = build_omni_clevr_transform(cfg.model)
    base_dataset = ClevrContextMultiDataset(
        cfg.data.dataset_config,
        transform=transform,
        split=cfg.data.split,
        use_disk=cfg_get(cfg.data, "use_disk", True),
        load_meta=cfg_get(cfg.data, "load_meta", True),
        load_images=cfg_get(cfg.data, "load_images", False),
        load_context=cfg_get(cfg.data, "load_context", False),
    )
    dataset = OmniClevrDataset(
        base_dataset,
        processor=processor,
        image_transform=transform,
        condition_dropout_prob=float(cfg_get(omni_cfg, "condition_dropout_prob", cfg.model.context_dropout_prob)),
        max_input_length_limit=int(cfg_get(omni_cfg, "max_input_length_limit", 18000)),
        use_feedback_images=bool(cfg_get(omni_cfg, "use_feedback_images", True)),
    )
    collate_fn = OmniClevrCollator(
        processor,
        hidden_size=model.llm.config.hidden_size,
        keep_raw_resolution=bool(cfg_get(omni_cfg, "keep_raw_resolution", True)),
    )
    sampler = DistributedWeightedSampler(
        base_dataset.sample_weights,
        num_replicas=dist.get_world_size(),
        rank=rank,
        seed=cfg.train.global_seed,
    )
    loader = DataLoader(
        dataset,
        **dataloader_kwargs(
            cfg.dataloader,
            batch_size=cfg.train.global_batch_size // dist.get_world_size(),
            shuffle=False,
            sampler=sampler,
            drop_last=cfg.dataloader.drop_last,
            collate_fn=collate_fn,
        ),
    )
    logger.info(f"Dataset contains {len(dataset):,} rows; split={cfg.data.split}; resize={image_width}x{image_height}")

    if ema is not None:
        update_ema(ema, model, decay=0)
    model = DDP(model, device_ids=[device], find_unused_parameters=False)
    model.train()

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.train.lr,
        weight_decay=float(cfg_get(cfg.train, "adam_weight_decay", 0.0)),
    )

    train_steps = 0
    log_steps = 0
    running = {key: 0.0 for key in ("loss", "grad_norm", "lr")}
    running_sources = torch.zeros(len(base_dataset.datasets), dtype=torch.float64, device=device)
    running_samples = 0
    start_time = time()
    done_training = False
    logger.info(f"Training OmniGen for {cfg.train.epochs} epochs...")

    for epoch in range(cfg.train.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for batch in loader:
            output_images = move_to_device(batch["output_images"], device)
            input_pixel_values = move_to_device(batch["input_pixel_values"], device)
            output_latents, input_latents = encode_images(vae, output_images, input_pixel_values, weight_dtype)

            model_kwargs = {
                "input_ids": batch["input_ids"],
                "input_img_latents": input_latents,
                "input_image_sizes": batch["input_image_sizes"],
                "attention_mask": batch["attention_mask"],
                "position_ids": batch["position_ids"],
                "padding_latent": batch["padding_images"],
                "past_key_values": None,
                "return_past_key_values": False,
            }
            model_kwargs = move_model_kwargs_to_device(model_kwargs, device)
            loss_vec = training_losses(model, output_latents, model_kwargs)["loss"]
            loss = loss_vec.mean()

            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            lr = get_lr(cfg, train_steps)
            set_optimizer_lr(opt, lr)
            opt.step()
            if ema is not None:
                update_ema(ema, model.module, decay=cfg.train.ema_decay)

            source_index = batch["source_index"].to(device)
            running["loss"] += loss.item()
            running["grad_norm"] += float(grad_norm)
            running["lr"] += lr
            running_sources += torch.bincount(source_index, minlength=len(base_dataset.datasets)).to(running_sources.dtype)
            running_samples += int(source_index.numel())
            log_steps += 1
            train_steps += 1

            if train_steps % cfg.train.log_every == 0:
                torch.cuda.synchronize()
                steps_per_sec = log_steps / (time() - start_time)
                avg_loss = torch.tensor(running["loss"] / log_steps, device=device)
                avg_grad = torch.tensor(running["grad_norm"] / log_steps, device=device)
                avg_lr = torch.tensor(running["lr"] / log_steps, device=device)
                source_counts = running_sources.clone()
                sample_count = torch.tensor(float(running_samples), device=device)
                for value in (avg_loss, avg_grad, avg_lr, source_counts, sample_count):
                    dist.all_reduce(value, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                avg_grad = avg_grad.item() / dist.get_world_size()
                avg_lr = avg_lr.item() / dist.get_world_size()
                source_fracs = (source_counts / sample_count.clamp_min(1)).detach().cpu().tolist()
                logger.info(
                    f"(step={train_steps:07d}) Loss: {avg_loss:.4f}, "
                    f"Grad Norm: {avg_grad:.4f}, LR: {avg_lr:.6g}, Steps/Sec: {steps_per_sec:.2f}"
                )
                if rank_is_zero():
                    log_payload = {
                        "train/loss": avg_loss,
                        "train/grad_norm": avg_grad,
                        "train/lr": avg_lr,
                        "train/steps_per_sec": steps_per_sec,
                        "train/epoch": epoch,
                    }
                    for idx, frac in enumerate(source_fracs):
                        log_payload[f"train/source_fraction/dataset_{idx}"] = frac
                        log_payload[f"train/source_fraction/{base_dataset.names[idx]}"] = frac
                    wandb.log(log_payload, step=train_steps)
                running = {key: 0.0 for key in running}
                running_sources.zero_()
                running_samples = 0
                log_steps = 0
                start_time = time()

            if train_steps % cfg.train.ckpt_every == 0 and train_steps > 0:
                if rank_is_zero():
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}"
                    os.makedirs(checkpoint_path, exist_ok=True)
                    if bool(cfg_get(omni_cfg, "use_lora", True)):
                        state_dict = get_peft_model_state_dict(model.module)
                        save_checkpoint_atomic(state_dict, f"{checkpoint_path}/adapter_model.pt")
                    else:
                        save_checkpoint_atomic(model.module.state_dict(), f"{checkpoint_path}/model.pt")
                    processor.text_tokenizer.save_pretrained(checkpoint_path)
                    model.module.llm.config.save_pretrained(checkpoint_path)
                    if ema is not None:
                        save_checkpoint_atomic(ema.state_dict(), f"{checkpoint_path}/ema.pt")
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

            if cfg.train.max_train_steps is not None and train_steps >= cfg.train.max_train_steps:
                done_training = True
                break
        if done_training:
            break

    logger.info("Done!")
    if rank_is_zero():
        wandb.finish()
    cleanup()


if __name__ == "__main__":
    main()
