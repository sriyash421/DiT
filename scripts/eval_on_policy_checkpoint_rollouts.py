#!/usr/bin/env python3
"""Evaluate on-policy checkpoints with history-conditioned feedback rollouts."""
import argparse
import json
import random
import sys
import textwrap
import time
from pathlib import Path

print(f"[{time.strftime('%H:%M:%S')}] importing matplotlib/torch/diffusers...", flush=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from diffusers.models import AutoencoderKL
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm
print(f"[{time.strftime('%H:%M:%S')}] imports complete", flush=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clevr_transforms import build_clevr_transform  # noqa: E402
from datasets_clevr import ClevrContextDataset, context_collate  # noqa: E402
from diffusion import create_diffusion  # noqa: E402
from feedback_verifiers import build_feedback_verifier  # noqa: E402
from models import DiT_models  # noqa: E402
from on_policy import PolicySampler, QwenContextEncoder  # noqa: E402
from train_text import load_checkpoint  # noqa: E402


def select_unique_caption_items(dataset, count, seed=0, start_index=0):
    rng = random.Random(seed)
    order = list(range(len(dataset)))
    rng.shuffle(order)
    order = order[start_index:] + order[:start_index]
    seen = set()
    items = []
    indices = []
    for idx in order:
        record = dataset.record_for_index(idx)
        caption = record.get("caption", "")
        if caption in seen:
            continue
        seen.add(caption)
        items.append(dataset[idx])
        indices.append(idx)
        if len(items) >= count:
            break
    if not items:
        raise RuntimeError("No captions selected for eval.")
    return indices, items


def load_model(checkpoint, cfg, context_dim, device):
    model = DiT_models[cfg.model.name](
        input_size=cfg.model.image_size // 8,
        num_classes=cfg.model.num_classes,
        text_conditioning=True,
        context_dim=context_dim,
        class_dropout_prob=cfg.model.context_dropout_prob,
    ).to(device)
    missing, unexpected = model.load_state_dict(load_checkpoint(str(checkpoint)), strict=False)
    model.eval()
    return model, missing, unexpected


def short_text(text, width=24, max_lines=3):
    if not text:
        return ""
    lines = textwrap.wrap(str(text), width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".") + "..."
    return "\n".join(lines)


def save_grid(path, captions, gt_images, rollout_images, rollout_feedback):
    rows = len(captions)
    steps = len(rollout_images[0]) if rollout_images else 0
    fig, axes = plt.subplots(
        rows,
        steps + 1,
        figsize=(2.2 * (steps + 1), 2.25 * rows),
        dpi=160,
        squeeze=False,
    )
    for row_idx, caption in enumerate(captions):
        for col_idx in range(steps + 1):
            ax = axes[row_idx, col_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            if col_idx == 0:
                ax.imshow(gt_images[row_idx])
                ax.set_title("GT" if row_idx == 0 else "", fontsize=9, fontweight="bold")
                ax.set_ylabel(short_text(caption, width=30, max_lines=4), fontsize=7, rotation=0, ha="right", va="center", labelpad=54)
            else:
                step_idx = col_idx - 1
                ax.imshow(rollout_images[row_idx][step_idx])
                title = short_text(rollout_feedback[row_idx][step_idx], width=24, max_lines=3)
                ax.set_title(title, fontsize=7)
    fig.tight_layout(pad=0.6, w_pad=0.8, h_pad=1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def eval_checkpoint(args, checkpoint, cfg, dataset, indices, items, vae, sampler, context_encoder, verifier, device):
    batch = context_collate(items)
    captions = batch["caption"]
    gt_images = [dataset.image_for_row(dataset.indices[idx]).convert("RGB") for idx in indices]
    model, missing, unexpected = load_model(checkpoint, cfg, dataset.context_dim, device)

    histories = [[] for _ in captions]
    history_images = [[] for _ in captions]
    rollout_images = [[] for _ in captions]
    rollout_feedback = [[] for _ in captions]
    current_tokens = batch["context_tokens"]
    current_mask = batch["context_mask"]
    metadata = [None] * len(captions)

    for step_idx in tqdm(range(args.steps), desc=f"{checkpoint.stem}", leave=False):
        _, attempt_images = sampler.sample(
            model,
            vae,
            current_tokens,
            current_mask,
            device,
            seed=args.seed + step_idx,
        )
        results = verifier.verify_history_batch(
            captions,
            metadata,
            gt_images,
            attempt_images,
            [list(history) for history in histories],
        )
        for idx, (image, result) in enumerate(zip(attempt_images, results)):
            rollout_images[idx].append(image.convert("RGB"))
            feedback = result.feedback if result.ok else f"VLM failed: {result.error}"
            rollout_feedback[idx].append(feedback)
            if result.ok:
                histories[idx].append(feedback)
                history_images[idx].append(image)
        if step_idx + 1 < args.steps:
            current_tokens, current_mask = context_encoder.encode_history(
                captions,
                histories,
                history_images,
            )

    out_png = Path(args.out_dir) / f"{checkpoint.stem}_rollouts.png"
    out_json = Path(args.out_dir) / f"{checkpoint.stem}_rollouts.json"
    save_grid(out_png, captions, gt_images, rollout_images, rollout_feedback)
    payload = {
        "checkpoint": str(checkpoint),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "indices": [int(idx) for idx in indices],
        "captions": list(captions),
        "feedback": rollout_feedback,
        "plot": str(out_png),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(payload, f, indent=2)
    return out_png


def checkpoint_paths(args):
    if args.checkpoints:
        return [Path(path) for path in args.checkpoints]
    ckpt_dir = Path(args.run_dir) / "checkpoints"
    checkpoints = sorted(ckpt_dir.glob(args.checkpoint_glob))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched {ckpt_dir / args.checkpoint_glob}")
    return checkpoints


def main(args):
    run_dir = Path(args.run_dir)
    print(f"[{time.strftime('%H:%M:%S')}] loading config from {args.config or (run_dir / 'config.yaml')}", flush=True)
    cfg = OmegaConf.load(args.config or (run_dir / "config.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[{time.strftime('%H:%M:%S')}] using device={device}", flush=True)
    transform = build_clevr_transform(cfg.model.image_size)
    dataset = ClevrContextDataset(
        cfg.data.dataset_path,
        transform=transform,
        split=args.split,
        use_disk=True,
        load_meta=True,
        load_images=True,
        load_context=True,
        max_dataset_size=args.max_dataset_size,
    )
    indices, items = select_unique_caption_items(dataset, args.num_captions, args.seed, args.caption_index)
    print(f"[{time.strftime('%H:%M:%S')}] selected {len(items)} captions from split={args.split}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading VAE {cfg.model.vae}", flush=True)
    vae = AutoencoderKL.from_pretrained(cfg.model.vae).to(device)
    vae.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps or cfg.sampler.num_sampling_steps))
    sampler = PolicySampler(
        diffusion,
        latent_size=cfg.model.image_size // 8,
        vae_scaling_factor=vae.config.scaling_factor,
        cfg_scale=args.cfg_scale if args.cfg_scale is not None else cfg.sampler.cfg_scale,
        sampler=cfg.sampler.type,
        ddim_eta=cfg.sampler.ddim_eta,
    )
    print(f"[{time.strftime('%H:%M:%S')}] loading context encoder {cfg.context_encoder.model}", flush=True)
    context_encoder = QwenContextEncoder(
        cfg.context_encoder.model,
        device=str(device),
        vlm_dtype=cfg.context_encoder.dtype,
        device_map=cfg.context_encoder.device_map,
        max_length=args.max_context_len or cfg.context_encoder.max_length,
        out_dtype=torch.float16,
        include_metadata=False,
    )
    print(f"[{time.strftime('%H:%M:%S')}] connecting verifier at {args.vllm_base_url or cfg.verifier.api_url}", flush=True)
    verifier = build_feedback_verifier(
        backend="qwen-vllm",
        model=args.vllm_model or cfg.verifier.model,
        api_url=args.vllm_base_url or cfg.verifier.api_url,
        api_key=args.vllm_api_key,
        temperature=args.feedback_temperature,
        max_tokens=args.max_feedback_tokens,
        retries=args.retries,
        timeout=args.timeout,
        workers=args.vlm_workers,
        enable_thinking=args.enable_thinking,
        include_caption=True,
        include_metadata=False,
        image_size=cfg.model.image_size,
    )

    outputs = []
    checkpoints = checkpoint_paths(args)
    print(f"[{time.strftime('%H:%M:%S')}] evaluating {len(checkpoints)} checkpoints", flush=True)
    for checkpoint in tqdm(checkpoints, desc="checkpoints"):
        outputs.append(eval_checkpoint(args, checkpoint, cfg, dataset, indices, items, vae, sampler, context_encoder, verifier, device))
    print("Saved plots:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="/gscratch/scrubbed/sriyash/clevr_onpolicy_debug/runs/pilot_l4_40k_100x128_len4_lr1e-5")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoints", nargs="*", default=None)
    parser.add_argument("--checkpoint-glob", default="*-ema.pt")
    parser.add_argument("--out-dir", default="results/on_policy_debug_plots")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-captions", type=int, default=16)
    parser.add_argument("--caption-index", type=int, default=0)
    parser.add_argument("--max-dataset-size", type=int, default=50)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-sampling-steps", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--max-context-len", type=int, default=None)
    parser.add_argument("--vllm-base-url", default=None)
    parser.add_argument("--vllm-api-key", default="EMPTY")
    parser.add_argument("--vllm-model", default=None)
    parser.add_argument("--vlm-workers", type=int, default=16)
    parser.add_argument("--feedback-temperature", type=float, default=0.0)
    parser.add_argument("--max-feedback-tokens", type=int, default=64)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
