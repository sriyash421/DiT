#!/usr/bin/env python3
"""Iteratively evaluate adaptive feedback: image -> Gemini feedback -> revised image."""
import argparse
import base64
import io
import json
import os
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from diffusers.models import AutoencoderKL
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion import create_diffusion  # noqa: E402
from models import DiT_models  # noqa: E402

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_GEMINI_MODEL_ID = "google/gemini-3.1-flash-lite"


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def pil_to_tensor(image, image_size):
    image = center_crop_arr(image.convert("RGB"), image_size)
    arr = np.asarray(image).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def tensor_to_pil(x):
    x = x.detach().float().cpu().clamp(-1, 1)
    x = (x + 1) / 2
    x = (x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(x)


def image_to_data_url(image):
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def masked_mean(hidden, mask):
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


@torch.no_grad()
def encode_texts(tokenizer, encoder, texts, max_length, device):
    encoded = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    hidden = encoder(**encoded).last_hidden_state
    mask = encoded["attention_mask"]
    hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)
    pooled = masked_mean(hidden, mask)
    return hidden, mask.bool(), pooled


def load_checkpoint(path, use_ema=True):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and ("ema" in checkpoint or "model" in checkpoint):
        return checkpoint["ema" if use_ema and "ema" in checkpoint else "model"]
    return checkpoint


def select_record(dataset_root, split, template, caption_seed, caption_index):
    records, index = select_records(dataset_root, split, template, caption_seed, caption_index, 1)
    return records[0], index


def select_records(dataset_root, split, template, caption_seed, caption_index, num_captions):
    index_path = Path(dataset_root) / "text_embeddings" / "index.json"
    with index_path.open() as f:
        index = json.load(f)
    records = [r for r in index["records"] if r["split"] == split and r["template"] == template]
    if not records:
        raise RuntimeError(f"No records for split={split}, template={template}")
    rng = random.Random(caption_seed)
    records = rng.sample(records, len(records))
    selected = [records[(caption_index + i) % len(records)] for i in range(num_captions)]
    return selected, index


def feedback_prompt(caption=None):
    caption_block = f"Caption:\n{caption}\n\n" if caption else ""
    return (
        "You are evaluating a text-conditioned diffusion model trained on CLEVR images. "
        "The first image is the ground-truth image. The second image is the current generated image.\n\n"
        f"{caption_block}"
        "Return exactly one short corrective feedback sentence. Do not use bullets. "
        "Do not mention anything already correct. Do not praise the image. "
        "Choose the highest-priority needed edit using this priority order: "
        "1) add missing objects or remove extra objects, "
        "2) fix object shape, "
        "3) fix object color, "
        "4) fix object position or depth ordering, "
        "5) fix material/texture, "
        "6) fix background or camera style. "
        "Use an imperative edit, for example: 'Move the gray sphere behind the yellow sphere.'"
    )


def gemini_feedback(args, gt_image, current_image, caption):
    import requests

    api_key = os.getenv(args.openrouter_api_key_env)
    if not api_key:
        raise ValueError(f"Set {args.openrouter_api_key_env} for OpenRouter access")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    content = [
        {"type": "text", "text": feedback_prompt(caption if args.feedback_uses_caption else None)},
        {"type": "text", "text": "Ground-truth image:"},
        {"type": "image_url", "image_url": {"url": image_to_data_url(gt_image)}},
        {"type": "text", "text": "Current generated image:"},
        {"type": "image_url", "image_url": {"url": image_to_data_url(current_image)}},
    ]
    payload = {
        "model": args.gemini_model,
        "messages": [{"role": "user", "content": content}],
        "temperature": args.feedback_temperature,
        "max_tokens": args.max_feedback_tokens,
    }
    for attempt in range(args.openrouter_retries + 1):
        try:
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip(), result.get("usage", {}) or {}
        except Exception:
            if attempt >= args.openrouter_retries:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


@torch.no_grad()
def sample_image(args, model, vae, diffusion, tokenizer, encoder, caption, feedback, attempt_image, device, seed):
    latent_size = args.image_size // 8
    text_tokens, text_mask, text_pooled = encode_texts(tokenizer, encoder, [caption], args.max_text_len, device)
    model_kwargs = {
        "text_tokens": text_tokens,
        "text_mask": text_mask,
        "text_pooled": text_pooled,
    }
    if feedback is not None and attempt_image is not None:
        feedback_tokens, feedback_mask, feedback_pooled = encode_texts(tokenizer, encoder, [feedback], args.max_text_len, device)
        attempt = pil_to_tensor(attempt_image, args.image_size).unsqueeze(0).to(device)
        attempt_latent = vae.encode(attempt).latent_dist.mode().mul_(vae.config.scaling_factor)
        model_kwargs.update({
            "feedback_tokens": feedback_tokens,
            "feedback_mask": feedback_mask,
            "feedback_pooled": feedback_pooled,
            "attempt_latent": attempt_latent,
        })
        if args.drop_caption_after_feedback:
            model_kwargs["drop_caption"] = torch.ones(1, device=device, dtype=torch.bool)

    generator = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(1, 4, latent_size, latent_size, device=device, generator=generator)
    if args.cfg_scale <= 1:
        forward_fn = model.forward
    else:
        z = torch.cat([z, z], 0)
        model_kwargs = {
            key: value.repeat(2, *([1] * (value.ndim - 1))) if torch.is_tensor(value) else value
            for key, value in model_kwargs.items()
        }
        model_kwargs["cfg_scale"] = args.cfg_scale
        forward_fn = model.forward_with_text_cfg

    sample_loop = diffusion.ddim_sample_loop if args.sampler == "ddim" else diffusion.p_sample_loop
    samples = sample_loop(
        forward_fn,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        device=device,
        **({"eta": args.ddim_eta} if args.sampler == "ddim" else {}),
    )
    if args.cfg_scale > 1:
        samples, _ = samples.chunk(2, dim=0)
    decoded = vae.decode(samples / vae.config.scaling_factor).sample
    return tensor_to_pil(decoded[0])


def plot_trace(args, gt_image, rows, out_path):
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4.2 * n), dpi=args.dpi)
    if n == 1:
        axes = np.expand_dims(axes, axis=0)
    for ax, title in zip(axes[0], ["Ground Truth", "Current Image", "Feedback used for next step"]):
        ax.set_title(title, fontsize=12, fontweight="bold")
    for i, row in enumerate(rows):
        axes[i, 0].imshow(gt_image)
        axes[i, 1].imshow(Image.open(row["image_path"]).convert("RGB"))
        for col in (0, 1):
            axes[i, col].set_xticks([])
            axes[i, col].set_yticks([])
        axes[i, 2].axis("off")
        axes[i, 2].text(0, 1, f"step {row['step']}\n\n{row.get('feedback_for_next', '')}", va="top", ha="left", wrap=True, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def stitch_trace_grids(trace_paths, out_path):
    images = [Image.open(path).convert("RGB") for path in trace_paths]
    target_w = max(image.width for image in images)
    resized = []
    for image in images:
        if image.width != target_w:
            target_h = round(image.height * target_w / image.width)
            image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        resized.append(image)
    gap = 32
    out = Image.new("RGB", (target_w, sum(image.height for image in resized) + gap * (len(resized) - 1)), (255, 255, 255))
    y = 0
    for image in resized:
        out.paste(image, (0, y))
        y += image.height + gap
    out.save(out_path)


def maybe_log_to_wandb(args, trace_paths, combined_path):
    if args.wandb_project is None:
        return
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        id=args.wandb_run_id,
        name=args.wandb_name,
        resume=args.wandb_resume,
        config={
            "eval_ckpt": args.ckpt,
            "eval_split": args.split,
            "eval_template": args.template,
            "eval_caption_index": args.caption_index,
            "eval_num_captions": args.num_captions,
            "eval_feedback_steps": args.steps,
            "eval_cfg_scale": args.cfg_scale,
            "eval_num_sampling_steps": args.num_sampling_steps,
            "eval_sampler": args.sampler,
            "eval_drop_caption_after_feedback": args.drop_caption_after_feedback,
        },
    )
    trace_images = []
    table = wandb.Table(columns=["caption_index", "caption", "trace"])
    for idx, path in enumerate(trace_paths):
        caption = ""
        trace_json = Path(path).parent / "trace.json"
        if trace_json.exists():
            with trace_json.open() as f:
                caption = json.load(f).get("caption", "")
        label = f"caption {idx:02d}"
        if caption:
            label = f"{label}: {caption}"
        image = wandb.Image(str(path), caption=label)
        trace_images.append(image)
        table.add_data(idx, caption, image)

    payload = {
        f"{args.wandb_key}/traces": trace_images,
        f"{args.wandb_key}/table": table,
        f"{args.wandb_key}/combined": wandb.Image(str(combined_path), caption="combined feedback-loop traces"),
    }
    run.log(payload, step=args.wandb_step)
    run.finish()


@torch.no_grad()
def run_trace(args, model, vae, diffusion, tokenizer, encoder, record, out_dir, device):
    out_dir.mkdir(parents=True, exist_ok=True)
    caption = record["caption"]
    gt_path = Path(args.dataset_root) / record["image_path"]
    gt_image = center_crop_arr(Image.open(gt_path).convert("RGB"), args.image_size)

    rows = []
    usage_totals = {}
    current_feedback = None
    current_image = None
    for step in range(args.steps + 1):
        image = sample_image(
            args,
            model,
            vae,
            diffusion,
            tokenizer,
            encoder,
            caption,
            current_feedback,
            current_image,
            device,
            args.seed if args.fixed_seed_per_trace else args.seed + step,
        )
        image_path = out_dir / f"step_{step:02d}.png"
        image.save(image_path)
        feedback_for_next = ""
        usage = {}
        if step < args.steps:
            feedback_for_next, usage = gemini_feedback(args, gt_image, image, caption)
            for key, value in usage.items():
                if isinstance(value, (int, float)):
                    usage_totals[key] = usage_totals.get(key, 0) + value
        rows.append({
            "step": step,
            "image_path": str(image_path),
            "feedback_used": current_feedback,
            "feedback_for_next": feedback_for_next,
            "token_usage": usage,
        })
        print(f"step={step} image={image_path}")
        if feedback_for_next:
            print(f"feedback_for_next={feedback_for_next}")
        current_image = image
        current_feedback = feedback_for_next or None

    trace = {
        "caption": caption,
        "gt_image_path": str(gt_path),
        "record": record,
        "steps": rows,
        "token_usage_totals": usage_totals,
        "args": vars(args),
    }
    with (out_dir / "trace.json").open("w") as f:
        json.dump(trace, f, indent=2)
    trace_grid_path = out_dir / "trace_grid.png"
    plot_trace(args, gt_image, rows, trace_grid_path)
    print(f"Saved trace to {out_dir / 'trace.json'}")
    print(f"Saved grid to {trace_grid_path}")
    print(f"Token usage totals: {usage_totals}")
    return trace_grid_path, usage_totals


@torch.no_grad()
def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    records, index = select_records(
        args.dataset_root,
        args.split,
        args.template,
        args.caption_seed,
        args.caption_index,
        args.num_captions,
    )

    model = DiT_models[args.model](
        input_size=args.image_size // 8,
        num_classes=args.num_classes,
        text_conditioning=True,
        text_embed_dim=index["embedding_dim"],
        max_text_len=index["max_length"],
    ).to(device)
    model.load_state_dict(load_checkpoint(args.ckpt, args.ema), strict=True)
    model.eval()
    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    vae.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps))
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    encoder = T5EncoderModel.from_pretrained(args.encoder).to(device)
    encoder.eval()

    trace_paths = []
    usage_totals = {}
    for idx, record in enumerate(records):
        trace_path, usage = run_trace(args, model, vae, diffusion, tokenizer, encoder, record, out_dir / f"caption_{idx:02d}", device)
        trace_paths.append(trace_path)
        for key, value in usage.items():
            usage_totals[key] = usage_totals.get(key, 0) + value

    final_grid_path = out_dir / "feedback_loop_grid.png"
    stitch_trace_grids(trace_paths, final_grid_path)
    with (out_dir / "usage_totals.json").open("w") as f:
        json.dump(usage_totals, f, indent=2)
    maybe_log_to_wandb(args, trace_paths, final_grid_path)
    print(f"Saved feedback-loop grid to {final_grid_path}")
    print(f"Combined token usage totals: {usage_totals}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model", type=str, default="DiT-S/4", choices=list(DiT_models.keys()))
    parser.add_argument("--vae", type=str, default="stabilityai/sdxl-vae")
    parser.add_argument("--dataset-root", type=str, default="/gpfs/scrubbed/sriyash/clevr_dit_dataset")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--template", type=str, default="chain")
    parser.add_argument("--caption-seed", type=int, default=0)
    parser.add_argument("--caption-index", type=int, default=0)
    parser.add_argument("--num-captions", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-seed-per-trace", action="store_true", help="Use the same sampling seed at every feedback step to isolate conditioning effects.")
    parser.add_argument("--steps", type=int, default=8, help="Number of feedback/regeneration steps after the initial image.")
    parser.add_argument("--drop-caption-after-feedback", action="store_true", help="After step 0, null only the caption while keeping feedback and image context.")
    parser.add_argument("--encoder", type=str, default="google/flan-t5-large")
    parser.add_argument("--max-text-len", type=int, default=128)
    parser.add_argument("--gemini-model", type=str, default=OPENROUTER_GEMINI_MODEL_ID)
    parser.add_argument("--openrouter-api-key-env", type=str, default="OPENROUTER_API_KEY")
    parser.add_argument("--openrouter-retries", type=int, default=2)
    parser.add_argument("--feedback-temperature", type=float, default=0.0)
    parser.add_argument("--max-feedback-tokens", type=int, default=96)
    parser.add_argument("--feedback-uses-caption", action="store_true")
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", type=str, default="results/adaptive_feedback_loop_eval")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-id", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-resume", type=str, default="allow")
    parser.add_argument("--wandb-key", type=str, default="eval/feedback_loop_grid")
    parser.add_argument("--wandb-step", type=int, default=None)
    main(parser.parse_args())
