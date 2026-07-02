"""
Generate fixed CLEVR evaluation grids from trained DiT checkpoints.

Text mode:
  - selects N captions from a dataset split
  - generates M samples per caption
  - saves a rows=N, cols=M image grid and a captions.txt sidecar

Unconditional mode:
  - generates rows * cols random samples
"""
import argparse
import random
import sys
from pathlib import Path

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from diffusers.models import AutoencoderKL
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import save_image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion import create_diffusion
from datasets_clevr import ClevrContextDataset, pad_contexts
from models import DiT_models


def sibling_full_checkpoint(path):
    path = Path(path)
    if path.stem.endswith("-ema"):
        candidate = path.with_name(path.stem[:-4] + path.suffix)
        if candidate.exists():
            return candidate
    return None


def load_checkpoint_and_config(path, use_ema):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = None
    if isinstance(checkpoint, dict) and ("ema" in checkpoint or "model" in checkpoint):
        key = "ema" if use_ema and "ema" in checkpoint else "model"
        state_dict = checkpoint[key]
        config = checkpoint.get("config")
    else:
        state_dict = checkpoint
    if config is None:
        sibling = sibling_full_checkpoint(path)
        if sibling is not None:
            sibling_checkpoint = torch.load(sibling, map_location="cpu", weights_only=False)
            if isinstance(sibling_checkpoint, dict):
                config = sibling_checkpoint.get("config")
    if state_dict is None:
        raise ValueError(f"Checkpoint {path} does not contain requested weights")
    return state_dict, config


def config_get(config, path, default=None):
    value = config
    for key in path.split("."):
        if value is None:
            return default
        if isinstance(value, dict):
            value = value.get(key)
        else:
            value = getattr(value, key, None)
    return default if value is None else value


def default_dataset_root(config):
    dataset_config = config_get(config, "data.dataset_config", None)
    if dataset_config:
        first = dataset_config[0]
        if isinstance(first, dict):
            return first.get("dataset_path")
        return getattr(first, "dataset_path", None)
    return None


def resolve_model_args(args, config):
    args.model = args.model or config_get(config, "model.name", "DiT-S/4")
    args.vae = args.vae or config_get(config, "model.vae", "stabilityai/sdxl-vae")
    args.image_size = args.image_size or int(config_get(config, "model.image_size", 256))
    args.num_classes = args.num_classes or int(config_get(config, "model.num_classes", 1000))
    args.dataset_root = args.dataset_root or default_dataset_root(config)
    if args.mode == "text" and args.dataset_root is None:
        raise ValueError("Provide --dataset-root or use a checkpoint with config.data.dataset_config.")
    return args


def select_records(dataset_root, split, num_captions, seed):
    dataset = ClevrContextDataset(dataset_root, transform=None, split=split)
    if len(dataset) < num_captions:
        raise RuntimeError(f"Only found {len(dataset)} records for split={split}")
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), num_captions)
    return [(dataset, idx, dataset.record_for_index(idx)) for idx in indices], {"context_dim": dataset.context_dim}

def load_context_batch(dataset_root, records, samples_per_caption):
    contexts = []
    captions = []
    image_paths = []
    gt_images = []
    for dataset, _, record in records:
        token = dataset.context_for_row(record["row_idx"]).float()
        captions.append(record["caption"])
        image_paths.append(record["image_path"])
        gt_images.append(dataset.image_for_row(record["row_idx"]))
        for _ in range(samples_per_caption):
            contexts.append(token)
    context_tokens, context_mask = pad_contexts(contexts)
    return {
        "context_tokens": context_tokens,
        "context_mask": context_mask,
        "captions": captions,
        "image_paths": image_paths,
        "gt_images": gt_images,
    }


def wrap_text(draw, text, max_width):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def add_caption_rows(image_path, captions, samples_per_caption):
    image_path = Path(image_path)
    grid = Image.open(image_path).convert("RGB")
    tile_w = grid.width // samples_per_caption
    tile_h = grid.height // len(captions)
    caption_w = max(420, tile_w * 2)
    pad = 12
    out = Image.new("RGB", (caption_w + grid.width, grid.height), (245, 245, 245))
    out.paste(grid, (caption_w, 0))
    draw = ImageDraw.Draw(out)
    for row, caption in enumerate(captions):
        y = row * tile_h
        draw.rectangle((0, y, caption_w, y + tile_h), fill=(255, 255, 255))
        draw.text((pad, y + pad), f"{row:02d}", fill=(20, 20, 20))
        lines = wrap_text(draw, caption, caption_w - 2 * pad)
        text_y = y + pad + 22
        line_height = 17
        max_lines = max(1, (tile_h - text_y + y - pad) // line_height)
        for line in lines[:max_lines]:
            draw.text((pad, text_y), line, fill=(20, 20, 20))
            text_y += line_height
        if len(lines) > max_lines:
            draw.text((pad, text_y - line_height), "...", fill=(20, 20, 20))
        draw.line((0, y + tile_h - 1, caption_w + grid.width, y + tile_h - 1), fill=(210, 210, 210))
    out.save(image_path)


def load_font(size):
    for font_path in (
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def add_caption_rows_with_ground_truth(image_path, captions, gt_images, dataset_root, samples_per_caption):
    image_path = Path(image_path)
    grid = Image.open(image_path).convert("RGB")
    tile_w = grid.width // samples_per_caption
    tile_h = grid.height // len(captions)
    row_label_w = 70
    label_h = 44
    heading_font = load_font(26)
    row_font = load_font(22)
    out = Image.new("RGB", (row_label_w + tile_w + grid.width, grid.height + label_h), (245, 245, 245))
    draw = ImageDraw.Draw(out)
    draw.text((row_label_w + 10, 8), "Ground Truth", fill=(20, 20, 20), font=heading_font)
    draw.text((row_label_w + tile_w + 10, 8), "Generated Samples", fill=(20, 20, 20), font=heading_font)
    out.paste(grid, (row_label_w + tile_w, label_h))
    for row, gt_image in enumerate(gt_images):
        y = label_h + row * tile_h
        draw.rectangle((0, y, row_label_w, y + tile_h), fill=(255, 255, 255))
        label = f"{row:02d}"
        bbox = draw.textbbox((0, 0), label, font=row_font)
        draw.text(
            ((row_label_w - (bbox[2] - bbox[0])) // 2, y + (tile_h - (bbox[3] - bbox[1])) // 2),
            label,
            fill=(20, 20, 20),
            font=row_font,
        )

        gt = gt_image.convert("RGB").resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        gt_x = row_label_w
        out.paste(gt, (gt_x, y))
        draw.rectangle((gt_x, y, gt_x + tile_w - 1, y + tile_h - 1), outline=(0, 0, 0), width=4)
        draw.line((0, y + tile_h - 1, row_label_w + tile_w + grid.width, y + tile_h - 1), fill=(210, 210, 210))
    out.save(image_path)


def log_to_wandb_if_configured(args, image_path):
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
            "eval_cfg_scale": args.cfg_scale,
            "eval_num_captions": args.num_captions,
            "eval_samples_per_caption": args.samples_per_caption,
            "eval_num_sampling_steps": args.num_sampling_steps,
            "eval_sampler": args.sampler,
            "model": args.model,
            "vae": args.vae,
            "image_size": args.image_size,
            "num_classes": args.num_classes,
        },
    )
    key = args.wandb_key
    run.log({key: wandb.Image(str(image_path))}, step=args.wandb_step)
    run.finish()


@torch.no_grad()
def main(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    state_dict, train_config = load_checkpoint_and_config(args.ckpt, args.ema)
    args = resolve_model_args(args, train_config)
    latent_size = args.image_size // 8
    text_conditioning = args.mode == "text"

    model_kwargs = {
        "input_size": latent_size,
        "num_classes": args.num_classes,
    }
    context_batch = None
    if text_conditioning:
        records, index = select_records(
            args.dataset_root,
            args.split,
            args.num_captions,
            args.caption_seed,
        )
        context_batch = load_context_batch(args.dataset_root, records, args.samples_per_caption)
        model_kwargs.update({
            "text_conditioning": True,
            "context_dim": index["context_dim"],
        })

    model = DiT_models[args.model](**model_kwargs).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    vae.eval()
    vae_scaling_factor = vae.config.scaling_factor
    diffusion = create_diffusion(str(args.num_sampling_steps))

    total = args.num_captions * args.samples_per_caption
    z = torch.randn(total, 4, latent_size, latent_size, device=device)
    if args.mode == "uncond":
        sample_kwargs = {}
        forward_fn = model.forward
    elif args.cfg_scale <= 1:
        sample_kwargs = {
            "context_tokens": context_batch["context_tokens"].to(device),
            "context_mask": context_batch["context_mask"].to(device),
        }
        forward_fn = model.forward
    else:
        z = torch.cat([z, z], dim=0)
        sample_kwargs = {
            "context_tokens": context_batch["context_tokens"].repeat(2, 1, 1).to(device),
            "context_mask": context_batch["context_mask"].repeat(2, 1).to(device),
            "cfg_scale": args.cfg_scale,
        }
        forward_fn = model.forward_with_cfg

    sample_loop = diffusion.ddim_sample_loop if args.sampler == "ddim" else diffusion.p_sample_loop
    samples = sample_loop(
        forward_fn,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs=sample_kwargs,
        progress=True,
        device=device,
        **({"eta": args.ddim_eta} if args.sampler == "ddim" else {}),
    )
    if args.mode == "text" and args.cfg_scale > 1:
        samples, _ = samples.chunk(2, dim=0)
    decoded = vae.decode(samples / vae_scaling_factor).sample

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(
        decoded,
        out,
        nrow=args.samples_per_caption,
        normalize=True,
        value_range=(-1, 1),
    )

    if context_batch is not None:
        captions_path = out.with_suffix(".captions.txt")
        with captions_path.open("w") as f:
            for idx, (caption, image_path) in enumerate(zip(context_batch["captions"], context_batch["image_paths"])):
                f.write(f"{idx:02d}\t{image_path}\t{caption}\n")
        if args.draw_captions and args.draw_ground_truth:
            add_caption_rows_with_ground_truth(
                out,
                context_batch["captions"],
                context_batch["gt_images"],
                args.dataset_root,
                args.samples_per_caption,
            )
        elif args.draw_captions:
            add_caption_rows(out, context_batch["captions"], args.samples_per_caption)
        print(f"Saved captions to {captions_path}")
    log_to_wandb_if_configured(args, out)
    print(f"Saved samples to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["text", "uncond"], required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default=None)
    parser.add_argument("--vae", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--num-captions", type=int, default=5)
    parser.add_argument("--samples-per-caption", type=int, default=4)
    parser.add_argument("--caption-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--draw-captions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--draw-ground-truth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-id", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-resume", type=str, default="allow")
    parser.add_argument("--wandb-key", type=str, default="eval/grid")
    parser.add_argument("--wandb-step", type=int, default=None)
    main(parser.parse_args())
