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
import json
import random
from pathlib import Path

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from diffusers.models import AutoencoderKL
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import save_image

from diffusion import create_diffusion
from models import DiT_models


def load_checkpoint(path, use_ema):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and ("ema" in checkpoint or "model" in checkpoint):
        key = "ema" if use_ema and "ema" in checkpoint else "model"
        state_dict = checkpoint[key]
    else:
        state_dict = checkpoint
    if state_dict is None:
        raise ValueError(f"Checkpoint {path} does not contain requested weights")
    return state_dict


def select_records(dataset_root, split, template, num_captions, seed):
    index_path = Path(dataset_root) / "text_embeddings" / "index.json"
    with index_path.open() as f:
        index = json.load(f)
    records = [
        record for record in index["records"]
        if record["split"] == split and record["template"] == template
    ]
    if len(records) < num_captions:
        raise RuntimeError(f"Only found {len(records)} records for split={split}, template={template}")
    rng = random.Random(seed)
    return rng.sample(records, num_captions), index

def load_text_batch(dataset_root, records, samples_per_caption):
    root = Path(dataset_root) / "text_embeddings"
    shard_cache = {}
    tokens = []
    masks = []
    pooled = []
    captions = []
    image_paths = []
    for record in records:
        shard_name = record["shard"]
        if shard_name not in shard_cache:
            shard_cache[shard_name] = torch.load(root / shard_name, map_location="cpu", weights_only=False)
        shard = shard_cache[shard_name]
        offset = record["offset"]
        token = shard["text_tokens"][offset].float()
        mask = shard["text_mask"][offset].bool()
        pool = shard["text_pooled"][offset].float()
        captions.append(record["caption"])
        image_paths.append(record["image_path"])
        for _ in range(samples_per_caption):
            tokens.append(token)
            masks.append(mask)
            pooled.append(pool)
    return {
        "text_tokens": torch.stack(tokens),
        "text_mask": torch.stack(masks),
        "text_pooled": torch.stack(pooled),
        "captions": captions,
        "image_paths": image_paths,
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


def add_caption_rows_with_ground_truth(image_path, captions, image_paths, dataset_root, samples_per_caption):
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
    for row, rel_image_path in enumerate(image_paths):
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

        gt = Image.open(Path(dataset_root) / rel_image_path).convert("RGB")
        gt = gt.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        gt_x = row_label_w
        out.paste(gt, (gt_x, y))
        draw.rectangle((gt_x, y, gt_x + tile_w - 1, y + tile_h - 1), outline=(0, 0, 0), width=4)
        draw.line((0, y + tile_h - 1, row_label_w + tile_w + grid.width, y + tile_h - 1), fill=(210, 210, 210))
    out.save(image_path)


@torch.no_grad()
def main(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    latent_size = args.image_size // 8
    text_conditioning = args.mode == "text"

    model_kwargs = {
        "input_size": latent_size,
        "num_classes": args.num_classes,
    }
    text_batch = None
    if text_conditioning:
        records, index = select_records(
            args.dataset_root,
            args.split,
            args.template,
            args.num_captions,
            args.caption_seed,
        )
        text_batch = load_text_batch(args.dataset_root, records, args.samples_per_caption)
        model_kwargs.update({
            "text_conditioning": True,
            "text_embed_dim": index["embedding_dim"],
            "max_text_len": index["max_length"],
        })

    model = DiT_models[args.model](**model_kwargs).to(device)
    model.load_state_dict(load_checkpoint(args.ckpt, args.ema), strict=True)
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
            "text_tokens": text_batch["text_tokens"].to(device),
            "text_mask": text_batch["text_mask"].to(device),
            "text_pooled": text_batch["text_pooled"].to(device),
        }
        forward_fn = model.forward
    else:
        z = torch.cat([z, z], dim=0)
        sample_kwargs = {
            "text_tokens": text_batch["text_tokens"].repeat(2, 1, 1).to(device),
            "text_mask": text_batch["text_mask"].repeat(2, 1).to(device),
            "text_pooled": text_batch["text_pooled"].repeat(2, 1).to(device),
            "cfg_scale": args.cfg_scale,
        }
        forward_fn = model.forward_with_text_cfg

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

    if text_batch is not None:
        captions_path = out.with_suffix(".captions.txt")
        with captions_path.open("w") as f:
            for idx, (caption, image_path) in enumerate(zip(text_batch["captions"], text_batch["image_paths"])):
                f.write(f"{idx:02d}\t{image_path}\t{caption}\n")
        if args.draw_captions and args.draw_ground_truth:
            add_caption_rows_with_ground_truth(
                out,
                text_batch["captions"],
                text_batch["image_paths"],
                args.dataset_root,
                args.samples_per_caption,
            )
        elif args.draw_captions:
            add_caption_rows(out, text_batch["captions"], args.samples_per_caption)
        print(f"Saved captions to {captions_path}")
    print(f"Saved samples to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["text", "uncond"], required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), required=True)
    parser.add_argument("--vae", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--dataset-root", type=str, default="/gpfs/scrubbed/sriyash/clevr_dit_dataset")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--template", type=str, default="chain")
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
    main(parser.parse_args())
