#!/usr/bin/env python3
"""Generate GT|prediction grids for train and val CLEVR splits."""
import argparse
import random
import sys
from pathlib import Path

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from diffusers.models import AutoencoderKL
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets_clevr import ClevrContextDataset, pad_contexts
from diffusion import create_diffusion
from models import DiT_models
from scripts.sample_clevr_eval_grid import load_checkpoint_and_config, config_get


DATASET_ROOT = "/gpfs/scrubbed/sriyash/clevr_dit_dataset/data.zarr"
DEFAULT_OUT_DIR = "results/eval_samples/gt_pred_100"


def load_font(size):
    for font_path in (
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def resolve_model_args(config):
    return {
        "model": config_get(config, "model.name", "DiT-L/4"),
        "vae": config_get(config, "model.vae", "stabilityai/sdxl-vae"),
        "image_size": int(config_get(config, "model.image_size", 256)),
        "num_classes": int(config_get(config, "model.num_classes", 1000)),
    }


def select_batch(dataset, split, seed, count=100):
    if len(dataset) < count:
        raise RuntimeError(f"Only found {len(dataset)} rows for split={split}; need {count}.")
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), count)
    contexts = []
    captions = []
    gt_images = []
    image_paths = []
    for idx in indices:
        record = dataset.record_for_index(idx)
        contexts.append(dataset.context_for_row(record["row_idx"]).float())
        captions.append(record["caption"])
        gt_images.append(dataset.image_for_row(record["row_idx"]).convert("RGB"))
        image_paths.append(record["image_path"])
    context_tokens, context_mask = pad_contexts(contexts)
    return {
        "indices": indices,
        "context_tokens": context_tokens,
        "context_mask": context_mask,
        "captions": captions,
        "gt_images": gt_images,
        "image_paths": image_paths,
    }


@torch.no_grad()
def sample_predictions(model, vae, diffusion, batch, image_size, cfg_scale, ddim_eta, device, seed):
    torch.manual_seed(seed)
    latent_size = image_size // 8
    count = int(batch["context_tokens"].shape[0])
    z = torch.randn(count, 4, latent_size, latent_size, device=device)
    model_dtype = next(model.parameters()).dtype
    if cfg_scale > 1:
        z = torch.cat([z, z], dim=0)
        model_kwargs = {
            "context_tokens": batch["context_tokens"].repeat(2, 1, 1).to(device=device, dtype=model_dtype),
            "context_mask": batch["context_mask"].repeat(2, 1).to(device),
            "cfg_scale": cfg_scale,
        }
        forward_fn = model.forward_with_cfg
    else:
        model_kwargs = {
            "context_tokens": batch["context_tokens"].to(device=device, dtype=model_dtype),
            "context_mask": batch["context_mask"].to(device),
        }
        forward_fn = model.forward

    samples = diffusion.ddim_sample_loop(
        forward_fn,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        device=device,
        eta=ddim_eta,
    )
    if cfg_scale > 1:
        samples, _ = samples.chunk(2, dim=0)
    decoded = vae.decode(samples / vae.config.scaling_factor).sample
    decoded = decoded.detach().float().cpu().clamp(-1, 1)
    decoded = ((decoded + 1) / 2 * 255).round().byte()
    return [Image.fromarray(image.permute(1, 2, 0).numpy(), mode="RGB") for image in decoded]


def save_gt_pred_grid(path, gt_images, pred_images, rows, cols, gap):
    tile = 128
    pair_w = tile * 2
    label_h = 24
    cell_h = tile + label_h
    out_w = cols * pair_w + max(cols - 1, 0) * gap
    out_h = rows * cell_h + max(rows - 1, 0) * gap
    out = Image.new("RGB", (out_w, out_h), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    font = load_font(14)
    for idx, (gt, pred) in enumerate(zip(gt_images, pred_images)):
        row = idx // cols
        col = idx % cols
        x = col * (pair_w + gap)
        y = row * (cell_h + gap)
        draw.text((x + 8, y + 4), "GT", fill=(20, 20, 20), font=font)
        draw.text((x + tile + 8, y + 4), "Pred", fill=(20, 20, 20), font=font)
        out.paste(gt.resize((tile, tile), Image.Resampling.LANCZOS), (x, y + label_h))
        out.paste(pred.resize((tile, tile), Image.Resampling.LANCZOS), (x + tile, y + label_h))
        draw.rectangle((x, y + label_h, x + tile - 1, y + label_h + tile - 1), outline=(0, 0, 0))
        draw.rectangle((x + tile, y + label_h, x + pair_w - 1, y + label_h + tile - 1), outline=(0, 0, 0))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def save_sidecar(path, batch):
    path = Path(path)
    with path.open("w") as f:
        for idx, image_path, caption in zip(batch["indices"], batch["image_paths"], batch["captions"]):
            f.write(f"{idx}\t{image_path}\t{caption}\n")


@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    state_dict, config = load_checkpoint_and_config(args.ckpt, use_ema=True)
    model_args = resolve_model_args(config)
    latent_size = model_args["image_size"] // 8

    train_dataset = ClevrContextDataset(DATASET_ROOT, transform=None, split="train")
    context_dim = train_dataset.context_dim
    model = DiT_models[model_args["model"]](
        input_size=latent_size,
        num_classes=model_args["num_classes"],
        text_conditioning=True,
        context_dim=context_dim,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    vae = AutoencoderKL.from_pretrained(model_args["vae"]).to(device)
    vae.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps))

    out_dir = Path(args.out_dir)
    rows, cols = args.grid
    count = rows * cols
    for split, split_seed in (("train", args.seed), ("val", args.seed + 1)):
        dataset = train_dataset if split == "train" else ClevrContextDataset(DATASET_ROOT, transform=None, split=split)
        batch = select_batch(dataset, split, split_seed, count=count)
        predictions = sample_predictions(
            model,
            vae,
            diffusion,
            batch,
            model_args["image_size"],
            args.cfg_scale,
            args.ddim_eta,
            device,
            seed=args.sample_seed + (0 if split == "train" else 1),
        )
        out_path = out_dir / f"{Path(args.ckpt).stem}_{split}_gt_pred_{rows}x{cols}.png"
        save_gt_pred_grid(out_path, batch["gt_images"], predictions, rows, cols, args.gap)
        save_sidecar(out_path.with_suffix(".captions.txt"), batch)
        print(f"Saved {split} grid to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=1234)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--grid", type=int, nargs=2, metavar=("ROWS", "COLS"), default=(10, 10))
    parser.add_argument("--gap", type=int, default=12)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
