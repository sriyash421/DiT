"""
Generate CLEVR DDIM x0_hat evolution GIFs for text-conditioned checkpoints.

For each selected caption, this script generates two samples. Each GIF has two
rows, one per sample. Each row shows:
  caption text | final generated image | evolving x0_hat prediction
"""
import argparse
import json
import random
import textwrap
from pathlib import Path

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from diffusers.models import AutoencoderKL
from PIL import Image, ImageDraw

from diffusion import create_diffusion
from models import DiT_models


def load_checkpoint(path, use_ema):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and ("ema" in checkpoint or "model" in checkpoint):
        key = "ema" if use_ema and "ema" in checkpoint else "model"
        return checkpoint[key]
    return checkpoint


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


def load_one_record(dataset_root, record, repeats):
    root = Path(dataset_root) / "text_embeddings"
    shard = torch.load(root / record["shard"], map_location="cpu", weights_only=False)
    offset = record["offset"]
    return {
        "text_tokens": shard["text_tokens"][offset].float().repeat(repeats, 1, 1),
        "text_mask": shard["text_mask"][offset].bool().repeat(repeats, 1),
        "text_pooled": shard["text_pooled"][offset].float().repeat(repeats, 1),
        "caption": record["caption"],
        "image_path": record["image_path"],
    }


@torch.no_grad()
def decode_latents(vae, latents, scaling_factor):
    return vae.decode(latents / scaling_factor).sample.detach().cpu()


def tensor_to_pil(image_tensor, size):
    image = image_tensor.clamp(-1, 1)
    image = ((image + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    image = image.permute(1, 2, 0).numpy()
    return Image.fromarray(image).resize((size, size), Image.Resampling.LANCZOS)


def draw_caption_panel(draw, xy, width, height, caption, image_path, row_idx):
    x, y = xy
    draw.rectangle((x, y, x + width, y + height), fill=(255, 255, 255))
    draw.text((x + 10, y + 10), f"sample {row_idx}", fill=(20, 20, 20))
    draw.text((x + 10, y + 30), str(image_path), fill=(80, 80, 80))
    wrapped = textwrap.wrap(caption, width=52)
    yy = y + 58
    for line in wrapped[:9]:
        draw.text((x + 10, yy), line, fill=(20, 20, 20))
        yy += 18
    if len(wrapped) > 9:
        draw.text((x + 10, yy), "...", fill=(20, 20, 20))


def make_gif(frames_x0, final_images, caption, image_path, out_path, frame_duration, panel_size):
    caption_w = 420
    gap = 12
    header_h = 28
    row_h = panel_size + header_h
    width = caption_w + gap + panel_size + gap + panel_size
    height = row_h * 2 + gap
    frames = []
    for step_idx in range(frames_x0.shape[0]):
        canvas = Image.new("RGB", (width, height), (240, 240, 240))
        draw = ImageDraw.Draw(canvas)
        for row in range(2):
            y = row * (row_h + gap)
            draw_caption_panel(draw, (0, y), caption_w, row_h, caption, image_path, row)
            draw.text((caption_w + gap, y + 6), "final image", fill=(20, 20, 20))
            draw.text((caption_w + gap + panel_size + gap, y + 6), f"x0_hat step {step_idx + 1}/{frames_x0.shape[0]}", fill=(20, 20, 20))
            final_pil = tensor_to_pil(final_images[row], panel_size)
            x0_pil = tensor_to_pil(frames_x0[step_idx, row], panel_size)
            canvas.paste(final_pil, (caption_w + gap, y + header_h))
            canvas.paste(x0_pil, (caption_w + gap + panel_size + gap, y + header_h))
        frames.append(canvas)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0,
    )


@torch.no_grad()
def sample_caption_gif(args, model, vae, diffusion, record, index, out_dir, caption_idx, device):
    text = load_one_record(args.dataset_root, record, repeats=2)
    latent_size = args.image_size // 8
    z = torch.randn(2, 4, latent_size, latent_size, device=device)
    if args.cfg_scale <= 1:
        model_kwargs = {
            "text_tokens": text["text_tokens"].to(device),
            "text_mask": text["text_mask"].to(device),
            "text_pooled": text["text_pooled"].to(device),
        }
        forward_fn = model.forward
    else:
        z = torch.cat([z, z], dim=0)
        model_kwargs = {
            "text_tokens": text["text_tokens"].repeat(2, 1, 1).to(device),
            "text_mask": text["text_mask"].repeat(2, 1).to(device),
            "text_pooled": text["text_pooled"].repeat(2, 1).to(device),
            "cfg_scale": args.cfg_scale,
        }
        forward_fn = model.forward_with_text_cfg

    x0_frames = []
    sample = None
    for out in diffusion.ddim_sample_loop_progressive(
        forward_fn,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        device=device,
        eta=args.ddim_eta,
    ):
        pred_xstart = out["pred_xstart"]
        if args.cfg_scale > 1:
            pred_xstart, _ = pred_xstart.chunk(2, dim=0)
        x0_images = decode_latents(vae, pred_xstart, vae.config.scaling_factor)
        x0_frames.append(x0_images)
        sample = out["sample"]

    if args.cfg_scale > 1:
        sample, _ = sample.chunk(2, dim=0)
    final_images = decode_latents(vae, sample, vae.config.scaling_factor)
    frames = torch.stack(x0_frames, dim=0)
    out_path = out_dir / f"{args.split}_caption_{caption_idx:02d}_x0_evolution.gif"
    make_gif(
        frames,
        final_images,
        text["caption"],
        text["image_path"],
        out_path,
        args.frame_duration,
        args.panel_size,
    )
    return out_path, text["caption"], text["image_path"]


def main(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    records, index = select_records(args.dataset_root, args.split, args.template, args.num_captions, args.caption_seed)

    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    captions_path = out_dir / f"{args.split}_x0_gif_captions.txt"
    with captions_path.open("w") as f:
        for idx, record in enumerate(records):
            gif_path, caption, image_path = sample_caption_gif(args, model, vae, diffusion, record, index, out_dir, idx, device)
            f.write(f"{idx:02d}\t{gif_path}\t{image_path}\t{caption}\n")
            print(f"Saved {gif_path}")
    print(f"Saved captions to {captions_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), required=True)
    parser.add_argument("--vae", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--dataset-root", type=str, default="/gpfs/scrubbed/sriyash/clevr_dit_dataset")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--template", type=str, default="chain")
    parser.add_argument("--num-captions", type=int, default=5)
    parser.add_argument("--caption-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--panel-size", type=int, default=256)
    parser.add_argument("--frame-duration", type=int, default=140)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
