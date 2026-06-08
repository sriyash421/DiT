import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from diffusers.models import AutoencoderKL
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageFont

from diffusion import create_diffusion
from models import DiT_models


def load_font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def tensor_to_pil(x):
    x = x.detach().float().cpu().clamp(-1, 1)
    x = (x + 1) / 2
    x = (x[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(x)


def resize_square(image, size):
    image = image.convert("RGB")
    w, h = image.size
    crop = min(w, h)
    left = (w - crop) // 2
    top = (h - crop) // 2
    image = image.crop((left, top, left + crop, top + crop))
    return image.resize((size, size), Image.Resampling.LANCZOS)


def flatten_index_records(index):
    records = []

    def visit(obj):
        if isinstance(obj, dict):
            if ("embedding_path" in obj or ("shard" in obj and "offset" in obj)) and ("caption" in obj or "image_path" in obj):
                records.append(obj)
                return
            for value in obj.values():
                visit(value)
        elif isinstance(obj, list):
            for value in obj:
                visit(value)

    visit(index)
    return records


def record_matches(record, split, template):
    image_path = str(record.get("image_path", ""))
    embedding_path = str(record.get("embedding_path", ""))
    shard_path = str(record.get("shard", ""))
    path_parts = set(Path(image_path).parts) | set(Path(embedding_path).parts) | set(Path(shard_path).parts)

    record_split = record.get("split")
    split_ok = record_split == split or (record_split is None and split in path_parts)

    record_template = record.get("template")
    template_ok = template is None or record_template == template or record_template is None

    return split_ok and template_ok


def select_record(dataset_root, split, template, caption_index):
    index_path = Path(dataset_root) / "text_embeddings" / "index.json"
    with index_path.open("r") as f:
        index = json.load(f)
    records = [
        r for r in flatten_index_records(index)
        if record_matches(r, split, template)
    ]
    if not records:
        raise ValueError(f"No records found for split={split!r}, template={template!r}")
    return records[caption_index % len(records)]


def load_embedding_record(dataset_root, record):
    emb_root = Path(dataset_root) / "text_embeddings"
    if "embedding_path" in record:
        return torch.load(emb_root / record["embedding_path"], map_location="cpu")

    shard = torch.load(emb_root / record["shard"], map_location="cpu")
    offset = int(record["offset"])
    return {
        "text_tokens": shard["text_tokens"][offset],
        "text_mask": shard["text_mask"][offset],
        "text_pooled": shard["text_pooled"][offset],
    }


def load_text_conditioning(dataset_root, record, device, dtype=torch.float32):
    data = load_embedding_record(dataset_root, record)
    text_tokens = data["text_tokens"].unsqueeze(0).to(device=device, dtype=dtype)
    text_mask = data["text_mask"].unsqueeze(0).to(device=device)
    text_pooled = data["text_pooled"].unsqueeze(0).to(device=device, dtype=dtype)
    return text_tokens, text_mask, text_pooled


def infer_text_shapes(dataset_root):
    index_path = Path(dataset_root) / "text_embeddings" / "index.json"
    with index_path.open("r") as f:
        records = flatten_index_records(json.load(f))
    if not records:
        raise ValueError(f"No embedding records found in {index_path}")
    first = records[0]
    data = load_embedding_record(dataset_root, first)
    return data["text_tokens"].shape[0], data["text_tokens"].shape[1]


def load_state_dict(path, use_ema=True):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        if use_ema and "ema" in ckpt:
            return ckpt["ema"]
        if "model" in ckpt:
            return ckpt["model"]
    return ckpt


@torch.no_grad()
def decode_latent(vae, latent, scaling_factor):
    return tensor_to_pil(vae.decode(latent / scaling_factor).sample)


def cfg_model_fn(model, text_tokens, text_mask, text_pooled, cfg_scale):
    def fn(x, t):
        cond = model(
            x,
            t,
            text_tokens=text_tokens,
            text_mask=text_mask,
            text_pooled=text_pooled,
        )
        drop = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        uncond = model(
            x,
            t,
            text_tokens=text_tokens,
            text_mask=text_mask,
            text_pooled=text_pooled,
            force_drop_text=drop,
        )
        # Match the repository CFG path: guide only the RGB/epsilon channels.
        cond_eps, cond_rest = cond[:, :3], cond[:, 3:]
        uncond_eps = uncond[:, :3]
        guided_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        return torch.cat([guided_eps, cond_rest], dim=1)

    return fn


@torch.no_grad()
def run_schedule(args, model, vae, diffusion, text_cond, start_noise, schedule):
    text_tokens, text_mask, text_pooled = text_cond
    img = start_noise.clone()
    x0_hats = []
    cfg_values = []

    for step_idx, i in enumerate(reversed(range(diffusion.num_timesteps))):
        progress = step_idx / max(1, diffusion.num_timesteps - 1)
        cfg_scale = schedule(progress)
        cfg_values.append(float(cfg_scale))
        t = torch.tensor([i], device=img.device)
        out = diffusion.ddim_sample(
            cfg_model_fn(model, text_tokens, text_mask, text_pooled, cfg_scale),
            img,
            t,
            clip_denoised=False,
            model_kwargs={},
            eta=0.0,
        )
        img = out["sample"]
        x0_hats.append(out["pred_xstart"].detach().clone())

    final_pil = decode_latent(vae, img, args.vae_scaling_factor)
    x0_pils = [decode_latent(vae, x0, args.vae_scaling_factor) for x0 in x0_hats]
    return final_pil, x0_pils, cfg_values


def pil_to_array(image):
    return np.asarray(image.convert("RGB"))


def make_frame(gt, rows, frame_idx, args, caption):
    fig, axes = plt.subplots(4, 3, figsize=(12, 14), dpi=args.dpi)
    fig.patch.set_facecolor("white")

    col_titles = ["ground truth", "final image", "current x0_hat"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=18, fontweight="bold", pad=12)

    for row_idx, row in enumerate(rows):
        panels = [gt, row["final"], row["x0_hats"][frame_idx]]
        for col_idx, panel in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(pil_to_array(panel))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2)
                spine.set_color("black")

        axes[row_idx, 0].set_ylabel(
            f"{row['label']}\nCFG {row['cfg_values'][frame_idx]:.2f}",
            fontsize=18,
            fontweight="bold",
            rotation=0,
            labelpad=78,
            va="center",
            ha="right",
        )

    fig.suptitle(f"CFG denoising comparison | step {frame_idx + 1:02d}/{args.num_frames}", fontsize=18, fontweight="bold", y=0.985)
    fig.text(0.5, 0.012, caption, ha="center", va="bottom", fontsize=18, wrap=True)
    fig.subplots_adjust(left=0.20, right=0.985, top=0.93, bottom=0.065, wspace=0.04, hspace=0.12)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    frame = rgba[..., :3].copy()
    plt.close(fig)
    return frame

def write_mp4(frames, out_path, fps):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.mimsave(out_path, frames, fps=fps, macro_block_size=1)
        return
    except Exception as imageio_error:
        try:
            import cv2

            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h),
            )
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            return
        except Exception as cv2_error:
            raise RuntimeError(
                f"Could not write mp4 with imageio ({imageio_error}) or cv2 ({cv2_error})"
            )


@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(args.seed)

    record = select_record(args.dataset_root, args.split, args.template, args.caption_index)
    max_text_len, text_embed_dim = infer_text_shapes(args.dataset_root)

    model = DiT_models[args.model](
        input_size=args.image_size // 8,
        num_classes=args.num_classes,
        text_conditioning=True,
        text_embed_dim=text_embed_dim,
        max_text_len=max_text_len,
    ).to(device)
    model.load_state_dict(load_state_dict(args.ckpt, use_ema=args.ema), strict=False)
    model.eval()
    model_dtype = next(model.parameters()).dtype
    text_cond = load_text_conditioning(args.dataset_root, record, device, dtype=model_dtype)

    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    vae.eval()
    args.vae_scaling_factor = float(vae.config.scaling_factor)

    diffusion = create_diffusion(f"ddim{args.num_frames}")
    latent_size = args.image_size // 8
    start_noise = torch.randn(1, 4, latent_size, latent_size, device=device)

    gt_path = Path(args.dataset_root) / record["image_path"]
    gt = resize_square(Image.open(gt_path), args.tile_size)

    schedules = [
        ("CFG = 1.0", lambda p: 1.0),
        ("CFG = 5.0", lambda p: 5.0),
        ("CFG 0.0 -> 5.0", lambda p: 5.0 * p),
        ("CFG 1.0 -> 5.0", lambda p: 1.0 + 4.0 * p),
    ]
    rows = []
    for label, schedule in schedules:
        final, x0_hats, cfg_values = run_schedule(args, model, vae, diffusion, text_cond, start_noise, schedule)
        rows.append({"label": label, "final": final, "x0_hats": x0_hats, "cfg_values": cfg_values})

    frames = [make_frame(gt, rows, i, args, record["caption"]) for i in range(args.num_frames)]
    write_mp4(frames, args.out, args.fps)

    sidecar = Path(args.out).with_suffix(".txt")
    sidecar.write_text(
        "\n".join(
            [
                f"checkpoint: {args.ckpt}",
                f"model: {args.model}",
                f"vae: {args.vae}",
                f"dataset_root: {args.dataset_root}",
                f"split: {args.split}",
                f"template: {args.template}",
                f"caption_index: {args.caption_index}",
                f"image_path: {record['image_path']}",
                f"caption: {record['caption']}",
                f"seed: {args.seed}",
                f"num_frames: {args.num_frames}",
                f"fps: {args.fps}",
            ]
        )
        + "\n"
    )
    print(f"wrote {args.out}")
    print(f"wrote {sidecar}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="/gpfs/scrubbed/sriyash/DiT-clevr-text-final/000-DiT-S-4-text/checkpoints/0100000-ema.pt")
    parser.add_argument("--model", type=str, default="DiT-S/4", choices=list(DiT_models.keys()))
    parser.add_argument("--vae", type=str, default="stabilityai/sdxl-vae")
    parser.add_argument("--dataset-root", type=str, default="/gpfs/scrubbed/sriyash/clevr_dit_dataset")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--template", type=str, default="chain")
    parser.add_argument("--caption-index", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-frames", type=int, default=50)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="results/eval_samples/cfg_denoising_schedules.mp4")
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
