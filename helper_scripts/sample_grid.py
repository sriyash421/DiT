"""Sample a fixed grid from a DiT checkpoint: N captions x M samples per caption, plus GT column."""
import argparse
import random
import sys
from pathlib import Path

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.on_policy import PolicySampler
from algorithms.utils import load_checkpoint
from datasets.clevr.dataset import ClevrContextDataset, pad_contexts
from diffusion import create_diffusion
from models.qwen_dit import DiT_models


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = ClevrContextDataset(args.dataset_root, transform=None, split=args.split)
    rng = random.Random(args.caption_seed)
    indices = rng.sample(range(len(dataset)), args.num_captions)
    records = [dataset.record_for_index(idx) for idx in indices]

    from diffusers.models import AutoencoderKL

    model = DiT_models[args.model](
        input_size=args.image_size // 8,
        context_dim=dataset.context_dim,
    ).to(device)
    model.load_state_dict(load_checkpoint(args.ckpt, use_ema=args.ema), strict=True)
    model.eval()
    vae = AutoencoderKL.from_pretrained(args.vae).to(device).eval()
    sampler = PolicySampler(
        create_diffusion(str(args.num_sampling_steps)),
        latent_size=args.image_size // 8,
        vae_scaling_factor=vae.config.scaling_factor,
        cfg_scale=args.cfg_scale,
        ddim_eta=args.ddim_eta,
    )

    contexts = []
    for record in records:
        token = dataset.context_for_row(record["row_idx"], dtype="float32")
        contexts.extend([token] * args.samples_per_caption)
    context_tokens, context_mask = pad_contexts(contexts)

    images = []
    for start in range(0, len(contexts), args.batch_size):
        end = start + args.batch_size
        _, batch_images = sampler.sample(
            model,
            vae,
            context_tokens[start:end],
            context_mask[start:end],
            device,
            seed=args.seed + start,
        )
        images.extend(batch_images)

    tile = args.image_size
    out = Image.new("RGB", (tile * (args.samples_per_caption + 1), tile * args.num_captions), (255, 255, 255))
    for row, record in enumerate(records):
        gt = dataset.image_for_row(record["row_idx"]).resize((tile, tile))
        out.paste(gt, (0, row * tile))
        for col in range(args.samples_per_caption):
            out.paste(images[row * args.samples_per_caption + col], ((col + 1) * tile, row * tile))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    with out_path.with_suffix(".captions.txt").open("w") as f:
        for row, record in enumerate(records):
            f.write(f"{row}\t{record['caption']}\n")
    print(f"Saved grid to {out_path} (first column is GT)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model", type=str, default="DiT-L/4", choices=list(DiT_models.keys()))
    parser.add_argument("--vae", type=str, default="stabilityai/sdxl-vae")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num-captions", type=int, default=8)
    parser.add_argument("--samples-per-caption", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--caption-seed", type=int, default=0)
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", type=str, default="results/sample_grid.png")
    main(parser.parse_args())
