"""Reconstruct CLEVR images through the VAE to visualize reconstruction quality."""
import argparse
import sys
from pathlib import Path

import torch
from diffusers.models import AutoencoderKL
from PIL import Image
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.utils import load_jsonl
from datasets.clevr.utils import build_clevr_transform


def main(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    records = [row for row in load_jsonl(Path(args.data_path) / "metadata.jsonl") if row["split"] == args.split]
    if args.num_images > len(records):
        raise ValueError(f"Requested {args.num_images} images, but split has only {len(records)} records.")

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(records), generator=generator)[:args.num_images].tolist()
    transform = build_clevr_transform(args.image_size)
    images = [
        transform(Image.open(Path(args.data_path) / records[idx]["image_path"]).convert("RGB"))
        for idx in indices
    ]
    x = torch.stack(images).to(device)

    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    vae.eval()

    reconstructions = []
    with torch.no_grad():
        for batch in x.split(args.batch_size):
            latents = vae.encode(batch).latent_dist.sample()
            reconstructions.append(vae.decode(latents).sample)
    recon = torch.cat(reconstructions, dim=0)

    rows = []
    for original, reconstructed in zip(x.cpu(), recon.cpu()):
        rows.extend([original, reconstructed])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(torch.stack(rows), out, nrow=2, normalize=True, value_range=(-1, 1))
    print(f"Saved VAE reconstruction grid to {out} (each row: original, reconstruction)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--vae", type=str, default="stabilityai/sdxl-vae")
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="results/vae_reconstructions.png")
    main(parser.parse_args())
