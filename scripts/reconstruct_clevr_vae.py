"""
Reconstruct CLEVR dataset images through the Stable Diffusion VAE.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from diffusers.models import AutoencoderKL
from PIL import Image
from torchvision.utils import save_image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clevr_transforms import build_clevr_transform  # noqa: E402


def load_records(data_path, split):
    metadata_path = Path(data_path) / "metadata.jsonl"
    with metadata_path.open() as f:
        records = [json.loads(line) for line in f if line.strip()]
    return [record for record in records if split is None or record["split"] == split]


def main(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    records = load_records(args.data_path, args.split)
    if args.num_images > len(records):
        raise ValueError(f"Requested {args.num_images} images, but split has only {len(records)} records.")

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(records), generator=generator)[:args.num_images].tolist()

    transform = build_clevr_transform(args.image_size)

    images = []
    data_root = Path(args.data_path)
    for idx in indices:
        image = Image.open(data_root / records[idx]["image_path"]).convert("RGB")
        images.append(transform(image))
    x = torch.stack(images).to(device)

    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    vae.eval()
    vae_scaling_factor = vae.config.scaling_factor

    reconstructions = []
    with torch.no_grad():
        for batch in x.split(args.batch_size):
            latent_dist = vae.encode(batch).latent_dist
            latents = latent_dist.mode() if args.use_mode else latent_dist.sample()
            latents = latents * vae_scaling_factor
            recon = vae.decode(latents / vae_scaling_factor).sample
            reconstructions.append(recon)
    recon = torch.cat(reconstructions, dim=0)

    rows = []
    for original, reconstructed in zip(x.cpu(), recon.cpu()):
        rows.extend([original, reconstructed])
    grid = torch.stack(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out, nrow=2, normalize=True, value_range=(-1, 1))
    print(f"Saved VAE reconstruction grid to {out}")
    print("Each row is: original, reconstruction")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/gpfs/scrubbed/sriyash/clevr_dit_dataset")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--vae", type=str, default="stabilityai/sdxl-vae")
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="results/vae_reconstructions.png")
    parser.add_argument("--use-mode", action="store_true", help="Use latent distribution mode instead of sampling.")
    parser.add_argument("--cpu", action="store_true", help="Run VAE on CPU even if CUDA is available.")
    main(parser.parse_args())
