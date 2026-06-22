"""
Reconstruct CLEVR dataset images through the Stable Diffusion VAE.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers.models import AutoencoderKL
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


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

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])

    images = []
    data_root = Path(args.data_path)
    for idx in indices:
        image = Image.open(data_root / records[idx]["image_path"]).convert("RGB")
        images.append(transform(image))
    x = torch.stack(images).to(device)

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    reconstructions = []
    with torch.no_grad():
        for batch in x.split(args.batch_size):
            latent_dist = vae.encode(batch).latent_dist
            latents = latent_dist.mode() if args.use_mode else latent_dist.sample()
            latents = latents * 0.18215
            recon = vae.decode(latents / 0.18215).sample
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
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="results/vae_reconstructions.png")
    parser.add_argument("--use-mode", action="store_true", help="Use latent distribution mode instead of sampling.")
    parser.add_argument("--cpu", action="store_true", help="Run VAE on CPU even if CUDA is available.")
    main(parser.parse_args())
