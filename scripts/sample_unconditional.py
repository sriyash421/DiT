"""
Sample images from an unconditional CLEVR DiT checkpoint.
"""
import argparse

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image

from diffusion import create_diffusion
from models import DiT_models


def load_state_dict(path, use_ema):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and ("ema" in checkpoint or "model" in checkpoint):
        key = "ema" if use_ema and "ema" in checkpoint else "model"
        return checkpoint[key]
    return checkpoint


def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device)
    model.load_state_dict(load_state_dict(args.ckpt, args.ema), strict=True)
    model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    vae_scaling_factor = vae.config.scaling_factor
    vae.eval()

    z = torch.randn(args.num_samples, 4, latent_size, latent_size, device=device)
    label = args.label if args.label is not None else args.num_classes
    y = torch.full((args.num_samples,), label, device=device, dtype=torch.long)
    model_kwargs = dict(y=y)

    samples = diffusion.p_sample_loop(
        model.forward,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        device=device,
    )
    samples = vae.decode(samples / vae_scaling_factor).sample
    save_image(samples, args.out, nrow=args.nrow, normalize=True, value_range=(-1, 1))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--label", type=int, default=None, help="Class label to sample. Defaults to the null class.")
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--nrow", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="sample_unconditional.png")
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
