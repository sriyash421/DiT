#!/usr/bin/env python3
"""Sample from a Qwen-context conditioned DiT checkpoint."""
import argparse
import json
from pathlib import Path

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image

from clevr_captions import render_caption
from diffusion import create_diffusion
from models import DiT_models
from vlm_utils import build_context_text, encode_contexts, load_vlm


def load_checkpoint(path, use_ema):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and ("ema" in checkpoint or "model" in checkpoint):
        return checkpoint["ema" if use_ema and "ema" in checkpoint else "model"]
    return checkpoint


def caption_and_metadata(args):
    if args.caption is not None:
        return args.caption, None
    with Path(args.metadata).open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    row = rows[args.metadata_row]
    return render_caption(row, template=args.template), row


@torch.no_grad()
def main(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    caption, metadata = caption_and_metadata(args)
    state_dict = load_checkpoint(args.ckpt, args.ema)
    context_dim = state_dict["null_context"].shape[-1]

    model = DiT_models[args.model](
        input_size=args.image_size // 8,
        num_classes=args.num_classes,
        text_conditioning=True,
        context_dim=context_dim,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    processor, vlm = load_vlm(args.vlm_model, device, dtype=args.vlm_dtype, device_map=args.device_map)
    context_text = build_context_text(caption, metadata)
    context_tokens, context_mask = encode_contexts(
        processor,
        vlm,
        [context_text],
        device,
        max_length=args.max_context_len,
        out_dtype=torch.float16,
    )
    context_tokens = context_tokens.float()

    latent_size = args.image_size // 8
    z = torch.randn(args.num_samples, 4, latent_size, latent_size, device=device)
    context_tokens = context_tokens.repeat(args.num_samples, 1, 1)
    context_mask = context_mask.repeat(args.num_samples, 1)
    if args.cfg_scale <= 1:
        model_kwargs = {
            "context_tokens": context_tokens.to(device),
            "context_mask": context_mask.to(device),
        }
        forward_fn = model.forward
    else:
        z = torch.cat([z, z], dim=0)
        model_kwargs = {
            "context_tokens": context_tokens.repeat(2, 1, 1).to(device),
            "context_mask": context_mask.repeat(2, 1).to(device),
            "cfg_scale": args.cfg_scale,
        }
        forward_fn = model.forward_with_cfg

    diffusion = create_diffusion(str(args.num_sampling_steps))
    samples = diffusion.p_sample_loop(
        forward_fn,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        device=device,
    )
    if args.cfg_scale > 1:
        samples, _ = samples.chunk(2, dim=0)
    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    vae.eval()
    samples = vae.decode(samples / vae.config.scaling_factor).sample
    save_image(samples, args.out, nrow=args.nrow, normalize=True, value_range=(-1, 1))
    print(f"Caption: {caption}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/4")
    parser.add_argument("--vae", type=str, default="stabilityai/sdxl-vae")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--caption", type=str, default=None)
    parser.add_argument("--metadata", type=str, default=None)
    parser.add_argument("--metadata-row", type=int, default=0)
    parser.add_argument("--template", type=str, default="chain")
    parser.add_argument("--vlm-model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--vlm-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--max-context-len", type=int, default=1024)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--nrow", type=int, default=4)
    parser.add_argument("--out", type=str, default="sample_text.png")
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.device_map == "":
        args.device_map = None
    if args.caption is None and args.metadata is None:
        raise ValueError("Provide either --caption or --metadata.")
    main(args)
