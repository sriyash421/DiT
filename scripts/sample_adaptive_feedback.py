#!/usr/bin/env python3
"""Sample with caption + feedback + attempted-image adaptive context."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from diffusers.models import AutoencoderKL
from PIL import Image
from torchvision.utils import save_image
from transformers import AutoTokenizer, T5EncoderModel

from clevr_captions import render_caption
from diffusion import create_diffusion
from models import DiT_models


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def image_to_tensor(path, image_size):
    image = center_crop_arr(Image.open(path).convert("RGB"), image_size)
    arr = np.asarray(image).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def masked_mean(hidden, mask):
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


@torch.no_grad()
def encode_texts(texts, encoder_name, max_length, device):
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    encoder = T5EncoderModel.from_pretrained(encoder_name).to(device)
    encoder.eval()
    encoded = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    hidden = encoder(**encoded).last_hidden_state
    mask = encoded["attention_mask"]
    hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)
    pooled = masked_mean(hidden, mask)
    return hidden, mask.bool(), pooled


def caption_from_metadata(args):
    with Path(args.metadata).open() as f:
        rows = [json.loads(line) for line in f]
    return render_caption(rows[args.metadata_row], template=args.template)


@torch.no_grad()
def main(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    caption = args.caption if args.caption is not None else caption_from_metadata(args)

    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        text_conditioning=True,
        text_embed_dim=args.text_embed_dim,
        max_text_len=args.max_text_len,
    ).to(device)
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and ("ema" in checkpoint or "model" in checkpoint):
        key = "ema" if args.ema and "ema" in checkpoint else "model"
        state_dict = checkpoint[key]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    vae.eval()
    vae_scaling_factor = vae.config.scaling_factor
    diffusion = create_diffusion(str(args.num_sampling_steps))

    text_tokens, text_mask, text_pooled = encode_texts([caption], args.encoder, args.max_text_len, device)
    model_kwargs = {
        "text_tokens": text_tokens.repeat(args.num_samples, 1, 1),
        "text_mask": text_mask.repeat(args.num_samples, 1),
        "text_pooled": text_pooled.repeat(args.num_samples, 1),
    }
    if args.feedback is not None:
        feedback_tokens, feedback_mask, feedback_pooled = encode_texts([args.feedback], args.encoder, args.max_text_len, device)
        model_kwargs.update({
            "feedback_tokens": feedback_tokens.repeat(args.num_samples, 1, 1),
            "feedback_mask": feedback_mask.repeat(args.num_samples, 1),
            "feedback_pooled": feedback_pooled.repeat(args.num_samples, 1),
        })
    if args.attempt_image is not None:
        attempt = image_to_tensor(args.attempt_image, args.image_size).unsqueeze(0).to(device)
        attempt_latent = vae.encode(attempt).latent_dist.mode().mul_(vae_scaling_factor)
        model_kwargs["attempt_latent"] = attempt_latent.repeat(args.num_samples, 1, 1, 1)

    z = torch.randn(args.num_samples, 4, latent_size, latent_size, device=device)
    if args.cfg_scale <= 1:
        forward_fn = model.forward
    else:
        z = torch.cat([z, z], 0)
        model_kwargs = {
            key: value.repeat(2, *([1] * (value.ndim - 1))) if torch.is_tensor(value) else value
            for key, value in model_kwargs.items()
        }
        model_kwargs["cfg_scale"] = args.cfg_scale
        forward_fn = model.forward_with_text_cfg

    sample_loop = diffusion.ddim_sample_loop if args.sampler == "ddim" else diffusion.p_sample_loop
    samples = sample_loop(
        forward_fn,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        device=device,
        **({"eta": args.ddim_eta} if args.sampler == "ddim" else {}),
    )
    if args.cfg_scale > 1:
        samples, _ = samples.chunk(2, dim=0)
    samples = vae.decode(samples / vae_scaling_factor).sample
    save_image(samples, args.out, nrow=args.nrow, normalize=True, value_range=(-1, 1))
    print(f"Caption: {caption}")
    if args.feedback:
        print(f"Feedback: {args.feedback}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/4")
    parser.add_argument("--vae", type=str, default="stabilityai/sdxl-vae")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--caption", type=str, default=None)
    parser.add_argument("--feedback", type=str, default=None)
    parser.add_argument("--attempt-image", type=str, default=None)
    parser.add_argument("--metadata", type=str, default=None)
    parser.add_argument("--metadata-row", type=int, default=0)
    parser.add_argument("--template", type=str, default="chain")
    parser.add_argument("--encoder", type=str, default="google/flan-t5-large")
    parser.add_argument("--text-embed-dim", type=int, default=1024)
    parser.add_argument("--max-text-len", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--nrow", type=int, default=4)
    parser.add_argument("--out", type=str, default="adaptive_sample.png")
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.caption is None and args.metadata is None:
        raise ValueError("Provide either --caption or --metadata.")
    main(args)
