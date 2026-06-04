"""
Sample images from a text-conditioned DiT checkpoint.
"""
import argparse
import json
from pathlib import Path

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image
from transformers import AutoTokenizer, T5EncoderModel

from clevr_captions import render_caption
from diffusion import create_diffusion
from download import find_model
from models import DiT_models


def masked_mean(hidden, mask):
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


@torch.no_grad()
def encode_caption(caption, encoder_name, max_length, device):
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    encoder = T5EncoderModel.from_pretrained(encoder_name).to(device)
    encoder.eval()
    encoded = tokenizer(
        [caption],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
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


def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    caption = args.caption if args.caption is not None else caption_from_metadata(args)

    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        text_conditioning=True,
        text_embed_dim=args.text_embed_dim,
        max_text_len=args.max_text_len,
    ).to(device)
    # state_dict = find_model(args.ckpt)
    state_dict = torch.load(args.ckpt)["model"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    text_tokens, text_mask, text_pooled = encode_caption(caption, args.encoder, args.max_text_len, device)
    text_tokens = text_tokens.repeat(args.num_samples * 2, 1, 1)
    text_mask = text_mask.repeat(args.num_samples * 2, 1)
    text_pooled = text_pooled.repeat(args.num_samples * 2, 1)

    z = torch.randn(args.num_samples, 4, latent_size, latent_size, device=device)
    z = torch.cat([z, z], 0)
    model_kwargs = dict(
        text_tokens=text_tokens,
        text_mask=text_mask,
        text_pooled=text_pooled,
        cfg_scale=args.cfg_scale,
    )
    samples = diffusion.p_sample_loop(
        model.forward_with_text_cfg,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        device=device,
    )
    samples, _ = samples.chunk(2, dim=0)
    samples = vae.decode(samples / 0.18215).sample
    save_image(samples, args.out, nrow=args.nrow, normalize=True, value_range=(-1, 1))
    print(f"Caption: {caption}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--caption", type=str, default=None)
    parser.add_argument("--metadata", type=str, default=None)
    parser.add_argument("--metadata-row", type=int, default=0)
    parser.add_argument("--template", type=str, default="chain")
    parser.add_argument("--encoder", type=str, default="google/flan-t5-large")
    parser.add_argument("--text-embed-dim", type=int, default=1024)
    parser.add_argument("--max-text-len", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--nrow", type=int, default=4)
    parser.add_argument("--out", type=str, default="sample_text.png")
    args = parser.parse_args()
    if args.caption is None and args.metadata is None:
        raise ValueError("Provide either --caption or --metadata.")
    main(args)
