#!/usr/bin/env python3
"""Visualize train-set fit: ground truth, training input, and model prediction."""
import argparse
import json
import random
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers.models import AutoencoderKL
from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets_clevr import ClevrContextMultiDataset, pad_contexts  # noqa: E402
from diffusion import create_diffusion  # noqa: E402
from models import DiT_models  # noqa: E402


def cfg_get(cfg, key, default=None):
    cur = cfg
    for part in key.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
    return cur


def load_checkpoint(path, use_ema=True):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        if "config" in checkpoint:
            config = OmegaConf.create(checkpoint["config"])
        else:
            config = None
        if "ema" in checkpoint or "model" in checkpoint:
            key = "ema" if use_ema and "ema" in checkpoint else "model"
            return checkpoint[key], config
    return checkpoint, None


def load_config(args, checkpoint_config):
    if args.config:
        return OmegaConf.load(args.config)
    if checkpoint_config is not None:
        return checkpoint_config
    raise ValueError("Provide --config when the checkpoint does not contain a config.")


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def tensor_to_pil(x):
    x = x.detach().float().cpu().clamp(-1, 1)
    x = (x + 1) / 2
    arr = (x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def generated_image_for_row(dataset, row_idx):
    generated_idx = int(dataset._data["generated_image_index"][row_idx])
    if generated_idx < 0:
        return None
    if dataset._generated_image_cache is not None and generated_idx in dataset._generated_image_cache:
        arr = np.asarray(dataset._generated_image_cache[generated_idx])
    else:
        arr = np.asarray(dataset._data["generated_images"][generated_idx])
    return Image.fromarray(arr, mode="RGB")


def sanitize_filename(text):
    keep = []
    for char in str(text):
        keep.append(char if char.isalnum() or char in ("-", "_") else "_")
    return "".join(keep).strip("_") or "dataset"


def output_path_for_dataset(out, source_name, multiple):
    path = Path(out)
    if not multiple:
        return path
    return path.with_name(f"{path.stem}_{sanitize_filename(source_name)}{path.suffix}")


def sample_local_indices(dataset, num_samples, seed):
    rng = random.Random(seed)
    count = min(num_samples, len(dataset))
    return rng.sample(range(len(dataset)), count) if count > 0 else []


def collect_examples(dataset, source_idx, local_indices, image_size):
    examples = []
    contexts = []
    source = dataset.datasets[source_idx]
    for local_idx in local_indices:
        row_idx = int(source.indices[int(local_idx)])
        record = source.record_for_index(local_idx)
        gt_image = center_crop_arr(source.image_for_row(row_idx), image_size)
        attempted = generated_image_for_row(source, row_idx)
        if attempted is not None:
            attempted = center_crop_arr(attempted, image_size)
        contexts.append(source.context_for_row(row_idx))
        examples.append({
            "source_name": dataset.names[source_idx],
            "record": record,
            "gt_image": gt_image,
            "attempted_image": attempted,
        })
    return examples, contexts


@torch.no_grad()
def sample_predictions(args, cfg, model, vae, diffusion, contexts, device):
    context_tokens, context_mask = pad_contexts([ctx.to(dtype=torch.float32) for ctx in contexts])
    context_tokens = context_tokens.to(device)
    context_mask = context_mask.to(device)
    latent_size = int(cfg.model.image_size) // 8
    generator = torch.Generator(device=device).manual_seed(args.seed)
    z = torch.randn(len(contexts), 4, latent_size, latent_size, device=device, generator=generator)
    model_kwargs = {
        "context_tokens": context_tokens,
        "context_mask": context_mask,
    }
    if args.cfg_scale > 1:
        z = torch.cat([z, z], 0)
        model_kwargs = {
            key: value.repeat(2, *([1] * (value.ndim - 1))) if torch.is_tensor(value) else value
            for key, value in model_kwargs.items()
        }
        model_kwargs["cfg_scale"] = args.cfg_scale
        forward_fn = model.forward_with_cfg
    else:
        forward_fn = model.forward
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
    decoded = vae.decode(samples / vae.config.scaling_factor).sample
    return [tensor_to_pil(image) for image in decoded]


def wrapped(text, width=34, max_lines=8):
    lines = textwrap.wrap(str(text), width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".") + "..."
    return "\n".join(lines)


def draw_input_panel(ax, example):
    record = example["record"]
    caption = record.get("caption", "")
    feedback = record.get("feedback", "")
    attempted = example["attempted_image"]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    if attempted is not None:
        ax.imshow(attempted)
        title = f"{example['source_name']} input"
        text = f"feedback: {feedback}" if feedback else "feedback row"
        ax.set_title(wrapped(title + "\n" + text, width=28, max_lines=4), fontsize=8)
    else:
        ax.axis("off")
        text = f"{example['source_name']} input\ncaption: {caption}"
        ax.text(0.0, 1.0, wrapped(text, width=36, max_lines=10), va="top", ha="left", fontsize=8)


def plot_grid(args, examples, predictions, pretrained_predictions=None):
    n = len(examples)
    has_pretrained = pretrained_predictions is not None
    content_cols = 4 if has_pretrained else 3
    ncols = content_cols + (content_cols - 1)
    width_ratios = []
    for idx in range(content_cols):
        width_ratios.append(1.0)
        if idx < content_cols - 1:
            width_ratios.append(0.06)
    fig, axes = plt.subplots(
        n,
        ncols,
        figsize=(3.1 * content_cols + 0.2 * (content_cols - 1), max(2.2, 2.45 * n)),
        dpi=args.dpi,
        squeeze=False,
        gridspec_kw={"width_ratios": width_ratios},
    )
    image_cols = [idx * 2 for idx in range(content_cols)]
    titles = ["GT", "Input", "Pred"] + (["Pretrained Pred"] if has_pretrained else [])
    for ax, title in zip([axes[0, col] for col in image_cols], titles):
        ax.set_title(title, fontsize=10, fontweight="bold")
    for row, (example, pred) in enumerate(zip(examples, predictions)):
        axes[row, image_cols[0]].imshow(example["gt_image"])
        axes[row, image_cols[0]].set_xticks([])
        axes[row, image_cols[0]].set_yticks([])
        axes[row, image_cols[0]].set_ylabel(
            wrapped(example["record"].get("caption", ""), width=30, max_lines=4),
            fontsize=7,
            rotation=0,
            ha="right",
            va="center",
            labelpad=64,
        )
        for sep_col in range(1, ncols, 2):
            axes[row, sep_col].set_facecolor("#b8b8b8")
            axes[row, sep_col].set_xticks([])
            axes[row, sep_col].set_yticks([])
            axes[row, sep_col].set_xlim(0, 1)
            axes[row, sep_col].set_ylim(0, 1)
            for spine in axes[row, sep_col].spines.values():
                spine.set_visible(False)
        draw_input_panel(axes[row, image_cols[1]], example)
        axes[row, image_cols[2]].imshow(pred)
        axes[row, image_cols[2]].set_xticks([])
        axes[row, image_cols[2]].set_yticks([])
        if has_pretrained:
            axes[row, image_cols[3]].imshow(pretrained_predictions[row])
            axes[row, image_cols[3]].set_xticks([])
            axes[row, image_cols[3]].set_yticks([])
    fig.tight_layout(pad=0.8, w_pad=0.8, h_pad=1.1)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_model(cfg, context_dim, state_dict, device):
    model = DiT_models[cfg.model.name](
        input_size=int(cfg.model.image_size) // 8,
        num_classes=int(cfg.model.num_classes),
        text_conditioning=True,
        context_dim=context_dim,
        class_dropout_prob=float(cfg_get(cfg.model, "context_dropout_prob", 0.0)),
    ).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")
    model.eval()
    return model


def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    state_dict, checkpoint_config = load_checkpoint(args.ckpt, args.ema)
    cfg = load_config(args, checkpoint_config)
    if args.dataset_config:
        cfg.data.dataset_config = OmegaConf.load(args.dataset_config)
    transform = None
    dataset = ClevrContextMultiDataset(
        cfg.data.dataset_config,
        transform=transform,
        split=args.split,
        use_disk=cfg_get(cfg.data, "use_disk", True),
        load_meta=cfg_get(cfg.data, "load_meta", True),
        load_images=False,
        load_context=False,
    )

    model = build_model(cfg, dataset.context_dim, state_dict, device)
    pretrained_model = None
    pretrained_ckpt = cfg_get(cfg.train, "ckpt", None)
    if pretrained_ckpt and Path(pretrained_ckpt).resolve() != Path(args.ckpt).resolve():
        pretrained_state, _ = load_checkpoint(pretrained_ckpt, args.ema)
        pretrained_model = build_model(cfg, dataset.context_dim, pretrained_state, device)
        print(f"Loaded pretrained comparison checkpoint: {pretrained_ckpt}")

    vae = AutoencoderKL.from_pretrained(cfg.model.vae).to(device)
    vae.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps))
    multiple = len(dataset.datasets) > 1
    summary = {
        "ckpt": args.ckpt,
        "pretrained_ckpt": pretrained_ckpt if pretrained_model is not None else None,
        "config": args.config,
        "split": args.split,
        "datasets": [],
    }
    for source_idx, source in enumerate(dataset.datasets):
        source_name = dataset.names[source_idx]
        local_indices = sample_local_indices(source, args.num_samples, args.seed + source_idx)
        examples, contexts = collect_examples(dataset, source_idx, local_indices, int(cfg.model.image_size))
        if not examples:
            continue
        predictions = sample_predictions(args, cfg, model, vae, diffusion, contexts, device)
        pretrained_predictions = None
        if pretrained_model is not None:
            pretrained_predictions = sample_predictions(args, cfg, pretrained_model, vae, diffusion, contexts, device)
        out_path = output_path_for_dataset(args.out, source_name, multiple)
        plot_args = argparse.Namespace(**vars(args))
        plot_args.out = str(out_path)
        plot_grid(plot_args, examples, predictions, pretrained_predictions=pretrained_predictions)
        summary["datasets"].append({
            "name": source_name,
            "out": str(out_path),
            "local_indices": local_indices,
            "records": [
                {
                    "source_name": example["source_name"],
                    **example["record"],
                }
                for example in examples
            ],
        })
        print(f"Saved {source_name} train-fit grid to {out_path}")

    metadata_path = Path(args.out).with_suffix(".json")
    with metadata_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved sampled records to {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/train_adaptive.yaml")
    parser.add_argument("--dataset-config", type=str, default=None, help="Optional YAML list overriding config.data.dataset_config.")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", type=str, default="results/train_fit_grid.png")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
