#!/usr/bin/env python3
"""Iteratively evaluate adaptive feedback: image -> Gemini feedback -> revised image."""
import argparse
import json
import os
import random
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from diffusers.models import AutoencoderKL
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion import create_diffusion  # noqa: E402
from datasets_clevr import ClevrContextDataset  # noqa: E402
from feedback_verifiers import (  # noqa: E402
    DEFAULT_GEMINI_MODEL,
    OPENROUTER_API_URL,
    build_feedback_verifier,
)
from models import DiT_models  # noqa: E402
from vlm_utils import build_context_text, encode_contexts, load_metadata_rows, load_vlm, metadata_for_row  # noqa: E402

OPENROUTER_GEMINI_MODEL_ID = DEFAULT_GEMINI_MODEL


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def pil_to_tensor(image, image_size):
    image = center_crop_arr(image.convert("RGB"), image_size)
    arr = np.asarray(image).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def tensor_to_pil(x):
    x = x.detach().float().cpu().clamp(-1, 1)
    x = (x + 1) / 2
    x = (x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(x)


def load_checkpoint(path, use_ema=True):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and ("ema" in checkpoint or "model" in checkpoint):
        return checkpoint["ema" if use_ema and "ema" in checkpoint else "model"]
    return checkpoint


def select_record(dataset_root, split, caption_seed, caption_index):
    records, index = select_records(dataset_root, split, caption_seed, caption_index, 1)
    return records[0], index


def select_records(dataset_root, split, caption_seed, caption_index, num_captions):
    if not dataset_root:
        raise ValueError("--dataset-root must be a non-empty path")
    dataset = ClevrContextDataset(dataset_root, transform=None, split=split)
    if len(dataset) == 0:
        raise RuntimeError(f"No records for split={split}")
    rng = random.Random(caption_seed)
    indices = rng.sample(range(len(dataset)), len(dataset))
    selected = []
    for i in range(num_captions):
        idx = indices[(caption_index + i) % len(indices)]
        record = dataset.record_for_index(idx)
        record["_dataset"] = dataset
        selected.append(record)
    return selected, {"context_dim": dataset.context_dim}


def jsonable_record(record):
    return {key: value for key, value in record.items() if not key.startswith("_")}


def generated_image_for_row(dataset, row_idx):
    generated_idx = int(dataset._data["generated_image_index"][row_idx])
    if generated_idx < 0:
        return None
    if dataset._generated_image_cache is not None and generated_idx in dataset._generated_image_cache:
        return Image.fromarray(np.asarray(dataset._generated_image_cache[generated_idx]), mode="RGB")
    return Image.fromarray(np.asarray(dataset._data["generated_images"][generated_idx]), mode="RGB")


def training_lookup_keys(record):
    metadata_index = int(record.get("metadata_index", -1))
    template = str(record.get("template", ""))
    caption = str(record.get("caption", ""))
    return [
        ("full", metadata_index, template, caption),
        ("caption", metadata_index, caption),
        ("metadata", metadata_index),
    ]


def build_training_feedback_lookup(args):
    if args.split != "train" or not args.train_feedback_dataset_root:
        return None, {}
    dataset = ClevrContextDataset(args.train_feedback_dataset_root, transform=None, split="train")
    lookup = {}
    for idx in range(len(dataset)):
        record = dataset.record_for_index(idx)
        if not record.get("feedback") or not record.get("generated_image_path"):
            continue
        entry = {
            "row_idx": record["row_idx"],
            "feedback": record["feedback"],
            "sample_index": record.get("sample_index", -1),
            "feedback_index": record.get("feedback_index", -1),
            "tuple_id": record.get("tuple_id", ""),
        }
        for key in training_lookup_keys(record):
            lookup.setdefault(key, []).append(entry)
    for entries in lookup.values():
        entries.sort(key=lambda item: (int(item.get("sample_index", -1)), int(item.get("feedback_index", -1))))
    print(f"Loaded {len(dataset)} train feedback rows from {args.train_feedback_dataset_root}")
    return dataset, lookup


def matched_training_pairs(args, lookup, record):
    for key in training_lookup_keys(record):
        entries = lookup.get(key)
        if entries:
            return entries[:args.max_training_pairs_per_caption]
    return []


def attach_training_pairs(args, trace, training_dataset, pairs):
    if not training_dataset or not pairs:
        trace["training_data"] = []
        return
    pair_dir = Path(trace["out_dir"]) / "training_pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    training_data = []
    for idx, pair in enumerate(pairs):
        image = generated_image_for_row(training_dataset, pair["row_idx"])
        if image is None:
            continue
        image = center_crop_arr(image, args.image_size)
        image_path = pair_dir / f"training_{idx:02d}.png"
        image.save(image_path)
        training_data.append({
            "image_path": str(image_path),
            "feedback": pair["feedback"],
            "sample_index": pair.get("sample_index", -1),
            "feedback_index": pair.get("feedback_index", -1),
            "tuple_id": pair.get("tuple_id", ""),
        })
    trace["training_data"] = training_data
    with (Path(trace["out_dir"]) / "trace.json").open("w") as f:
        json.dump(trace, f, indent=2)


def build_eval_feedback_verifier(args):
    if args.feedback_vlm == "gemini":
        api_key = os.getenv(args.openrouter_api_key_env)
        if not api_key:
            raise ValueError(f"Set {args.openrouter_api_key_env} for OpenRouter access")
        return build_feedback_verifier(
            backend="openai-chat",
            model=args.gemini_model,
            api_url=OPENROUTER_API_URL,
            api_key=api_key,
            temperature=args.feedback_temperature,
            max_tokens=args.max_feedback_tokens,
            retries=args.openrouter_retries,
            include_caption=args.use_caption,
            include_metadata=False,
            image_size=args.image_size,
        )
    if args.feedback_vlm == "qwen-vllm":
        return build_feedback_verifier(
            backend="qwen-vllm",
            model=args.vllm_model,
            api_url=args.vllm_base_url,
            api_key=args.vllm_api_key,
            temperature=args.feedback_temperature,
            max_tokens=args.max_feedback_tokens,
            retries=args.openrouter_retries,
            enable_thinking=args.enable_thinking,
            include_caption=args.use_caption,
            include_metadata=False,
            image_size=args.image_size,
        )
    raise ValueError(args.feedback_vlm)


@torch.no_grad()
def sample_image(args, model, vae, diffusion, processor, vlm, metadata, caption, feedback, current_image, device, seed):
    latent_size = args.image_size // 8
    context_text = build_context_text(caption, metadata, feedback)
    images = [current_image] if feedback is not None and current_image is not None else [None]
    context_tokens, context_mask = encode_contexts(
        processor,
        vlm,
        [context_text],
        device,
        images=images,
        max_length=args.max_context_len,
        out_dtype=torch.float16,
    )
    model_kwargs = {
        "context_tokens": context_tokens.float().to(device),
        "context_mask": context_mask.to(device),
    }

    generator = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(1, 4, latent_size, latent_size, device=device, generator=generator)
    if args.cfg_scale <= 1:
        forward_fn = model.forward
    else:
        z = torch.cat([z, z], 0)
        model_kwargs = {
            key: value.repeat(2, *([1] * (value.ndim - 1))) if torch.is_tensor(value) else value
            for key, value in model_kwargs.items()
        }
        model_kwargs["cfg_scale"] = args.cfg_scale
        forward_fn = model.forward_with_cfg

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
    return tensor_to_pil(decoded[0])


def _short_title(text, width=28, max_lines=4):
    if not text:
        return ""
    lines = textwrap.wrap(str(text), width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".") + "..."
    return "\n".join(lines)


def plot_feedback_loop_grid(args, traces, out_path):
    if not traces:
        return
    nrows = len(traces)
    max_steps = max(len(trace["steps"]) for trace in traces)
    max_training = max(len(trace.get("training_data", [])) for trace in traces)
    separator_cols = 1 if max_training > 0 else 0
    ncols = 1 + max_steps + separator_cols + max_training
    width_ratios = [1.0] * (1 + max_steps)
    if max_training > 0:
        width_ratios.extend([0.08] + [1.0] * max_training)
    fig_w = max(8, 2.15 * (1 + max_steps + max_training) + 0.25 * separator_cols)
    fig_h = max(2.4, 2.25 * nrows)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        dpi=args.dpi,
        squeeze=False,
        gridspec_kw={"width_ratios": width_ratios},
    )
    separator_col = 1 + max_steps

    for row_idx, trace in enumerate(traces):
        gt_image = Image.open(trace["gt_image_path"]).convert("RGB")
        caption_title = _short_title(trace.get("caption", ""), width=34, max_lines=3)
        for col_idx in range(ncols):
            ax = axes[row_idx, col_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            if col_idx == 0:
                ax.imshow(gt_image)
                ax.set_title("Ground Truth", fontsize=9, fontweight="bold")
                ax.set_ylabel(caption_title, fontsize=8, rotation=0, ha="right", va="center", labelpad=48)
                continue

            if max_training > 0 and col_idx == separator_col:
                ax.set_facecolor("#b8b8b8")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                if row_idx == 0:
                    ax.set_title("", fontsize=8)
                continue

            if col_idx < separator_col:
                step_idx = col_idx - 1
                if step_idx >= len(trace["steps"]):
                    ax.axis("off")
                    continue
                step = trace["steps"][step_idx]
                ax.imshow(Image.open(step["image_path"]).convert("RGB"))
                if step_idx > 0:
                    ax.set_title(_short_title(step.get("feedback_used", ""), width=24, max_lines=3), fontsize=8)
                continue

            training_idx = col_idx - separator_col - 1
            training_data = trace.get("training_data", [])
            if training_idx >= len(training_data):
                ax.axis("off")
                continue
            pair = training_data[training_idx]
            ax.imshow(Image.open(pair["image_path"]).convert("RGB"))
            title = "Training data"
            if pair.get("feedback"):
                title += "\n" + _short_title(pair["feedback"], width=24, max_lines=3)
            ax.set_title(title, fontsize=8, color="#404040")

    fig.tight_layout(pad=0.8, w_pad=0.8, h_pad=1.6)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def log_to_wandb_if_configured(args, traces, combined_path):
    if args.wandb_project is None:
        return
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        id=args.wandb_run_id,
        name=args.wandb_name,
        resume=args.wandb_resume,
        config={
            "eval_ckpt": args.ckpt,
            "eval_split": args.split,
            "eval_caption_index": args.caption_index,
            "eval_num_captions": args.num_captions,
            "eval_feedback_steps": args.steps,
            "eval_cfg_scale": args.cfg_scale,
            "eval_num_sampling_steps": args.num_sampling_steps,
            "eval_sampler": args.sampler,
            "eval_vlm_model": args.vlm_model,
        },
    )
    table = wandb.Table(columns=["caption_index", "caption", "trace_json"])
    for idx, trace in enumerate(traces):
        table.add_data(idx, trace.get("caption", ""), str(Path(trace["out_dir"]) / "trace.json"))

    payload = {
        f"{args.wandb_key}/table": table,
        f"{args.wandb_key}/combined": wandb.Image(str(combined_path), caption="combined feedback-loop traces"),
    }
    run.log(payload, step=args.wandb_step)
    run.finish()


@torch.no_grad()
def run_trace(args, model, vae, diffusion, processor, vlm, feedback_verifier, record, metadata, out_dir, device):
    out_dir.mkdir(parents=True, exist_ok=True)
    caption = record["caption"]
    gt_image = center_crop_arr(record["_dataset"].image_for_row(record["row_idx"]), args.image_size)
    gt_path = out_dir / "ground_truth.png"
    gt_image.save(gt_path)

    rows = []
    usage_totals = {}
    current_feedback = None
    current_image = None
    for step in range(args.steps + 1):
        image = sample_image(
            args,
            model,
            vae,
            diffusion,
            processor,
            vlm,
            metadata,
            caption,
            current_feedback,
            current_image,
            device,
            args.seed if args.fixed_seed_per_trace else args.seed + step,
        )
        image_path = out_dir / f"step_{step:02d}.png"
        image.save(image_path)
        feedback_for_next = ""
        usage = {}
        if step < args.steps:
            result = feedback_verifier.verify_one(caption, metadata, gt_image, image)
            if not result.ok:
                raise RuntimeError(f"feedback verifier failed: {result.error}")
            feedback_for_next = result.feedback
            usage = result.token_usage or {}
            for key, value in usage.items():
                if isinstance(value, (int, float)):
                    usage_totals[key] = usage_totals.get(key, 0) + value
        rows.append({
            "step": step,
            "image_path": str(image_path),
            "feedback_used": current_feedback,
            "feedback_for_next": feedback_for_next,
            "token_usage": usage,
        })
        print(f"step={step} image={image_path}")
        if feedback_for_next:
            print(f"feedback_for_next={feedback_for_next}")
        current_image = image
        current_feedback = feedback_for_next or None

    trace = {
        "caption": caption,
        "gt_image_path": str(gt_path),
        "source_gt_image_path": record.get("image_path", ""),
        "out_dir": str(out_dir),
        "record": jsonable_record(record),
        "steps": rows,
        "token_usage_totals": usage_totals,
        "args": vars(args),
    }
    with (out_dir / "trace.json").open("w") as f:
        json.dump(trace, f, indent=2)
    print(f"Saved trace to {out_dir / 'trace.json'}")
    print(f"Token usage totals: {usage_totals}")
    return trace, usage_totals


@torch.no_grad()
def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    records, index = select_records(
        args.dataset_root,
        args.split,
        args.caption_seed,
        args.caption_index,
        args.num_captions,
    )

    metadata_rows, metadata_by_image_path = load_metadata_rows(args.dataset_root)
    context_dim = index["context_dim"]
    model = DiT_models[args.model](
        input_size=args.image_size // 8,
        num_classes=args.num_classes,
        text_conditioning=True,
        context_dim=context_dim,
    ).to(device)
    model.load_state_dict(load_checkpoint(args.ckpt, args.ema), strict=True)
    model.eval()
    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    vae.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps))
    processor, vlm = load_vlm(args.vlm_model, device, dtype=args.vlm_dtype, device_map=args.device_map)
    feedback_verifier = build_eval_feedback_verifier(args)

    traces = []
    usage_totals = {}
    for idx, record in enumerate(records):
        metadata = metadata_for_row(record, metadata_rows, metadata_by_image_path)
        trace, usage = run_trace(
            args,
            model,
            vae,
            diffusion,
            processor,
            vlm,
            feedback_verifier,
            record,
            metadata,
            out_dir / f"caption_{idx:02d}",
            device,
        )
        traces.append(trace)
        for key, value in usage.items():
            usage_totals[key] = usage_totals.get(key, 0) + value

    final_grid_path = out_dir / "feedback_loop_grid.png"
    plot_feedback_loop_grid(args, traces, final_grid_path)
    with (out_dir / "usage_totals.json").open("w") as f:
        json.dump(usage_totals, f, indent=2)
    log_to_wandb_if_configured(args, traces, final_grid_path)
    print(f"Saved feedback-loop grid to {final_grid_path}")
    print(f"Combined token usage totals: {usage_totals}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model", type=str, default="DiT-S/4", choices=list(DiT_models.keys()))
    parser.add_argument("--vae", type=str, default="stabilityai/sdxl-vae")
    parser.add_argument("--dataset-root", type=str, default="/gpfs/scrubbed/sriyash/clevr_dit_dataset")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--caption-seed", type=int, default=0)
    parser.add_argument("--caption-index", type=int, default=0)
    parser.add_argument("--num-captions", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-seed-per-trace", action="store_true", help="Use the same sampling seed at every feedback step to isolate conditioning effects.")
    parser.add_argument("--steps", type=int, default=8, help="Number of feedback/regeneration steps after the initial image.")
    parser.add_argument("--vlm-model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--vlm-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--max-context-len", type=int, default=1024)
    parser.add_argument("--gemini-model", type=str, default=OPENROUTER_GEMINI_MODEL_ID)
    parser.add_argument("--feedback-vlm", choices=["gemini", "qwen-vllm"], default="gemini")
    parser.add_argument("--vllm-base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--vllm-api-key", type=str, default="EMPTY")
    parser.add_argument("--vllm-model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable Qwen thinking in vLLM and extract the final correction.")
    parser.add_argument("--openrouter-api-key-env", type=str, default="OPENROUTER_API_KEY")
    parser.add_argument("--openrouter-retries", type=int, default=2)
    parser.add_argument("--feedback-temperature", type=float, default=0.0)
    parser.add_argument("--max-feedback-tokens", type=int, default=96)
    parser.add_argument("--use-caption", action="store_true", help="Include the rendered caption in the feedback prompt.")
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", type=str, default="results/adaptive_feedback_loop_eval")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-id", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-resume", type=str, default="allow")
    parser.add_argument("--wandb-key", type=str, default="eval/feedback_loop_grid")
    parser.add_argument("--wandb-step", type=int, default=None)
    args = parser.parse_args()
    if args.device_map == "":
        args.device_map = None
    main(args)
