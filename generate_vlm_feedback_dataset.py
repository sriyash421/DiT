#!/usr/bin/env python3
"""
Generate a CLEVR VLM-feedback dataset from a text-conditioned DiT checkpoint.

Default debug scale:
  50 images * all cached caption records/image * 10 samples/caption * 10 feedbacks/sample
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers.models import AutoencoderKL
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion import create_diffusion  # noqa: E402
from datasets_clevr import ClevrContextDataset, pad_contexts  # noqa: E402
from feedback_verifiers import (  # noqa: E402
    DEFAULT_GEMINI_MODEL,
    OPENROUTER_API_URL,
    build_feedback_verifier,
    resize_square,
)
from models import DiT_models  # noqa: E402

OPENROUTER_GEMINI_MODEL_ID = DEFAULT_GEMINI_MODEL


def load_jsonl(path):
    if not Path(path).exists():
        return []
    rows = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def load_checkpoint(path, use_ema):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and ("ema" in checkpoint or "model" in checkpoint):
        key = "ema" if use_ema and "ema" in checkpoint else "model"
        return checkpoint[key]
    return checkpoint


def load_metadata_rows(dataset_root):
    rows = load_jsonl(Path(dataset_root).parent / "metadata.jsonl")
    by_image_path = {row.get("image_path"): row for row in rows}
    return rows, by_image_path


def metadata_for_record(record, metadata_rows, metadata_by_image_path):
    metadata_index = record.get("metadata_index")
    if isinstance(metadata_index, int) and 0 <= metadata_index < len(metadata_rows):
        return metadata_rows[metadata_index]
    return metadata_by_image_path.get(record.get("image_path"))


def load_index(dataset_root):
    dataset = ClevrContextDataset(dataset_root, transform=None, split=None)
    records = []
    for idx in range(len(dataset)):
        record = dataset.record_for_index(idx)
        record["_dataset"] = dataset
        records.append(record)
    return {"records": records, "context_dim": dataset.context_dim}


def select_caption_records(index, split, max_images):
    records = [
        record for record in index["records"]
        if record["split"] == split
    ]
    if max_images is None:
        return records

    selected = []
    seen_metadata = set()
    for record in records:
        metadata_index = record["metadata_index"]
        if metadata_index not in seen_metadata:
            if len(seen_metadata) >= max_images:
                break
            seen_metadata.add(metadata_index)
        selected.append(record)
    return selected


def load_context_batch(dataset_root, records, device, dtype):
    tokens = []
    for record in records:
        tokens.append(record["_dataset"].context_for_row(record["row_idx"]).to(dtype=dtype))
    context_tokens, context_mask = pad_contexts(tokens)
    return {
        "context_tokens": context_tokens.to(device),
        "context_mask": context_mask.to(device),
    }


def tensor_to_pil(x):
    x = x.detach().float().cpu().clamp(-1, 1)
    x = (x + 1) / 2
    x = (x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(x)


def save_image_tensor(img_tensor, out_path):
    tensor_to_pil(img_tensor).save(out_path)


def drain_save_futures(save_futures, wait_for_one=False):
    if not save_futures:
        return
    if wait_for_one:
        done, pending = concurrent.futures.wait(
            save_futures,
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
    else:
        done, pending = concurrent.futures.wait(save_futures)
    for future in done:
        future.result()
    save_futures.clear()
    save_futures.update(pending)


def save_tensor_grid(images, path, nrow):
    images = images.detach().float().cpu().clamp(-1, 1)
    images = (images + 1) / 2
    n = images.shape[0]
    nrow = max(1, min(nrow, n))
    ncol = (n + nrow - 1) // nrow
    h, w = images.shape[-2:]
    canvas = Image.new("RGB", (nrow * w, ncol * h), (255, 255, 255))
    for idx in range(n):
        pil = tensor_to_pil(images[idx] * 2 - 1)
        x = (idx % nrow) * w
        y = (idx // nrow) * h
        canvas.paste(pil, (x, y))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def tuple_key(task):
    return f"{task['metadata_index']}|{task['template']}|{task['sample_index']}|{task['feedback_index']}"


def tuple_id(task):
    raw = tuple_key(task)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"fb_{digest}"


class DiTGenerator:
    def __init__(self, args, index, device):
        self.args = args
        self.index = index
        self.device = device
        self.latent_size = args.image_size // 8
        self.model = DiT_models[args.model](
            input_size=self.latent_size,
            num_classes=args.num_classes,
            text_conditioning=True,
            context_dim=index["context_dim"],
        ).to(device)
        self.model.load_state_dict(load_checkpoint(args.ckpt, args.ema), strict=True)
        self.model.eval()
        self.model_dtype = next(self.model.parameters()).dtype
        self.vae = AutoencoderKL.from_pretrained(args.vae).to(device)
        self.vae.eval()
        self.vae_scaling_factor = self.vae.config.scaling_factor
        self.diffusion = create_diffusion(str(args.num_sampling_steps))

    @torch.no_grad()
    def generate_batch(self, records, seeds):
        context_batch = load_context_batch(self.args.dataset_root, records, self.device, self.model_dtype)
        latents = []
        for seed in seeds:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
            latents.append(torch.randn(1, 4, self.latent_size, self.latent_size, device=self.device, generator=generator))
        z = torch.cat(latents, dim=0)
        if self.args.cfg_scale <= 1:
            model_kwargs = context_batch
            forward_fn = self.model.forward
        else:
            z = torch.cat([z, z], dim=0)
            model_kwargs = {
                "context_tokens": context_batch["context_tokens"].repeat(2, 1, 1),
                "context_mask": context_batch["context_mask"].repeat(2, 1),
                "cfg_scale": self.args.cfg_scale,
            }
            forward_fn = self.model.forward_with_cfg

        sample_loop = self.diffusion.ddim_sample_loop if self.args.sampler == "ddim" else self.diffusion.p_sample_loop
        samples = sample_loop(
            forward_fn,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=self.device,
            **({"eta": self.args.ddim_eta} if self.args.sampler == "ddim" else {}),
        )
        if self.args.cfg_scale > 1:
            samples, _ = samples.chunk(2, dim=0)
        return self.vae.decode(samples / self.vae_scaling_factor).sample


def generated_image_path(out_dir, task):
    return Path(out_dir) / "generated" / (
        f"meta_{task['metadata_index']:06d}_{task['template']}_sample_{task['sample_index']:03d}.png"
    )


def make_generation_tasks(args, caption_records):
    tasks = []
    gt_root = Path(args.out_dir) / "ground_truth"
    gt_root.mkdir(parents=True, exist_ok=True)
    for caption_record in tqdm(caption_records, desc="make generation tasks", unit="caption"):
        gt_path = gt_root / (
            f"meta_{int(caption_record['metadata_index']):06d}_{caption_record['template']}.png"
        )
        if not gt_path.exists():
            caption_record["_dataset"].image_for_row(caption_record["row_idx"]).save(gt_path)
        for sample_index in range(args.samples_per_caption):
            seed = args.seed + int(caption_record["metadata_index"]) * 100_000 + sample_index
            tasks.append({
                "record": caption_record,
                "metadata_index": int(caption_record["metadata_index"]),
                "template": caption_record["template"],
                "caption": caption_record["caption"],
                "source_image_path": caption_record["image_path"],
                "gt_image_path": str(gt_path),
                "sample_index": sample_index,
                "sample_seed": seed,
            })
    return tasks


def generate_missing_images(args, index, generation_tasks, device):
    generated_root = Path(args.out_dir) / "generated"
    generated_root.mkdir(parents=True, exist_ok=True)
    pending = []
    for task in generation_tasks:
        out_path = generated_image_path(args.out_dir, task)
        task["generated_image_path"] = str(out_path)
        if args.overwrite or not out_path.exists():
            pending.append(task)
    if not pending:
        print("All generated images already exist; skipping DiT generation.")
        return

    generator = DiTGenerator(args, index, device)
    save_futures = set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.image_save_workers) as executor:
        for start in tqdm(range(0, len(pending), args.generation_batch_size), desc="generate images", unit="batch"):
            batch = pending[start:start + args.generation_batch_size]
            records = [task["record"] for task in batch]
            seeds = [task["sample_seed"] for task in batch]
            decoded = generator.generate_batch(records, seeds)
            decoded_cpu = decoded.detach().cpu()
            del decoded
            for img_tensor, task in zip(decoded_cpu, batch):
                out_path = Path(task["generated_image_path"])
                save_futures.add(executor.submit(save_image_tensor, img_tensor, out_path))
                while len(save_futures) >= args.image_save_queue_size:
                    drain_save_futures(save_futures, wait_for_one=True)
        drain_save_futures(save_futures)


def make_feedback_tasks(args, generation_tasks, metadata_rows, metadata_by_image_path):
    tasks = []
    for gen_task in generation_tasks:
        metadata = metadata_for_record(gen_task["record"], metadata_rows, metadata_by_image_path)
        caption_text = gen_task["caption"] if args.use_caption else None
        for feedback_index in range(args.feedbacks_per_image):
            task = {
                **{k: v for k, v in gen_task.items() if k != "record"},
                "feedback_index": feedback_index,
                "metadata": metadata,
                "caption_text": caption_text,
                "metadata_text": None,
                "caption_used_for_feedback": bool(args.use_caption),
                "metadata_used_for_feedback": False,
            }
            tasks.append(task)
    return tasks


def load_completed_keys(jsonl_path):
    return {tuple_key(row) for row in load_jsonl(jsonl_path)}


def build_output_row(args, task, feedback, token_usage):
    return {
        "tuple_id": tuple_id(task),
        "metadata_index": task["metadata_index"],
        "template": task["template"],
        "caption": task["caption"],
        "metadata": task["metadata"],
        "caption_used_for_feedback": task.get("caption_used_for_feedback", False),
        "metadata_used_for_feedback": task["metadata_used_for_feedback"],
        "gt_image_path": task["gt_image_path"],
        "generated_image_path": task["generated_image_path"],
        "source_image_path": task["source_image_path"],
        "sample_index": task["sample_index"],
        "feedback_index": task["feedback_index"],
        "sample_seed": task["sample_seed"],
        "feedback": feedback,
        "vlm": args.vlm,
        "vlm_model": selected_vlm_model(args),
        "vlm_temperature": args.vlm_temperature,
        "token_usage": token_usage,
        "ckpt": args.ckpt,
        "model": args.model,
        "vae": args.vae,
        "cfg_scale": args.cfg_scale,
        "sampler": args.sampler,
        "num_sampling_steps": args.num_sampling_steps,
    }


def selected_vlm_model(args):
    if args.vlm == "gemini":
        return args.gemini_model
    if args.vlm == "qwen-local":
        return args.qwen_model
    if args.vlm == "qwen-vllm":
        return args.vllm_model
    raise ValueError(args.vlm)


def openrouter_api_key(args):
    key = os.getenv(args.openrouter_api_key_env)
    if not key:
        raise ValueError(f"Set {args.openrouter_api_key_env} for OpenRouter access")
    return key


def build_dataset_feedback_verifier(args, device):
    if args.vlm == "gemini":
        return build_feedback_verifier(
            backend="openai-chat",
            model=args.gemini_model,
            api_url=OPENROUTER_API_URL,
            api_key=openrouter_api_key(args),
            temperature=args.vlm_temperature,
            max_tokens=args.max_new_tokens,
            retries=args.openrouter_retries,
            workers=args.vlm_workers,
            include_caption=args.use_caption,
            include_metadata=False,
            image_size=args.image_size,
        )
    if args.vlm == "qwen-vllm":
        return build_feedback_verifier(
            backend="qwen-vllm",
            model=args.vllm_model,
            api_url=args.vllm_base_url,
            api_key=args.vllm_api_key,
            temperature=args.vlm_temperature,
            max_tokens=args.max_new_tokens,
            retries=args.openrouter_retries,
            workers=args.vlm_batch_size,
            enable_thinking=args.enable_thinking,
            include_caption=args.use_caption,
            include_metadata=False,
            image_size=args.image_size,
        )
    if args.vlm == "qwen-local":
        return build_feedback_verifier(
            backend="qwen-local",
            model=args.qwen_model,
            device=device,
            qwen_dtype=args.qwen_dtype,
            device_map=args.qwen_device_map,
            temperature=args.vlm_temperature,
            max_tokens=args.max_new_tokens,
            workers=args.vlm_batch_size,
            enable_thinking=args.enable_thinking,
            include_caption=args.use_caption,
            include_metadata=False,
            image_size=args.image_size,
        )
    raise ValueError(args.vlm)


def task_feedback_images(args, task):
    return (
        resize_square(Image.open(task["gt_image_path"]), args.image_size),
        resize_square(Image.open(task["generated_image_path"]), args.image_size),
    )


def run_feedback(args, tasks, jsonl_path, failures_path, device):
    verifier = build_dataset_feedback_verifier(args, device)
    usage_totals = {}
    for start in tqdm(range(0, len(tasks), args.vlm_batch_size), desc="vlm feedback", unit="batch"):
        batch = tasks[start:start + args.vlm_batch_size]
        try:
            image_pairs = [task_feedback_images(args, task) for task in batch]
            results = verifier.verify_batch(
                [task.get("caption_text") or task["caption"] for task in batch],
                [task.get("metadata") for task in batch],
                [pair[0] for pair in image_pairs],
                [pair[1] for pair in image_pairs],
            )
            for task, result in zip(batch, results):
                if result.ok:
                    usage = result.token_usage or {}
                    append_jsonl(jsonl_path, build_output_row(args, task, result.feedback, usage))
                    for key, value in usage.items():
                        if isinstance(value, (int, float)):
                            usage_totals[key] = usage_totals.get(key, 0) + value
                else:
                    append_jsonl(failures_path, {**task, "error": result.error, "vlm": args.vlm})
        except Exception as exc:
            for task in batch:
                append_jsonl(failures_path, {**task, "error": repr(exc), "vlm": args.vlm})
    return usage_totals


def preview_grid(args, rows, out_path, max_rows=20):
    if not rows:
        return
    rows = rows[:max_rows]
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 4.2 * len(rows)), dpi=args.preview_dpi)
    if len(rows) == 1:
        axes = np.expand_dims(axes, axis=0)
    for ax, title in zip(axes[0], ["Ground Truth", "Generated", "Feedback"]):
        ax.set_title(title, fontsize=12, fontweight="bold")
    for i, row in enumerate(rows):
        gt = resize_square(Image.open(row["gt_image_path"]), args.image_size)
        gen = resize_square(Image.open(row["generated_image_path"]), args.image_size)
        axes[i, 0].imshow(gt)
        axes[i, 1].imshow(gen)
        for col in (0, 1):
            axes[i, col].set_xticks([])
            axes[i, col].set_yticks([])
        axes[i, 2].axis("off")
        axes[i, 2].text(0, 1, row["feedback"], va="top", ha="left", wrap=True, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model", type=str, default="DiT-S/4", choices=list(DiT_models.keys()))
    parser.add_argument("--vae", type=str, default="stabilityai/sdxl-vae")
    parser.add_argument("--dataset-root", type=str, default="data/clevr_50_train")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--samples-per-caption", type=int, default=10)
    parser.add_argument("--feedbacks-per-image", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--image-save-workers", type=int, default=8)
    parser.add_argument("--image-save-queue-size", type=int, default=1024)
    parser.add_argument("--vlm", choices=["gemini", "qwen-local", "qwen-vllm"], default="gemini")
    parser.add_argument("--vlm-temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--use-caption", action="store_true", help="Include the rendered caption in the VLM prompt.")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable Qwen thinking in vLLM and extract the final correction.")
    parser.add_argument("--vlm-workers", type=int, default=8)
    parser.add_argument("--vlm-batch-size", type=int, default=8)
    parser.add_argument("--gemini-model", type=str, default=OPENROUTER_GEMINI_MODEL_ID)
    parser.add_argument("--openrouter-api-key-env", type=str, default="OPENROUTER_API_KEY")
    parser.add_argument("--openrouter-retries", type=int, default=2)
    parser.add_argument("--qwen-model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--qwen-dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--qwen-device-map", type=str, default="auto")
    parser.add_argument("--vllm-base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--vllm-api-key", type=str, default="EMPTY")
    parser.add_argument("--vllm-model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--out-dir", type=str, default="results/vlm_feedback_dataset")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--preview-dpi", type=int, default=150)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.image_save_workers <= 0:
        raise ValueError("--image-save-workers must be positive")
    if args.image_save_queue_size <= 0:
        raise ValueError("--image-save-queue-size must be positive")
    if args.qwen_device_map == "":
        args.qwen_device_map = None
    return args


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "feedback_dataset.jsonl"
    failures_path = out_dir / "failures.jsonl"
    manifest_path = out_dir / "manifest.json"
    progress_path = out_dir / "progress.json"

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    index = load_index(args.dataset_root)
    metadata_rows, metadata_by_image_path = load_metadata_rows(args.dataset_root)
    caption_records = select_caption_records(
        index,
        args.split,
        args.max_images,
    )
    if not caption_records:
        raise RuntimeError("No caption records selected")

    generation_tasks = make_generation_tasks(args, caption_records)
    manifest = {
        "args": vars(args),
        "selected_caption_records": len(caption_records),
        "generation_tasks": len(generation_tasks),
        "expected_feedback_rows": len(generation_tasks) * args.feedbacks_per_image,
        "vlm_model": selected_vlm_model(args),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(manifest_path, manifest)

    generate_missing_images(args, index, generation_tasks, device)

    feedback_tasks = make_feedback_tasks(args, generation_tasks, metadata_rows, metadata_by_image_path)
    completed_keys = load_completed_keys(jsonl_path)
    pending_feedback = [task for task in feedback_tasks if tuple_key(task) not in completed_keys]
    print(f"Selected captions: {len(caption_records)}")
    print(f"Generated images: {len(generation_tasks)}")
    print(f"Feedback rows complete: {len(completed_keys)}")
    print(f"Feedback rows pending: {len(pending_feedback)}")

    usage_totals = {}
    if pending_feedback:
        usage_totals = run_feedback(args, pending_feedback, jsonl_path, failures_path, device)

    all_rows = load_jsonl(jsonl_path)
    manifest.update({
        "completed_feedback_rows": len(all_rows),
        "failed_feedback_rows": len(load_jsonl(failures_path)),
        "token_usage_totals_last_run": usage_totals,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    write_json(manifest_path, manifest)
    write_json(progress_path, {
        "completed_feedback_rows": len(all_rows),
        "failed_feedback_rows": len(load_jsonl(failures_path)),
        "expected_feedback_rows": len(feedback_tasks),
    })
    preview_grid(args, all_rows, out_dir / "preview_grid.png")
    print(f"Saved dataset to {jsonl_path}")
    print(f"Saved manifest to {manifest_path}")
    print(f"Saved preview to {out_dir / 'preview_grid.png'}")


if __name__ == "__main__":
    main()
