#!/usr/bin/env python3
"""
Generate a CLEVR VLM-feedback dataset from a text-conditioned DiT checkpoint.

Default debug scale:
  50 images * all cached caption records/image * 10 samples/caption * 10 feedbacks/sample
"""
import argparse
import base64
import concurrent.futures
import hashlib
import io
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion import create_diffusion  # noqa: E402
from datasets_clevr import ClevrContextDataset, pad_contexts  # noqa: E402
from models import DiT_models  # noqa: E402

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_GEMINI_MODEL_ID = "google/gemini-3.1-flash-lite"


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


def compact_metadata_text(metadata):
    if metadata is None:
        return None
    objects = metadata.get("objects", [])
    object_lines = []
    for obj in objects:
        object_lines.append(
            f"id {obj.get('id')}: {obj.get('size')} {obj.get('color')} "
            f"{obj.get('material')} {obj.get('shape')} ({obj.get('label')})"
        )
    orders = metadata.get("orders", {})

    def order_text(key):
        labels = []
        for idx in orders.get(key, []):
            match = next((obj for obj in objects if obj.get("id") == idx), None)
            labels.append(match.get("label", str(idx)) if match else str(idx))
        return ", ".join(labels)

    return "\n".join([
        "Objects:",
        *object_lines,
        f"Left-to-right order: {order_text('left_to_right')}",
        f"Front-to-back order: {order_text('front_to_back')}",
    ])


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


def resize_square(image, size):
    image = image.convert("RGB")
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return image.crop((left, top, left + side, top + side)).resize((size, size), Image.Resampling.LANCZOS)


def image_to_data_url(image):
    image_bytes = io.BytesIO()
    image.convert("RGB").save(image_bytes, format="PNG")
    encoded = base64.b64encode(image_bytes.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


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


def build_feedback_prompt(metadata_text=None):
    # Previous prompt version kept for reference:
    # prompt = (
    #     "You are evaluating a text-conditioned diffusion model trained on CLEVR images. "
    #     "The first image is the ground-truth image. The second image is the generated image.\n\n"
    #     f"Caption: {caption}\n\n"
    #     "Write only very short corrective feedback. Do not mention anything that is already correct. "
    #     "Do not praise the image and do not restate the caption. "
    #     "List only the changes needed to make the generated image match the caption and ground truth."
    # )
    metadata_block = ""
    if metadata_text:
        metadata_block = f"Original CLEVR metadata, if useful:\n{metadata_text}\n\n"
    return (
        "You are evaluating a text-conditioned diffusion model trained on CLEVR images. "
        "The first image is the ground-truth image. The second image is the generated image.\n\n"
        f"{metadata_block}"
        "Return exactly one short corrective feedback sentence. Do not use bullets. "
        "Do not mention anything already correct. Do not praise the image. Do not restate any prompt. "
        "Choose the highest-priority needed edit using this priority order: "
        "1) add missing objects or remove extra objects, "
        "2) fix object shape, "
        "3) fix object color, "
        "4) fix object position or depth ordering, "
        "5) fix material/texture, "
        "6) fix background or camera style. "
        "Use an imperative edit, for example: 'Add the missing small red metal cube on the left.'"
    )


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
    for caption_record in caption_records:
        for sample_index in range(args.samples_per_caption):
            seed = args.seed + int(caption_record["metadata_index"]) * 100_000 + sample_index
            gt_path = Path(args.out_dir) / "ground_truth" / (
                f"meta_{int(caption_record['metadata_index']):06d}_{caption_record['template']}.png"
            )
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            if not gt_path.exists():
                caption_record["_dataset"].image_for_row(caption_record["row_idx"]).save(gt_path)
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
    for start in range(0, len(pending), args.generation_batch_size):
        batch = pending[start:start + args.generation_batch_size]
        records = [task["record"] for task in batch]
        seeds = [task["sample_seed"] for task in batch]
        decoded = generator.generate_batch(records, seeds)
        for img_tensor, task in zip(decoded, batch):
            out_path = Path(task["generated_image_path"])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tensor_to_pil(img_tensor).save(out_path)
        done = min(start + len(batch), len(pending))
        print(f"Generated {done}/{len(pending)} missing images")


def make_feedback_tasks(args, generation_tasks, metadata_rows, metadata_by_image_path):
    tasks = []
    for gen_task in generation_tasks:
        metadata = metadata_for_record(gen_task["record"], metadata_rows, metadata_by_image_path)
        metadata_text = compact_metadata_text(metadata) if args.use_caption else None
        for feedback_index in range(args.feedbacks_per_image):
            task = {
                **{k: v for k, v in gen_task.items() if k != "record"},
                "feedback_index": feedback_index,
                "metadata": metadata,
                "metadata_text": metadata_text,
                "metadata_used_for_feedback": bool(args.use_caption),
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


def normalize_chat_url(url):
    url = url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def request_chat_completion(url, api_key, model, content, max_tokens, temperature, retries):
    import requests

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip(), result.get("usage", {}) or {}
        except (requests.RequestException, KeyError, IndexError) as exc:
            if attempt >= retries:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def remote_feedback_one(args, task, url, api_key, model):
    gt = resize_square(Image.open(task["gt_image_path"]), args.image_size)
    gen = resize_square(Image.open(task["generated_image_path"]), args.image_size)
    content = [
        {"type": "text", "text": build_feedback_prompt(task["metadata_text"])},
        {"type": "text", "text": "Ground-truth image:"},
        {"type": "image_url", "image_url": {"url": image_to_data_url(gt)}},
        {"type": "text", "text": "Generated image:"},
        {"type": "image_url", "image_url": {"url": image_to_data_url(gen)}},
    ]
    return request_chat_completion(url, api_key, model, content, args.max_new_tokens, args.vlm_temperature, args.openrouter_retries)


def run_remote_feedback(args, tasks, jsonl_path, failures_path):
    if args.vlm == "gemini":
        url = OPENROUTER_API_URL
        api_key = openrouter_api_key(args)
        model = args.gemini_model
        workers = args.vlm_workers
    else:
        url = normalize_chat_url(args.vllm_base_url)
        api_key = args.vllm_api_key
        model = args.vllm_model
        workers = args.vlm_batch_size

    completed = 0
    usage_totals = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_task = {
            executor.submit(remote_feedback_one, args, task, url, api_key, model): task
            for task in tasks
        }
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                feedback, usage = future.result()
                append_jsonl(jsonl_path, build_output_row(args, task, feedback, usage))
                for key, value in usage.items():
                    if isinstance(value, (int, float)):
                        usage_totals[key] = usage_totals.get(key, 0) + value
            except Exception as exc:
                append_jsonl(failures_path, {**task, "error": repr(exc), "vlm": args.vlm})
            completed += 1
            if completed % args.log_every == 0 or completed == len(tasks):
                print(f"Feedback {completed}/{len(tasks)} complete; usage_totals={usage_totals}")
    return usage_totals


def load_qwen_local(args, device):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.qwen_model, trust_remote_code=True)
    errors = []
    for class_name in (
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen2VLForConditionalGeneration",
    ):
        try:
            module = __import__("transformers", fromlist=[class_name])
            model_cls = getattr(module, class_name)
            kwargs = {"trust_remote_code": True}
            kwargs["torch_dtype"] = "auto" if args.qwen_dtype == "auto" else getattr(torch, args.qwen_dtype)
            if args.qwen_device_map:
                kwargs["device_map"] = args.qwen_device_map
            model = model_cls.from_pretrained(args.qwen_model, **kwargs)
            if not args.qwen_device_map:
                model = model.to(device)
            model.eval()
            return processor, model
        except Exception as exc:
            errors.append(f"{class_name}: {exc}")
    raise RuntimeError("Could not load local Qwen model. Tried:\n" + "\n".join(errors))


def build_qwen_messages(args, task):
    gt = resize_square(Image.open(task["gt_image_path"]), args.image_size)
    gen = resize_square(Image.open(task["generated_image_path"]), args.image_size)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": gt},
                {"type": "image", "image": gen},
                {"type": "text", "text": build_feedback_prompt(task["metadata_text"])},
            ],
        }
    ]


@torch.no_grad()
def qwen_local_feedback_batch(args, processor, model, tasks):
    messages = [build_qwen_messages(args, task) for task in tasks]
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    try:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    except Exception:
        images = []
        for msg in messages:
            images.extend([item["image"] for item in msg[0]["content"] if item["type"] == "image"])
        inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")

    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}
    generated = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.vlm_temperature > 0,
        temperature=args.vlm_temperature if args.vlm_temperature > 0 else None,
    )
    input_len = inputs["input_ids"].shape[1]
    generated = generated[:, input_len:]
    return processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def run_qwen_local_feedback(args, tasks, jsonl_path, failures_path, device):
    processor, model = load_qwen_local(args, device)
    completed = 0
    for start in range(0, len(tasks), args.vlm_batch_size):
        batch = tasks[start:start + args.vlm_batch_size]
        try:
            feedbacks = qwen_local_feedback_batch(args, processor, model, batch)
            for task, feedback in zip(batch, feedbacks):
                append_jsonl(jsonl_path, build_output_row(args, task, feedback.strip(), {}))
        except Exception as exc:
            for task in batch:
                append_jsonl(failures_path, {**task, "error": repr(exc), "vlm": args.vlm})
        completed += len(batch)
        if completed % args.log_every == 0 or completed == len(tasks):
            print(f"Feedback {completed}/{len(tasks)} complete")
    return {}


def preview_grid(args, rows, out_path, max_rows=8):
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
    parser.add_argument("--vlm", choices=["gemini", "qwen-local", "qwen-vllm"], default="gemini")
    parser.add_argument("--vlm-temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--use-caption", action="store_true", help="Include compact CLEVR metadata, not rendered caption text, in VLM prompt.")
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
        if args.vlm in {"gemini", "qwen-vllm"}:
            usage_totals = run_remote_feedback(args, pending_feedback, jsonl_path, failures_path)
        elif args.vlm == "qwen-local":
            usage_totals = run_qwen_local_feedback(args, pending_feedback, jsonl_path, failures_path, device)

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
