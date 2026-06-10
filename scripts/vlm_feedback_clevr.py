#!/usr/bin/env python3
"""
Generate CLEVR samples from a text-conditioned DiT checkpoint and ask a VLM for
caption-grounded feedback comparing the generated image against ground truth.

Example:
  python scripts/vlm_feedback_clevr.py \
    --ckpt /gpfs/scrubbed/sriyash/DiT-clevr-text-final/000-DiT-S-4-text/checkpoints/0100000-ema.pt \
    --model DiT-S/4 \
    --vae stabilityai/sdxl-vae \
    --dataset-root /gpfs/scrubbed/sriyash/clevr_dit_dataset \
    --vlm-model Qwen/Qwen2.5-VL-7B-Instruct \
    --num-examples 4 \
    --cfg-scale 1.0 \
    --num-sampling-steps 50 \
    --out-dir results/vlm_feedback/scratch_text_s4
"""
import argparse
import base64
import io
import json
import os
import random
import sys
import textwrap
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers.models import AutoencoderKL
from PIL import Image
from torchvision.utils import save_image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion import create_diffusion  # noqa: E402
from models import DiT_models  # noqa: E402

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_GEMINI_MODEL_ID = "google/gemini-3.1-flash-lite"
HOSTED_QWEN70B_MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct"
HOSTED_INTERN78B_MODEL_ID = "OpenGVLab/InternVL3-78B"


def load_checkpoint(path, use_ema):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and ("ema" in checkpoint or "model" in checkpoint):
        key = "ema" if use_ema and "ema" in checkpoint else "model"
        return checkpoint[key]
    return checkpoint


def select_records(dataset_root, split, template, num_examples, seed):
    index_path = Path(dataset_root) / "text_embeddings" / "index.json"
    with index_path.open() as f:
        index = json.load(f)
    records = [
        record for record in index["records"]
        if record["split"] == split and record["template"] == template
    ]
    if len(records) < num_examples:
        raise RuntimeError(f"Only found {len(records)} records for split={split}, template={template}")
    rng = random.Random(seed)
    return rng.sample(records, num_examples), index


def load_metadata_rows(dataset_root):
    metadata_path = Path(dataset_root) / "metadata.jsonl"
    rows = []
    with metadata_path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    by_image_path = {row.get("image_path"): row for row in rows}
    return rows, by_image_path


def metadata_for_record(record, metadata_rows, metadata_by_image_path):
    metadata_index = record.get("metadata_index")
    if isinstance(metadata_index, int) and 0 <= metadata_index < len(metadata_rows):
        row = metadata_rows[metadata_index]
        if row.get("image_path") == record.get("image_path"):
            return row
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
        ids = orders.get(key, [])
        labels = []
        for idx in ids:
            match = next((obj for obj in objects if obj.get("id") == idx), None)
            labels.append(match.get("label", str(idx)) if match else str(idx))
        return ", ".join(labels)

    return "\n".join([
        "Objects:",
        *object_lines,
        f"Left-to-right order: {order_text('left_to_right')}",
        f"Front-to-back order: {order_text('front_to_back')}",
    ])


def load_text_batch(dataset_root, records, device, dtype):
    root = Path(dataset_root) / "text_embeddings"
    shard_cache = {}
    tokens = []
    masks = []
    pooled = []
    for record in records:
        shard_name = record["shard"]
        if shard_name not in shard_cache:
            shard_cache[shard_name] = torch.load(root / shard_name, map_location="cpu", weights_only=False)
        shard = shard_cache[shard_name]
        offset = record["offset"]
        tokens.append(shard["text_tokens"][offset].to(dtype=dtype))
        masks.append(shard["text_mask"][offset].bool())
        pooled.append(shard["text_pooled"][offset].to(dtype=dtype))
    return {
        "text_tokens": torch.stack(tokens).to(device),
        "text_mask": torch.stack(masks).to(device),
        "text_pooled": torch.stack(pooled).to(device),
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


@torch.no_grad()
def generate_dit_images(args, records, index, device):
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        text_conditioning=True,
        text_embed_dim=index["embedding_dim"],
        max_text_len=index["max_length"],
    ).to(device)
    model.load_state_dict(load_checkpoint(args.ckpt, args.ema), strict=True)
    model.eval()
    model_dtype = next(model.parameters()).dtype

    text_batch = load_text_batch(args.dataset_root, records, device, model_dtype)
    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    vae.eval()
    vae_scaling_factor = vae.config.scaling_factor
    diffusion = create_diffusion(str(args.num_sampling_steps))

    torch.manual_seed(args.seed)
    z = torch.randn(len(records), 4, latent_size, latent_size, device=device)
    if args.cfg_scale <= 1:
        model_kwargs = text_batch
        forward_fn = model.forward
    else:
        z = torch.cat([z, z], dim=0)
        model_kwargs = {
            "text_tokens": text_batch["text_tokens"].repeat(2, 1, 1),
            "text_mask": text_batch["text_mask"].repeat(2, 1),
            "text_pooled": text_batch["text_pooled"].repeat(2, 1),
            "cfg_scale": args.cfg_scale,
        }
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
    decoded = vae.decode(samples / vae_scaling_factor).sample
    return decoded


def load_vlm_model(model_name, device, dtype, device_map):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
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
            if dtype != "auto":
                kwargs["torch_dtype"] = getattr(torch, dtype)
            else:
                kwargs["torch_dtype"] = "auto"
            if device_map:
                kwargs["device_map"] = device_map
            model = model_cls.from_pretrained(model_name, **kwargs)
            if not device_map:
                model = model.to(device)
            model.eval()
            return processor, model
        except Exception as exc:
            errors.append(f"{class_name}: {exc}")
    raise RuntimeError("Could not load VLM model. Tried:\n" + "\n".join(errors))


def build_feedback_prompt(metadata_text=None):
    # Previous prompt version kept for reference:
    # prompt = (
    #     "You are evaluating a text-conditioned diffusion model trained on CLEVR images. "
    #     "The first image is the ground-truth image. The second image is the generated image.\n\n"
    #     f"Caption: {caption}\n\n"
    #     "Write only very short corrective feedback. Do not mention anything that is already correct. "
    #     "Do not praise the image and do not restate the caption. "
    #     "List only the changes needed to make the generated image match the caption and ground truth: "
    #     "missing or extra objects, wrong shape/color/material/size, wrong left-right order, "
    #     "wrong front-back depth, or wrong background/camera style. "
    #     "Use at most 3 short bullet points. Each bullet must be an imperative edit."
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
        "1) add missing objects, "
        "2) fix object shape, "
        "3) fix object color, "
        "4) fix object position or depth ordering, "
        "5) fix material/texture, "
        "6) fix background or camera style. "
        "Use an imperative edit, for example: 'Add the missing small red metal cube on the left.'"
    )


def build_qwen_messages(gt_image, gen_image, metadata_text=None):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": gt_image},
                {"type": "image", "image": gen_image},
                {"type": "text", "text": build_feedback_prompt(metadata_text)},
            ],
        }
    ]


def pil_image_to_data_url(image):
    image_bytes = io.BytesIO()
    image.convert("RGB").save(image_bytes, format="PNG")
    encoded = base64.b64encode(image_bytes.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def load_openrouter_api_key(api_key_env):
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"Set {api_key_env} for OpenRouter access")
    return api_key


def gemini_feedback(api_key, model_name, metadata_text, gt_image, gen_image, max_new_tokens, retries):
    import requests

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_feedback_prompt(metadata_text)},
                    {"type": "text", "text": "Ground-truth image:"},
                    {"type": "image_url", "image_url": {"url": pil_image_to_data_url(gt_image)}},
                    {"type": "text", "text": "Generated image:"},
                    {"type": "image_url", "image_url": {"url": pil_image_to_data_url(gen_image)}},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": max_new_tokens,
    }
    for attempt in range(retries + 1):
        try:
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            feedback = result["choices"][0]["message"]["content"].strip()
            usage = result.get("usage", {}) or {}
            return feedback, usage
        except (requests.RequestException, KeyError, IndexError) as exc:
            if attempt >= retries:
                raise RuntimeError(f"OpenRouter Gemini request failed: {exc}") from exc
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def hosted_model_id(vlm_name):
    if vlm_name == "qwen70b":
        return HOSTED_QWEN70B_MODEL_ID
    if vlm_name == "intern78b":
        return HOSTED_INTERN78B_MODEL_ID
    raise ValueError(f"No hosted model preset for {vlm_name}")


def normalize_vlm_server_url(url):
    url = url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def hosted_vlm_feedback(server_url, api_key, model_name, metadata_text, gt_image, gen_image, max_new_tokens, retries):
    import requests

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_feedback_prompt(metadata_text)},
                    {"type": "text", "text": "Ground-truth image:"},
                    {"type": "image_url", "image_url": {"url": pil_image_to_data_url(gt_image)}},
                    {"type": "text", "text": "Generated image:"},
                    {"type": "image_url", "image_url": {"url": pil_image_to_data_url(gen_image)}},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": max_new_tokens,
    }
    endpoint = normalize_vlm_server_url(server_url)
    for attempt in range(retries + 1):
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            feedback = result["choices"][0]["message"]["content"].strip()
            usage = result.get("usage", {}) or {}
            return feedback, usage
        except (requests.RequestException, KeyError, IndexError) as exc:
            if attempt >= retries:
                raise RuntimeError(f"Hosted VLM request failed at {endpoint}: {exc}") from exc
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


@torch.no_grad()
def vlm_feedback(processor, model, model_name, gt_image, gen_image, max_new_tokens, metadata_text=None):
    messages = build_qwen_messages(gt_image, gen_image, metadata_text)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    except Exception:
        inputs = processor(
            text=[text],
            images=[gt_image, gen_image],
            padding=True,
            return_tensors="pt",
        )

    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    generated = generated[:, input_len:]
    feedback = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return feedback.strip()


def wrap_text(text, width):
    return "\n".join(textwrap.wrap(text, width=width))


def plot_report(rows, out_path, font_size, dpi):
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(15, max(4.6, 4.6 * n)), dpi=dpi)
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for ax, title in zip(axes[0], ["Ground Truth", "Generated", "VLM Feedback"]):
        ax.set_title(title, fontsize=font_size, fontweight="bold", pad=10)

    for row_idx, row in enumerate(rows):
        axes[row_idx, 0].imshow(row["gt"])
        axes[row_idx, 1].imshow(row["generated"])
        for col in (0, 1):
            axes[row_idx, col].set_xticks([])
            axes[row_idx, col].set_yticks([])
            for spine in axes[row_idx, col].spines.values():
                spine.set_linewidth(2)
                spine.set_color("black")

        text_ax = axes[row_idx, 2]
        text_ax.axis("off")
        text_ax.text(
            0.0,
            0.98,
            "Caption",
            fontsize=font_size,
            fontweight="bold",
            va="top",
            ha="left",
            transform=text_ax.transAxes,
        )
        text_ax.text(
            0.0,
            0.88,
            wrap_text(row["caption"], 58),
            fontsize=font_size,
            va="top",
            ha="left",
            transform=text_ax.transAxes,
        )
        text_ax.text(
            0.0,
            0.42,
            "Feedback",
            fontsize=font_size,
            fontweight="bold",
            va="top",
            ha="left",
            transform=text_ax.transAxes,
        )
        text_ax.text(
            0.0,
            0.32,
            wrap_text(row["feedback"], 58),
            fontsize=font_size,
            va="top",
            ha="left",
            transform=text_ax.transAxes,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f4f4f4", "edgecolor": "#cccccc"},
        )
        axes[row_idx, 0].set_ylabel(f"sample {row_idx:02d}", fontsize=font_size, fontweight="bold")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_jsonl(rows, path):
    with Path(path).open("w") as f:
        for row in rows:
            serializable = {k: v for k, v in row.items() if k not in {"gt", "generated"}}
            f.write(json.dumps(serializable) + "\n")


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    records, index = select_records(args.dataset_root, args.split, args.template, args.num_examples, args.caption_seed)
    metadata_rows, metadata_by_image_path = load_metadata_rows(args.dataset_root)
    decoded = generate_dit_images(args, records, index, device)
    generated_pils = [tensor_to_pil(decoded[i]) for i in range(decoded.shape[0])]

    save_image(decoded, out_dir / "generated_grid.png", nrow=args.num_examples, normalize=True, value_range=(-1, 1))

    processor = vlm = None
    openrouter_api_key = None
    hosted_api_key = None
    if args.vlm == "qwen":
        processor, vlm = load_vlm_model(args.vlm_model, device, args.vlm_dtype, args.vlm_device_map)
    elif args.vlm == "gemini":
        openrouter_api_key = load_openrouter_api_key(args.openrouter_api_key_env)
    elif args.vlm in {"qwen70b", "intern78b"}:
        if not args.vlm_server_url:
            raise ValueError(f"--vlm {args.vlm} requires --vlm-server-url, e.g. http://g001:8000/v1")
        hosted_api_key = os.getenv(args.vlm_server_api_key_env, args.vlm_server_api_key)
    rows = []
    for idx, (record, gen_image) in enumerate(zip(records, generated_pils)):
        gt = resize_square(Image.open(Path(args.dataset_root) / record["image_path"]), args.image_size)
        gen_image = gen_image.resize((args.image_size, args.image_size), Image.Resampling.LANCZOS)
        gt.save(out_dir / f"sample_{idx:02d}_gt.png")
        gen_image.save(out_dir / f"sample_{idx:02d}_generated.png")
        metadata = metadata_for_record(record, metadata_rows, metadata_by_image_path)
        metadata_text = compact_metadata_text(metadata) if args.use_caption else None
        if args.vlm == "qwen":
            feedback = vlm_feedback(processor, vlm, args.vlm_model, gt, gen_image, args.max_new_tokens, metadata_text)
            token_usage = {}
        elif args.vlm == "gemini":
            feedback, token_usage = gemini_feedback(
                openrouter_api_key,
                args.gemini_model,
                metadata_text,
                gt,
                gen_image,
                args.max_new_tokens,
                args.openrouter_retries,
            )
        else:
            feedback, token_usage = hosted_vlm_feedback(
                args.vlm_server_url,
                hosted_api_key,
                hosted_model_id(args.vlm),
                metadata_text,
                gt,
                gen_image,
                args.max_new_tokens,
                args.vlm_server_retries,
            )
        rows.append({
            "image_path": record["image_path"],
            "caption": record["caption"],
            "metadata": metadata,
            "metadata_used_for_feedback": bool(args.use_caption),
            "vlm": args.vlm,
            "vlm_model": args.vlm_model if args.vlm == "qwen" else (args.gemini_model if args.vlm == "gemini" else hosted_model_id(args.vlm)),
            "vlm_server_url": args.vlm_server_url if args.vlm in {"qwen70b", "intern78b"} else None,
            "token_usage": token_usage,
            "feedback": feedback,
            "gt": gt,
            "generated": gen_image,
        })
        usage_str = ""
        if token_usage:
            usage_str = f" usage={token_usage}"
        print(f"[{idx + 1}/{len(records)}] {feedback}{usage_str}\n")

    if args.vlm in {"gemini", "qwen70b", "intern78b"}:
        usage_totals = {}
        for row in rows:
            for key, value in row.get("token_usage", {}).items():
                if isinstance(value, (int, float)):
                    usage_totals[key] = usage_totals.get(key, 0) + value
        print(f"Remote VLM token usage total: {usage_totals}")

    plot_report(rows, out_dir / "vlm_feedback_report.png", args.font_size, args.dpi)
    save_jsonl(rows, out_dir / "vlm_feedback.jsonl")
    print(f"Saved report to {out_dir / 'vlm_feedback_report.png'}")
    print(f"Saved feedback to {out_dir / 'vlm_feedback.jsonl'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model", type=str, default="DiT-S/4", choices=list(DiT_models.keys()))
    parser.add_argument("--vae", type=str, default="stabilityai/sdxl-vae")
    parser.add_argument("--dataset-root", type=str, default="/gpfs/scrubbed/sriyash/clevr_dit_dataset")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--template", type=str, default="chain")
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument("--caption-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vlm", choices=["qwen", "gemini", "qwen70b", "intern78b"], default="qwen")
    parser.add_argument("--use-caption", action="store_true", help="Include original CLEVR metadata, not the rendered caption, in the VLM feedback prompt.")
    parser.add_argument("--vlm-model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--gemini-model", type=str, default=OPENROUTER_GEMINI_MODEL_ID)
    parser.add_argument("--openrouter-api-key-env", type=str, default="OPENROUTER_API_KEY")
    parser.add_argument("--openrouter-retries", type=int, default=2)
    parser.add_argument("--vlm-server-url", type=str, default=None, help="OpenAI-compatible vLLM endpoint on another node, e.g. http://g001:8000/v1")
    parser.add_argument("--vlm-server-api-key", type=str, default="EMPTY")
    parser.add_argument("--vlm-server-api-key-env", type=str, default="VLLM_API_KEY")
    parser.add_argument("--vlm-server-retries", type=int, default=2)
    parser.add_argument("--vlm-dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--vlm-device-map", type=str, default="auto", help="Use 'auto' for multi-GPU/accelerate, or empty string to place on the selected device.")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--font-size", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--out-dir", type=str, default="results/vlm_feedback")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.vlm_device_map == "":
        args.vlm_device_map = None
    main(args)
