"""Generate a CLEVR VLM-feedback dataset (feedback_dataset.jsonl) from a DiT checkpoint.

For each selected caption: sample images from the checkpoint, ask a VLM verifier for feedback
against the ground truth, and append rows consumed by datasets/clevr/convert_to_zarr.py --mode feedback.
"""
import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.utils import load_checkpoint, load_jsonl, tensor_to_pil, write_json
from datasets.clevr.dataset import ClevrContextDataset, pad_contexts
from diffusion import create_diffusion
from models.qwen_dit import DiT_models
from verifiers import build_verifier
from verifiers.base import resize_square
from verifiers.gemini import DEFAULT_GEMINI_MODEL


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def load_index(dataset_root):
    dataset = ClevrContextDataset(dataset_root, transform=None, split=None)
    records = []
    for idx in range(len(dataset)):
        record = dataset.record_for_index(idx)
        record["_dataset"] = dataset
        records.append(record)
    return {"records": records, "context_dim": dataset.context_dim}


def select_caption_records(index, split, max_images):
    records = [record for record in index["records"] if record["split"] == split]
    if max_images is not None:
        records = records[:max_images]
    return records


def tuple_key(task):
    return f"{task['metadata_index']}|{task['sample_index']}|{task['feedback_index']}"


def tuple_id(task):
    digest = hashlib.sha1(tuple_key(task).encode("utf-8")).hexdigest()[:12]
    return f"fb_{digest}"


class DiTGenerator:
    """Samples DiT images from cached context tokens with one seed per sample."""

    def __init__(self, args, index, device):
        self.args = args
        self.device = device
        self.latent_size = args.image_size // 8
        self.model = DiT_models[args.model](
            input_size=self.latent_size,
            context_dim=index["context_dim"],
        ).to(device)
        self.model.load_state_dict(load_checkpoint(args.ckpt, use_ema=args.ema), strict=True)
        self.model.eval()
        self.model_dtype = next(self.model.parameters()).dtype
        from diffusers.models import AutoencoderKL

        self.vae = AutoencoderKL.from_pretrained(args.vae).to(device)
        self.vae.eval()
        self.diffusion = create_diffusion(str(args.num_sampling_steps))

    @torch.no_grad()
    def generate_batch(self, records, seeds):
        tokens = [record["_dataset"].context_for_row(record["row_idx"]).to(dtype=self.model_dtype) for record in records]
        context_tokens, context_mask = pad_contexts(tokens)
        context_tokens = context_tokens.to(self.device)
        context_mask = context_mask.to(self.device)
        latents = []
        for seed in seeds:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
            latents.append(torch.randn(1, 4, self.latent_size, self.latent_size, device=self.device, generator=generator))
        z = torch.cat(latents, dim=0)
        if self.args.cfg_scale <= 1:
            model_kwargs = {"context_tokens": context_tokens, "context_mask": context_mask}
            forward_fn = self.model.forward
        else:
            z = torch.cat([z, z], dim=0)
            model_kwargs = {
                "context_tokens": context_tokens.repeat(2, 1, 1),
                "context_mask": context_mask.repeat(2, 1),
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
        return self.vae.decode(samples / self.vae.config.scaling_factor).sample


def make_generation_tasks(args, caption_records):
    tasks = []
    gt_root = Path(args.out_dir) / "ground_truth"
    gt_root.mkdir(parents=True, exist_ok=True)
    for caption_record in tqdm(caption_records, desc="make generation tasks", unit="caption"):
        gt_path = gt_root / f"meta_{int(caption_record['metadata_index']):06d}.png"
        if not gt_path.exists():
            caption_record["_dataset"].image_for_row(caption_record["row_idx"]).save(gt_path)
        for sample_index in range(args.samples_per_caption):
            seed = args.seed + int(caption_record["metadata_index"]) * 100_000 + sample_index
            tasks.append({
                "record": caption_record,
                "metadata_index": int(caption_record["metadata_index"]),
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
        out_path = generated_root / f"meta_{task['metadata_index']:06d}_sample_{task['sample_index']:03d}.png"
        task["generated_image_path"] = str(out_path)
        if args.overwrite or not out_path.exists():
            pending.append(task)
    if not pending:
        print("All generated images already exist; skipping DiT generation.")
        return

    generator = DiTGenerator(args, index, device)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.image_save_workers) as executor:
        futures = []
        for start in tqdm(range(0, len(pending), args.generation_batch_size), desc="generate images", unit="batch"):
            batch = pending[start:start + args.generation_batch_size]
            decoded = generator.generate_batch([task["record"] for task in batch], [task["sample_seed"] for task in batch])
            decoded_cpu = decoded.detach().cpu()
            for img_tensor, task in zip(decoded_cpu, batch):
                futures.append(executor.submit(lambda t, p: tensor_to_pil(t).save(p), img_tensor, task["generated_image_path"]))
        for future in futures:
            future.result()


def build_dataset_verifier(args, device):
    if args.vlm == "gemini":
        return build_verifier(
            "open-router",
            model=args.gemini_model,
            api_key_env=args.openrouter_api_key_env,
            temperature=args.vlm_temperature,
            max_tokens=args.max_new_tokens,
            retries=args.retries,
            workers=args.vlm_workers,
            include_caption=args.use_caption,
            include_metadata=False,
            image_size=args.image_size,
        )
    if args.vlm == "qwen-vllm":
        return build_verifier(
            "vllm-qwen",
            model=args.vllm_model,
            api_url=args.vllm_base_url,
            api_key=args.vllm_api_key,
            temperature=args.vlm_temperature,
            max_tokens=args.max_new_tokens,
            retries=args.retries,
            workers=args.vlm_workers,
            enable_thinking=args.enable_thinking,
            include_caption=args.use_caption,
            include_metadata=False,
            image_size=args.image_size,
        )
    if args.vlm == "qwen-local":
        return build_verifier(
            "local-qwen",
            model_id=args.qwen_model,
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


def selected_vlm_model(args):
    if args.vlm == "gemini":
        return args.gemini_model
    if args.vlm == "qwen-local":
        return args.qwen_model
    return args.vllm_model


def build_output_row(args, task, feedback, token_usage):
    return {
        "tuple_id": tuple_id(task),
        "metadata_index": task["metadata_index"],
        "caption": task["caption"],
        "metadata": task["metadata"],
        "gt_image_path": task["gt_image_path"],
        "generated_image_path": task["generated_image_path"],
        "source_image_path": task["source_image_path"],
        "sample_index": task["sample_index"],
        "feedback_index": task["feedback_index"],
        "sample_seed": task["sample_seed"],
        "feedback": feedback,
        "vlm": args.vlm,
        "vlm_model": selected_vlm_model(args),
        "token_usage": token_usage,
        "ckpt": args.ckpt,
    }


def run_feedback(args, tasks, jsonl_path, failures_path, device):
    verifier = build_dataset_verifier(args, device)
    for start in tqdm(range(0, len(tasks), args.vlm_batch_size), desc="vlm feedback", unit="batch"):
        batch = tasks[start:start + args.vlm_batch_size]
        image_pairs = [
            (
                resize_square(Image.open(task["gt_image_path"]), args.image_size),
                resize_square(Image.open(task["generated_image_path"]), args.image_size),
            )
            for task in batch
        ]
        results = verifier.verify_batch(
            [task["caption"] for task in batch],
            [task["metadata"] for task in batch],
            [pair[0] for pair in image_pairs],
            [pair[1] for pair in image_pairs],
        )
        for task, result in zip(batch, results):
            row_task = {k: v for k, v in task.items() if k != "record"}
            if result.ok:
                append_jsonl(jsonl_path, build_output_row(args, row_task, result.feedback, result.token_usage or {}))
            else:
                append_jsonl(failures_path, {**row_task, "metadata": None, "error": result.error, "vlm": args.vlm})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model", type=str, default="DiT-L/4", choices=list(DiT_models.keys()))
    parser.add_argument("--vae", type=str, default="stabilityai/sdxl-vae")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--samples-per-caption", type=int, default=10)
    parser.add_argument("--feedbacks-per-image", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--image-save-workers", type=int, default=8)
    parser.add_argument("--vlm", choices=["gemini", "qwen-local", "qwen-vllm"], default="gemini")
    parser.add_argument("--vlm-temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--use-caption", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--vlm-workers", type=int, default=8)
    parser.add_argument("--vlm-batch-size", type=int, default=8)
    parser.add_argument("--gemini-model", type=str, default=f"google/{DEFAULT_GEMINI_MODEL}")
    parser.add_argument("--openrouter-api-key-env", type=str, default="OPENROUTER_API_KEY")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--qwen-model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--qwen-dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--qwen-device-map", type=str, default="auto")
    parser.add_argument("--vllm-base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--vllm-api-key", type=str, default="EMPTY")
    parser.add_argument("--vllm-model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "feedback_dataset.jsonl"
    failures_path = out_dir / "failures.jsonl"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = load_index(args.dataset_root)
    metadata_rows = load_jsonl(Path(args.dataset_root).parent / "metadata.jsonl") if (Path(args.dataset_root).parent / "metadata.jsonl").exists() else []
    caption_records = select_caption_records(index, args.split, args.max_images)
    if not caption_records:
        raise RuntimeError("No caption records selected")

    generation_tasks = make_generation_tasks(args, caption_records)
    generate_missing_images(args, index, generation_tasks, device)

    feedback_tasks = []
    for gen_task in generation_tasks:
        metadata_index = gen_task["metadata_index"]
        metadata = metadata_rows[metadata_index] if 0 <= metadata_index < len(metadata_rows) else None
        for feedback_index in range(args.feedbacks_per_image):
            feedback_tasks.append({**gen_task, "feedback_index": feedback_index, "metadata": metadata})
    completed_keys = {tuple_key(row) for row in load_jsonl(jsonl_path)} if jsonl_path.exists() else set()
    pending = [task for task in feedback_tasks if tuple_key(task) not in completed_keys]
    print(f"Captions: {len(caption_records)}, generated: {len(generation_tasks)}, "
          f"feedback complete: {len(completed_keys)}, pending: {len(pending)}")

    if pending:
        run_feedback(args, pending, jsonl_path, failures_path, device)

    all_rows = load_jsonl(jsonl_path) if jsonl_path.exists() else []
    write_json(out_dir / "manifest.json", {
        "args": {k: v for k, v in vars(args).items()},
        "completed_feedback_rows": len(all_rows),
        "expected_feedback_rows": len(feedback_tasks),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    print(f"Saved dataset to {jsonl_path}")


if __name__ == "__main__":
    main()
