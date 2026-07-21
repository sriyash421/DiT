"""Stage 1: generate the full T2I-CompBench generation dump (run in .venv-omni).

For every train prompt, generate all N FLUX candidates, score each with the CompBench metric, and
save every candidate image plus a manifest row (prompt, category, split, candidate index, score).
Val prompts get a single black placeholder. No selection and no zarr here — build_compbench_zarr.py
consumes the manifest and keeps the best. Resumable: prompts already in the manifest are skipped;
images are written atomically so a killed job leaves no truncated PNG.
"""
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verifiers.compbench import CompBenchEval

CATEGORIES = ["color", "shape", "texture", "spatial", "3d_spatial", "non_spatial", "numeracy", "complex"]
PROMPT_URL = "https://raw.githubusercontent.com/Karine-Huang/T2I-CompBench/main/examples/dataset/{category}_{split}.txt"


def fetch_prompts(category, split, cache_dir):
    path = cache_dir / f"{category}_{split}.txt"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(PROMPT_URL.format(category=category, split=split), path)
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def save_atomic(image, path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.png")
    image.convert("RGB").resize((size, size)).save(tmp)
    os.replace(tmp, path)


def generate_candidates(pipe, prompt, n, batch_size, seed, steps, guidance, size):
    import torch

    images = []
    while len(images) < n:
        chunk = min(batch_size, n - len(images))
        generators = [torch.Generator("cuda").manual_seed(seed + len(images) + i) for i in range(chunk)]
        out = pipe(
            prompt,
            num_images_per_prompt=chunk,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=size,
            width=size,
            generator=generators,
        )
        images.extend(out.images)
    return images


def load_done(manifest_path):
    done = set()
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["category"], row["split"], row["prompt_index"]))
    return done


def build_tasks(categories, prompt_cache):
    """Flat (category, split, prompt_index, prompt) list over every prompt, sharded by position."""
    tasks = []
    for category in categories:
        for split in ("train", "val"):
            for idx, prompt in enumerate(fetch_prompts(category, split, prompt_cache)):
                tasks.append((category, split, idx, prompt))
    return tasks


def main(args):
    out = Path(args.out)
    images_dir = out / "images"
    manifest_path = out / f"manifest_{args.shard:02d}.jsonl"
    out.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args.categories, out / "prompts")
    tasks = [task for i, task in enumerate(tasks) if i % args.num_shards == args.shard]
    done = load_done(manifest_path)

    pipe = None
    scorer = None
    with open(manifest_path, "a") as manifest:
        for category, split, idx, prompt in tasks:
            if (category, split, idx) in done:
                continue
            if split == "val":
                path = images_dir / "val" / f"{category}_{idx:04d}.png"
                save_atomic(Image.new("RGB", (args.save_size, args.save_size), (0, 0, 0)), path, args.save_size)
                manifest.write(json.dumps({
                    "category": category, "split": "val", "prompt_index": idx,
                    "candidate_index": 0, "image_path": str(path.relative_to(out)),
                    "prompt": prompt, "score": None,
                }) + "\n")
                manifest.flush()
                continue
            if pipe is None:
                import torch
                from diffusers import FluxPipeline

                pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")
                scorer = CompBenchEval(device="cuda")
            candidates = generate_candidates(
                pipe, prompt, args.n_candidates, args.gen_batch_size,
                args.seed + idx * args.n_candidates, args.steps, args.guidance, args.gen_size,
            )
            for cand_idx, image in enumerate(candidates):
                score = scorer.score_one(prompt, image, category)
                path = images_dir / "train" / f"{category}_{idx:04d}_{cand_idx}.png"
                save_atomic(image, path, args.save_size)
                manifest.write(json.dumps({
                    "category": category, "split": "train", "prompt_index": idx,
                    "candidate_index": cand_idx, "image_path": str(path.relative_to(out)),
                    "prompt": prompt, "score": float(score),
                }) + "\n")
            manifest.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="/gscratch/scrubbed/sriyash/comp_bench")
    parser.add_argument("--categories", nargs="+", default=["color"], choices=CATEGORIES)
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--n-candidates", type=int, default=16)
    parser.add_argument("--gen-batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--gen-size", type=int, default=1024)
    parser.add_argument("--save-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    main(parser.parse_args())
