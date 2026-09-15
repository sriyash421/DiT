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
import urllib.error
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
        # Cluster nodes lack a system CA bundle, so stdlib urllib fails SSL verification; use certifi.
        import ssl
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(PROMPT_URL.format(category=category, split=split), context=ctx) as r:
            path.write_bytes(r.read())
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def openrouter_candidates(model, prompt, n, workers, timeout=300, retries=7):
    """Generate n candidates for one prompt via OpenRouter, concurrently.

    These are blocking HTTP calls, so serialising them wastes almost all the wall clock: measured
    44.5s/call serial vs 1.7s/image amortised at 32-way concurrency (26x). Retries cover 429/5xx,
    which do occur at this concurrency.
    """
    import base64
    import io
    import ssl
    import time
    from concurrent.futures import ThreadPoolExecutor

    import certifi
    from PIL import Image as PILImage

    ctx = ssl.create_default_context(cafile=certifi.where())
    key = os.environ.get("OPENROUTER_API_KEY", "")

    def one(_):
        payload = {"model": model, "modalities": ["image", "text"],
                   "messages": [{"role": "user",
                                 "content": f"Generate a photograph showing exactly: {prompt}"}]}
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
                    msg = json.loads(r.read())["choices"][0]["message"]
                out = []
                for im in (msg.get("images") or []):
                    url = (im.get("image_url") or {}).get("url", "")
                    if url.startswith("data:"):
                        out.append(PILImage.open(
                            io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB"))
                return out
            except urllib.error.HTTPError as e:
                if e.code in (408, 409, 429, 500, 502, 503, 504, 529):
                    time.sleep(min(2 ** attempt * 3, 90))
                    continue
                print(f"      HTTP {e.code}: {e.read()[:160]}", flush=True)
                return []
            except Exception as e:
                if attempt == retries - 1:
                    print(f"      gave up: {type(e).__name__}: {str(e)[:160]}", flush=True)
                time.sleep(min(2 ** attempt * 2, 60))
        return []

    imgs = []
    with ThreadPoolExecutor(max_workers=min(workers, n)) as ex:
        for got in ex.map(one, range(n)):
            imgs.extend(got)
    return imgs[:n]


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
    """Union over EVERY shard manifest, not just this shard's.

    Sharding is positional (i % num_shards), so re-running with a different --num-shards remaps
    prompts to different shards. Reading only this shard's manifest would then miss work another
    shard already did and silently pay to generate it again.
    """
    done = set()
    for path in sorted(manifest_path.parent.glob("manifest_*.jsonl")):
        for line in path.read_text().splitlines():
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
            if scorer is None:
                import torch
                if args.backend == "flux":
                    from diffusers import FluxPipeline
                    pipe = FluxPipeline.from_pretrained(args.model,
                                                        torch_dtype=torch.bfloat16).to("cuda")
                detector = None
                if any(c in ("spatial", "3d_spatial", "numeracy") for c in args.categories):
                    # Open-vocab Grounding DINO scorer (replaces UniDet's closed 722-class taxonomy).
                    from verifiers.gdino_numeracy import GDinoNumeracyScorer
                    # min(gdino, owlv2): exact-count 79.0% -> 85.2% on 81 hand-labelled images
                    detector = GDinoNumeracyScorer(model_name=args.gdino_model, device="cuda",
                                                   owlv2_min=True)
                scorer = CompBenchEval(device="cuda", use_unidet=bool(detector), unidet_scorer=detector)
            if args.backend == "openrouter":
                candidates = openrouter_candidates(args.api_model, prompt, args.n_candidates,
                                                   args.api_workers)
                if not candidates:
                    print(f"  no candidates for '{prompt}' -- skipping (will retry on requeue)",
                          flush=True)
                    continue
            else:
                candidates = generate_candidates(
                    pipe, prompt, args.n_candidates, args.gen_batch_size,
                    args.seed + idx * args.n_candidates, args.steps, args.guidance, args.gen_size,
                )
            # Prompt-level atomic: save all images, then write all manifest lines at once. A
            # preemption mid-prompt leaves no manifest lines, so the whole prompt is redone on requeue
            # (never a half-set silently marked done).
            lines = []
            for cand_idx, image in enumerate(candidates):
                score = scorer.score_one(prompt, image, category)
                path = images_dir / "train" / f"{category}_{idx:04d}_{cand_idx}.png"
                save_atomic(image, path, args.save_size)
                lines.append(json.dumps({
                    "category": category, "split": "train", "prompt_index": idx,
                    "candidate_index": cand_idx, "image_path": str(path.relative_to(out)),
                    "prompt": prompt, "score": float(score),
                }))
            manifest.write("\n".join(lines) + "\n")
            manifest.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="/gscratch/scrubbed/sriyash/comp_bench")
    parser.add_argument("--categories", nargs="+", default=["color"], choices=CATEGORIES)
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--n-candidates", type=int, default=16)
    parser.add_argument("--gen-batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--gen-size", type=int, default=1024)
    parser.add_argument("--save-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--backend", choices=["flux", "openrouter"], default="flux",
                        help="flux = local diffusers; openrouter = hosted API (no GPU for generation)")
    parser.add_argument("--api-model", type=str, default="openai/gpt-5-image-mini",
                        help="OpenRouter image model. gpt-5-image-mini measured best AND cheapest: "
                             "exact-count 0.88 vs 0.55 for FLUX.1-dev on the 10 hardest prompts.")
    parser.add_argument("--api-workers", type=int, default=32)
    parser.add_argument("--gdino-model", type=str, default="IDEA-Research/grounding-dino-base",
                        help="Open-vocab Grounding DINO checkpoint used to score "
                             "spatial/3d_spatial/numeracy candidates.")
    main(parser.parse_args())
