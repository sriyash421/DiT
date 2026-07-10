"""Stage 3: encode captions (plus feedback and generated images) with a frozen Qwen VLM
and append the context tokens to an existing data.zarr store.
"""
import argparse

import numpy as np
import torch
import zarr
from PIL import Image
from tqdm import tqdm

from datasets.clevr.utils import ContextTokenWriter, build_context_text, load_metadata_for_zarr
from models.qwen_vlm import encode_contexts, load_vlm


def main(args):
    root = zarr.open(args.zarr, mode="r")
    meta = root["meta"]
    data = root["data"]
    captions = meta["caption"][:]
    feedbacks = meta["feedback"][:]
    metadata_indices = meta["metadata_index"][:]
    generated_image_indices = data["generated_image_index"][:]
    num_rows = len(captions)
    metadata_rows = load_metadata_for_zarr(args.zarr)

    device = torch.device(args.device)
    out_dtype = getattr(torch, args.dtype)
    processor, vlm = load_vlm(args.vlm_model, device, dtype=args.vlm_dtype, device_map=args.device_map)

    writer = None
    for start in tqdm(range(0, num_rows, args.batch_size), desc="encode context", unit="batch"):
        row_indices = range(start, min(start + args.batch_size, num_rows))
        texts = []
        images = []
        for row_idx in row_indices:
            metadata_index = int(metadata_indices[row_idx])
            metadata = metadata_rows[metadata_index] if 0 <= metadata_index < len(metadata_rows) else None
            texts.append(build_context_text(str(captions[row_idx]), metadata, str(feedbacks[row_idx]) or None))
            generated_idx = int(generated_image_indices[row_idx])
            if generated_idx >= 0:
                images.append(Image.fromarray(np.asarray(data["generated_images"][generated_idx]), mode="RGB"))
            else:
                images.append(None)

        tokens, masks = encode_contexts(
            processor,
            vlm,
            texts,
            device,
            images=images,
            max_length=args.max_context_len,
            out_dtype=out_dtype,
        )
        valid_tokens = [token[mask.bool()].contiguous() for token, mask in zip(tokens, masks)]
        if writer is None:
            writer = ContextTokenWriter(
                zarr_path=args.zarr,
                vlm_model=args.vlm_model,
                context_dim=int(valid_tokens[0].shape[-1]),
                max_context_len=args.max_context_len,
                dtype=args.dtype,
                overwrite=args.overwrite,
            )
        writer.append_batch(valid_tokens)

    stats = writer.stats()
    lengths = stats["token_lengths"]
    print(f"Encoded {stats['rows']} rows into {args.zarr} (context_dim={stats['context_dim']})")
    print(
        f"token_lengths: min={lengths['min']}, mean={lengths['mean']:.1f}, "
        f"p50={lengths['p50']:.1f}, p95={lengths['p95']:.1f}, max={lengths['max']}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", type=str, required=True)
    parser.add_argument("--vlm-model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--vlm-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", type=str, default=None)
    parser.add_argument("--max-context-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
