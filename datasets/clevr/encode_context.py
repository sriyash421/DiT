"""Stage 3: encode captions (plus feedback and generated images) with a frozen Qwen VLM
and append the context tokens to an existing data.zarr store.

Contexts use the single interleaved history format: base rows are caption-only (empty
history), feedback rows are 1-step histories (caption, generated image, feedback).
"""
import argparse

import numpy as np
import torch
import zarr
from PIL import Image
from tqdm import tqdm

from datasets.clevr.utils import ContextTokenWriter


def encode_context_tokens(
    zarr_path,
    vlm_model,
    device,
    *,
    batch_size=4,
    max_context_len=1024,
    dtype="float16",
    vlm_dtype="bfloat16",
    overwrite=False,
    device_map=None,
):
    """Append frozen-VLM context tokens to a stage-2 zarr; returns the writer stats."""
    from models.qwen_vlm import QwenEncoder

    root = zarr.open(str(zarr_path), mode="r")
    meta = root["meta"]
    data = root["data"]
    captions = meta["caption"][:]
    feedbacks = meta["feedback"][:]
    generated_image_indices = data["generated_image_index"][:]
    num_rows = len(captions)

    device = torch.device(device)
    out_dtype = getattr(torch, dtype)
    encoder = QwenEncoder(
        model_id=vlm_model,
        device=device,
        dtype=vlm_dtype,
        max_length=max_context_len,
        freeze=True,
        device_map=device_map,
    )

    writer = None
    for start in tqdm(range(0, num_rows, batch_size), desc="encode context", unit="batch"):
        row_indices = range(start, min(start + batch_size, num_rows))
        batch_captions = []
        feedback_histories = []
        image_histories = []
        for row_idx in row_indices:
            feedback = str(feedbacks[row_idx])
            generated_idx = int(generated_image_indices[row_idx])
            assert bool(feedback) == (generated_idx >= 0), (
                f"Row {row_idx}: feedback rows must carry a generated image and base rows must not."
            )
            batch_captions.append(str(captions[row_idx]))
            if feedback:
                image = Image.fromarray(np.asarray(data["generated_images"][generated_idx]), mode="RGB")
                feedback_histories.append([feedback])
                image_histories.append([image])
            else:
                feedback_histories.append([])
                image_histories.append([])

        tokens, masks = encoder.encode_history(batch_captions, feedback_histories, image_histories)
        tokens = tokens.detach().cpu().to(out_dtype)
        masks = masks.detach().cpu()
        valid_tokens = [token[mask].contiguous() for token, mask in zip(tokens, masks)]
        if writer is None:
            writer = ContextTokenWriter(
                zarr_path=zarr_path,
                vlm_model=vlm_model,
                context_dim=int(valid_tokens[0].shape[-1]),
                max_context_len=max_context_len,
                dtype=dtype,
                overwrite=overwrite,
            )
        writer.append_batch(valid_tokens)

    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return writer.stats()


def main(args):
    stats = encode_context_tokens(
        args.zarr,
        args.vlm_model,
        args.device,
        batch_size=args.batch_size,
        max_context_len=args.max_context_len,
        dtype=args.dtype,
        vlm_dtype=args.vlm_dtype,
        overwrite=args.overwrite,
        device_map=args.device_map,
    )
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
