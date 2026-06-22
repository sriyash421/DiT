#!/usr/bin/env python3
"""Precompute frozen Qwen context tokens for CLEVR base or feedback rows."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from clevr_captions import render_caption
from clevr_zarr import ClevrZarrWriter, write_json
from vlm_utils import (
    build_context_text,
    encode_contexts,
    load_metadata_rows,
    load_vlm,
    metadata_for_row,
    resolve_path,
)

DEFAULT_TEMPLATES = ("chain", "order", "compact")


class BaseRows(Dataset):
    def __init__(self, dataset_root):
        self.root = Path(dataset_root)
        self.rows = []
        with (self.root / "metadata.jsonl").open() as f:
            metadata = [json.loads(line) for line in f if line.strip()]
        for metadata_index, row in enumerate(metadata):
            for template in DEFAULT_TEMPLATES:
                caption = render_caption(row, template=template)
                self.rows.append({
                    "dataset_type": "base",
                    "metadata_index": metadata_index,
                    "image_path": row["image_path"],
                    "split": row["split"],
                    "template": template,
                    "caption": caption,
                    "metadata": row,
                })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class FeedbackRows(Dataset):
    def __init__(self, dataset_root):
        self.root = Path(dataset_root)
        self.rows = []
        with (self.root / "feedback_dataset.jsonl").open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                self.rows.append({
                    "dataset_type": "feedback",
                    "metadata_index": row["metadata_index"],
                    "gt_image_path": row["gt_image_path"],
                    "generated_image_path": row["generated_image_path"],
                    "source_image_path": row.get("source_image_path", row["gt_image_path"]),
                    "split": row.get("split", "train"),
                    "template": row.get("template", "feedback"),
                    "caption": row["caption"],
                    "feedback": row["feedback"],
                    "metadata": row.get("metadata"),
                    "sample_index": row.get("sample_index"),
                    "feedback_index": row.get("feedback_index"),
                    "tuple_id": row.get("tuple_id"),
                })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def collate_rows(rows):
    return rows


def print_stats(stats):
    lengths = stats["token_lengths"]
    print("\nCached context dataset")
    print(f"  rows: {stats['rows']:,}")
    print(f"  type: {stats['dataset_type']}")
    print(f"  context_dim: {stats['context_dim']}")
    print(f"  max_context_len: {stats['max_context_len']}")
    print(f"  splits: {stats['splits']}")
    print(f"  templates: {stats['templates']}")
    print(
        "  token_lengths: "
        f"min={lengths['min']}, mean={lengths['mean']:.1f}, p50={lengths['p50']:.1f}, "
        f"p95={lengths['p95']:.1f}, p99={lengths['p99']:.1f}, max={lengths['max']}"
    )
    if "feedback_rows" in stats:
        print(f"  feedback_rows: {stats['feedback_rows']:,}")
    print(f"  output: {stats.get('output', stats.get('output_dir'))}")


def main(args):
    dataset_root = Path(args.dataset)
    out_dir = Path(args.out) if args.out is not None else dataset_root / "data.zarr"

    if args.mode == "base":
        rows = BaseRows(dataset_root)
        metadata_rows, metadata_by_image_path = [], {}
    else:
        rows = FeedbackRows(dataset_root)
        metadata_rows, metadata_by_image_path = load_metadata_rows(dataset_root)

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    processor, vlm = load_vlm(args.vlm_model, device, dtype=args.vlm_dtype, device_map=args.device_map)
    loader = DataLoader(rows, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_rows)

    token_lengths = []
    writer = None

    progress = tqdm(total=len(rows), desc=f"cache {args.mode} context", unit="row")
    for batch in loader:
        texts = []
        images = []
        for row in batch:
            metadata = metadata_for_row(row, metadata_rows, metadata_by_image_path)
            texts.append(build_context_text(row.get("caption", ""), metadata, row.get("feedback")))
            if args.mode == "feedback":
                image_path = resolve_path(dataset_root, row["generated_image_path"])
                images.append(Image.open(image_path).convert("RGB"))
            else:
                images.append(None)
        tokens, masks = encode_contexts(
            processor,
            vlm,
            texts,
            device,
            images=images,
            max_length=args.max_context_len,
            out_dtype=dtype,
        )
        valid_tokens = []
        for row, token, mask in zip(batch, tokens, masks):
            valid_len = int(mask.sum().item())
            token_lengths.append(valid_len)
            valid_tokens.append(token[mask.bool()].contiguous())
        if writer is None:
            writer = ClevrZarrWriter(
                output=out_dir,
                dataset_root=dataset_root,
                dataset_type=args.mode,
                vlm_model=args.vlm_model,
                max_context_len=args.max_context_len,
                dtype=args.dtype,
                image_size=args.image_size,
                context_dim=int(valid_tokens[0].shape[-1]),
                overwrite=args.overwrite,
                token_chunk_length=args.token_chunk_length,
                row_chunk_length=args.row_chunk_length,
            )
        writer.append_batch(batch, valid_tokens)
        progress.update(len(batch))
    progress.close()

    if writer is None:
        raise RuntimeError("No rows were encoded.")
    stats = writer.stats()
    write_json(out_dir.parent / f"{out_dir.name}.stats.json", stats)
    print(f"Wrote {stats['rows']} rows to {out_dir}")
    print(f"Wrote stats to {out_dir.parent / f'{out_dir.name}.stats.json'}")
    print_stats(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--mode", choices=["base", "feedback"], default="base")
    parser.add_argument("--vlm-model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--vlm-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--max-context-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--token-chunk-length", type=int, default=512)
    parser.add_argument("--row-chunk-length", type=int, default=4096)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.device_map == "":
        args.device_map = None
    main(args)
