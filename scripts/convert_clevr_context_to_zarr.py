#!/usr/bin/env python3
"""Convert existing CLEVR context shard datasets to data.zarr."""
import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clevr_zarr import ClevrZarrWriter, write_json  # noqa: E402


def load_index(dataset_root):
    with (Path(dataset_root) / "context_embeddings" / "index.json").open() as f:
        return json.load(f)


def main(args):
    dataset_root = Path(args.dataset)
    out = Path(args.out) if args.out is not None else dataset_root / "data.zarr"
    index = load_index(dataset_root)
    records = index["records"]
    if args.limit is not None:
        records = records[:args.limit]
    if not records:
        raise RuntimeError("No records to convert.")

    context_root = dataset_root / "context_embeddings"
    shard_cache = {}
    first_shard = torch.load(context_root / records[0]["shard"], map_location="cpu", weights_only=False)
    first_token = first_shard["context_tokens"][int(records[0]["offset"])]
    context_dim = int(first_token.shape[-1])
    shard_cache[records[0]["shard"]] = first_shard

    writer = ClevrZarrWriter(
        output=out,
        dataset_root=dataset_root,
        dataset_type=args.mode or index.get("dataset_type", "base"),
        vlm_model=index.get("vlm_model", ""),
        max_context_len=int(index.get("max_context_len", args.max_context_len)),
        dtype=index.get("dtype", args.dtype),
        image_size=args.image_size,
        context_dim=context_dim,
        overwrite=args.overwrite,
        token_chunk_length=args.token_chunk_length,
        row_chunk_length=args.row_chunk_length,
    )

    batch_rows = []
    batch_tokens = []
    for record in tqdm(records, desc="convert context shards", unit="row"):
        shard_name = record["shard"]
        if shard_name not in shard_cache:
            shard_cache.clear()
            shard_cache[shard_name] = torch.load(context_root / shard_name, map_location="cpu", weights_only=False)
        shard = shard_cache[shard_name]
        token = shard["context_tokens"][int(record["offset"])]
        batch_rows.append(record)
        batch_tokens.append(token)
        if len(batch_rows) >= args.batch_size:
            writer.append_batch(batch_rows, batch_tokens)
            batch_rows = []
            batch_tokens = []
    writer.append_batch(batch_rows, batch_tokens)

    stats = writer.stats()
    write_json(out.parent / f"{out.name}.stats.json", stats)
    print(f"Wrote {stats['rows']:,} rows to {out}")
    print(f"Wrote stats to {out.parent / f'{out.name}.stats.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--mode", choices=["base", "feedback"], default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-context-len", type=int, default=1024)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--token-chunk-length", type=int, default=512)
    parser.add_argument("--row-chunk-length", type=int, default=4096)
    main(parser.parse_args())
