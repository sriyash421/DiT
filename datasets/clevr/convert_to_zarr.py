"""Stage 2: convert a filtered CLEVR dataset (metadata.jsonl or feedback_dataset.jsonl) into data.zarr.

Writes images, captions, and row metadata. Context tokens are added by encode_context.py.
"""
import argparse
import json
from pathlib import Path

from tqdm import tqdm

from algorithms.utils import load_jsonl, write_json
from datasets.clevr.utils import ClevrZarrWriter


def base_rows(dataset_root):
    rows = []
    for metadata_index, row in enumerate(load_jsonl(Path(dataset_root) / "metadata.jsonl")):
        rows.append({
            "dataset_type": "base",
            "metadata_index": metadata_index,
            "image_path": row["image_path"],
            "split": row["split"],
            "caption": row["caption"],
        })
    return rows


def feedback_rows(dataset_root):
    rows = []
    for row in load_jsonl(Path(dataset_root) / "feedback_dataset.jsonl"):
        rows.append({
            "dataset_type": "feedback",
            "metadata_index": row["metadata_index"],
            "gt_image_path": row["gt_image_path"],
            "generated_image_path": row["generated_image_path"],
            "split": row.get("split", "train"),
            "caption": row["caption"],
            "feedback": row["feedback"],
            "sample_index": row.get("sample_index"),
            "feedback_index": row.get("feedback_index"),
            "tuple_id": row.get("tuple_id"),
        })
    return rows


def main(args):
    dataset_root = Path(args.dataset)
    out = Path(args.out) if args.out is not None else dataset_root / "data.zarr"
    rows = base_rows(dataset_root) if args.mode == "base" else feedback_rows(dataset_root)

    writer = ClevrZarrWriter(
        output=out,
        dataset_root=dataset_root,
        dataset_type=args.mode,
        image_size=args.image_size,
        overwrite=args.overwrite,
    )
    for start in tqdm(range(0, len(rows), args.batch_size), desc=f"write {args.mode} zarr", unit="batch"):
        writer.append_batch(rows[start:start + args.batch_size])

    stats = writer.stats()
    write_json(out.parent / f"{out.name}.stats.json", stats)
    print(f"Wrote {stats['rows']} rows to {out}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--mode", choices=["base", "feedback"], default="base")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
