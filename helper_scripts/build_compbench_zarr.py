"""Stage 2: build the CompBench zarr from the generation dump (run in .venv, zarr 2.18.7).

Reads the manifest written by generate_compbench_dataset.py, keeps the top-k candidates per train
prompt by CompBench score, and writes a base zarr in the exact CLEVR schema via ClevrZarrWriter
(category carried in tuple_id). Val prompts contribute their single black placeholder.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.clevr.utils import ClevrZarrWriter


def main(args):
    data = Path(args.data)
    out = Path(args.out) if args.out else data / "data.zarr"
    manifest = [
        json.loads(line)
        for path in sorted(data.glob(f"{args.manifest_prefix}*.jsonl"))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    assert manifest, f"no rows matched {data}/{args.manifest_prefix}*.jsonl"

    train = defaultdict(list)
    val = []
    for row in manifest:
        if row["split"] == "train":
            train[(row["category"], row["prompt_index"])].append(row)
        else:
            val.append(row)

    rows = []
    for key in sorted(train):
        best = sorted(train[key], key=lambda r: r["score"], reverse=True)[:args.top_k]
        for row in best:
            rows.append({"image_path": row["image_path"], "split": "train",
                         "caption": row["prompt"], "tuple_id": row["category"], "metadata_index": len(rows)})
    for row in sorted(val, key=lambda r: (r["category"], r["prompt_index"])):
        rows.append({"image_path": row["image_path"], "split": "val",
                     "caption": row["prompt"], "tuple_id": row["category"], "metadata_index": len(rows)})

    writer = ClevrZarrWriter(out, data, "base", args.image_size, overwrite=args.overwrite)
    for start in range(0, len(rows), 4096):
        writer.append_batch(rows[start:start + 4096])
    Path(f"{out}.stats.json").write_text(json.dumps(writer.stats(), indent=2))
    print(writer.stats())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/gscratch/scrubbed/sriyash/comp_bench")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--manifest-prefix", type=str, default="manifest_",
                        help="Manifest file prefix to read; use 'gdino_manifest_' to select on the "
                             "open-vocab Grounding DINO rescore instead of the original scores.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
