#!/usr/bin/env python3
"""Remove duplicate feedback rows from a generated VLM-feedback dataset.

By default, duplicates are exact repeated feedback strings for the same
generated image. The first occurrence is kept.
"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def load_jsonl(path):
    rows = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def normalize_feedback(text):
    return " ".join(str(text).strip().split())


def dedupe_key(row, mode):
    feedback = normalize_feedback(row.get("feedback", ""))
    if mode == "per-generated-image":
        return (row.get("generated_image_path"), feedback)
    if mode == "per-caption-generated-image":
        return (row.get("metadata_index"), row.get("template"), row.get("sample_index"), feedback)
    if mode == "global-feedback":
        return (feedback,)
    raise ValueError(f"Unknown dedupe mode: {mode}")


def summarize(rows):
    by_image = {}
    for row in rows:
        by_image.setdefault(row.get("generated_image_path"), []).append(normalize_feedback(row.get("feedback", "")))
    unique_counts = Counter(len(set(feedbacks)) for feedbacks in by_image.values())
    return {
        "rows": len(rows),
        "generated_images": len(by_image),
        "unique_feedbacks_per_generated_image": dict(sorted(unique_counts.items())),
    }


def main(args):
    dataset_dir = Path(args.dataset)
    input_jsonl = dataset_dir / "feedback_dataset.jsonl"
    rows = load_jsonl(input_jsonl)

    seen = set()
    kept = []
    removed = []
    for row in rows:
        key = dedupe_key(row, args.mode)
        if key in seen:
            removed.append(row)
            continue
        seen.add(key)
        kept.append(row)

    if args.in_place:
        output_dir = dataset_dir
        backup_path = dataset_dir / args.backup_name
        if backup_path.exists() and not args.overwrite:
            raise FileExistsError(f"Backup already exists: {backup_path}. Pass --overwrite to replace it.")
        if backup_path.exists():
            backup_path.unlink()
        shutil.copy2(input_jsonl, backup_path)
    else:
        if args.out_dir is None:
            raise ValueError("Provide --out-dir or pass --in-place.")
        output_dir = Path(args.out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("manifest.json", "progress.json"):
            src = dataset_dir / name
            if src.exists():
                shutil.copy2(src, output_dir / name)

    output_jsonl = output_dir / "feedback_dataset.jsonl"
    write_jsonl(output_jsonl, kept)
    if args.write_removed:
        write_jsonl(output_dir / "removed_duplicates.jsonl", removed)

    manifest = {
        "input_dataset": str(dataset_dir),
        "output_dataset": str(output_dir),
        "mode": args.mode,
        "input_summary": summarize(rows),
        "output_summary": summarize(kept),
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "removed_rows": len(removed),
    }
    with (output_dir / "dedupe_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    print(f"Wrote deduplicated dataset to {output_jsonl}")
    if args.in_place:
        print(f"Original JSONL backed up to {backup_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--backup-name", type=str, default="feedback_dataset.before_dedupe.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["per-generated-image", "per-caption-generated-image", "global-feedback"],
        default="per-generated-image",
    )
    parser.add_argument("--write-removed", action=argparse.BooleanOptionalAction, default=True)
    main(parser.parse_args())
