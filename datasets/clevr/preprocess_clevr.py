"""Stage 1: filter raw CLEVR scenes by object count and render the chain caption per image.

Writes images/ (symlinks or copies) and metadata.jsonl into the output directory.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

from tqdm import tqdm

from datasets.clevr.utils import full_description, render_caption, unique_labels


def relation_before(scene, a, b, rel):
    # CLEVR: relationships[rel][i] contains objects that are rel of object i.
    return a in scene["relationships"][rel][b]


def sorted_by_relation(scene, rel):
    num_objects = len(scene["objects"])
    return sorted(
        range(num_objects),
        key=lambda idx: sum(relation_before(scene, other, idx, rel) for other in range(num_objects)),
    )


def make_object_records(scene):
    labels = unique_labels(scene["objects"])
    records = []
    for idx, obj in enumerate(scene["objects"]):
        records.append({
            "id": idx,
            "label": labels[idx],
            "description": full_description(obj),
            "size": obj["size"],
            "color": obj["color"],
            "material": obj["material"],
            "shape": obj["shape"],
            "3d_coords": obj["3d_coords"],
            "pixel_coords": obj["pixel_coords"],
        })
    return records


def link_or_copy(src, dst, copy_images):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_images:
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.relpath(src, dst.parent), dst)


def main(args):
    clevr_root = Path(args.clevr_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records = []
    for split in args.splits:
        scene_path = clevr_root / "scenes" / f"CLEVR_{split}_scenes.json"
        with scene_path.open() as f:
            scenes = json.load(f)["scenes"]
        for scene in tqdm(scenes, desc=f"filter {split}", unit="scene"):
            if len(scene["objects"]) != args.num_objects:
                continue
            image_path = Path("images") / split / scene["image_filename"]
            link_or_copy(clevr_root / "images" / split / scene["image_filename"], out / image_path, args.copy_images)
            record = {
                "image_path": str(image_path),
                "split": split,
                "image_index": scene["image_index"],
                "source_image_filename": scene["image_filename"],
                "objects": make_object_records(scene),
                "orders": {
                    "left_to_right": sorted_by_relation(scene, "left"),
                    "front_to_back": sorted_by_relation(scene, "front"),
                },
            }
            record["caption"] = render_caption(record)
            records.append(record)

    metadata_path = out / "metadata.jsonl"
    with metadata_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"Wrote {len(records)} rows to {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clevr-root", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--num-objects", type=int, default=3)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of creating symlinks.")
    main(parser.parse_args())
