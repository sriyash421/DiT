"""CLEVR caption rendering, transforms, metadata helpers, and zarr writers."""
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc, VLenUTF8
from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------------------------
# Captions (single "chain" template).
# ---------------------------------------------------------------------------

def full_description(obj):
    return f"{obj['size']} {obj['color']} {obj['material']} {obj['shape']}"


def label_candidates(obj):
    return [
        f"{obj['color']} {obj['shape']}",
        f"{obj['size']} {obj['color']} {obj['shape']}",
        f"{obj['color']} {obj['material']} {obj['shape']}",
        full_description(obj),
    ]


def unique_labels(objects):
    labels = [None] * len(objects)
    for level in range(4):
        candidates = [label_candidates(obj)[level] for obj in objects]
        counts = {candidate: candidates.count(candidate) for candidate in candidates}
        for idx, candidate in enumerate(candidates):
            if labels[idx] is None and counts[candidate] == 1:
                labels[idx] = candidate
    for idx, label in enumerate(labels):
        if label is None:
            labels[idx] = f"object {idx + 1} {full_description(objects[idx])}"
    return labels


def adjacent_chain(labels, order, relation):
    return ", ".join(
        f"{labels[order[idx + 1]]} is {relation} {labels[order[idx]]}"
        for idx in range(len(order) - 1)
    )


def render_caption(row):
    """Render the chain caption from a structured metadata row."""
    objects = row["objects"]
    labels = unique_labels(objects)
    object_list = ", ".join(full_description(obj) for obj in objects)
    horizontal = adjacent_chain(labels, row["orders"]["left_to_right"], "right of")
    depth = adjacent_chain(labels, row["orders"]["front_to_back"], "behind")
    return f"objects: {object_list}. horizontal: {horizontal}. depth: {depth}."


# ---------------------------------------------------------------------------
# Image transforms.
# ---------------------------------------------------------------------------

def build_clevr_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])


def resize_square_array(path, image_size):
    """Center-crop an image file to a square and resize; returns a uint8 array."""
    image = Image.open(path).convert("RGB")
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    image = image.crop((left, top, left + side, top + side))
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Metadata helpers.
# ---------------------------------------------------------------------------

def resolve_path(root, path):
    path = Path(path)
    if path.is_absolute():
        return path
    rooted = Path(root) / path
    if rooted.exists():
        return rooted
    return path


def load_metadata_rows(dataset_root):
    """Load metadata.jsonl from a dataset root; returns (rows, rows by image_path)."""
    path = Path(dataset_root) / "metadata.jsonl"
    if not path.exists():
        return [], {}
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows, {row["image_path"]: row for row in rows}


def load_metadata_for_zarr(dataset_path):
    """Load the metadata.jsonl that sits next to a data.zarr store."""
    path = Path(dataset_path)
    candidates = []
    if path.name == "data.zarr":
        candidates.append(path.parent / "metadata.jsonl")
    candidates.append(path / "metadata.jsonl")
    for candidate in candidates:
        if candidate.exists():
            rows = []
            with candidate.open() as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
            return rows
    return []


def metadata_by_index(metadata_rows, metadata_indices):
    out = []
    for idx in metadata_indices:
        idx = int(idx)
        out.append(metadata_rows[idx] if 0 <= idx < len(metadata_rows) else None)
    return out


def metadata_for_row(row, metadata_rows, metadata_by_image_path):
    if row.get("metadata") is not None:
        return row["metadata"]
    idx = row.get("metadata_index")
    if isinstance(idx, int) and 0 <= idx < len(metadata_rows):
        return metadata_rows[idx]
    return metadata_by_image_path.get(row.get("image_path") or row.get("source_image_path"))


def compact_metadata_text(metadata):
    if metadata is None:
        return None
    objects = metadata.get("objects", [])
    object_lines = []
    for obj in objects:
        object_lines.append(
            f"id {obj.get('id')}: {obj.get('size')} {obj.get('color')} "
            f"{obj.get('material')} {obj.get('shape')} ({obj.get('label')})"
        )
    orders = metadata.get("orders", {})

    def order_text(key):
        labels = []
        for idx in orders.get(key, []):
            match = next((obj for obj in objects if obj.get("id") == idx), None)
            labels.append(match.get("label", str(idx)) if match else str(idx))
        return ", ".join(labels)

    return "\n".join([
        "Objects:",
        *object_lines,
        f"Left-to-right order: {order_text('left_to_right')}",
        f"Front-to-back order: {order_text('front_to_back')}",
    ])


def build_context_text(caption, metadata=None, feedback=None, include_metadata=False):
    parts = []
    if caption:
        parts.append(f"Caption: {caption}")
    meta_text = compact_metadata_text(metadata) if include_metadata else None
    if meta_text:
        parts.append(f"CLEVR metadata:\n{meta_text}")
    if feedback:
        parts.append(f"Feedback: {feedback}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Zarr writers (schema v2: stage 2 writes rows + images, stage 3 appends context tokens).
# ---------------------------------------------------------------------------

def percentile(values, q):
    if not values:
        return 0
    values = sorted(values)
    pos = (len(values) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    weight = pos - lo
    return values[lo] * (1 - weight) + values[hi] * weight


class ClevrZarrWriter:
    """Stage-2 writer: rows with images, captions, and metadata (no context tokens)."""

    def __init__(self, output, dataset_root, dataset_type, image_size, overwrite=False, row_chunk_length=4096):
        self.output = Path(output)
        self.dataset_root = Path(dataset_root)
        self.dataset_type = dataset_type
        self.image_size = int(image_size)
        self.row_chunk_length = int(row_chunk_length)
        self.split_to_id = {}
        self.image_to_id = {}
        self.generated_image_to_id = {}
        self.rows = []

        if self.output.exists():
            if not overwrite:
                raise FileExistsError(f"{self.output} already exists. Pass --overwrite to replace it.")
            shutil.rmtree(self.output)

        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        root = zarr.open(str(self.output), mode="w")
        data = root.require_group("data")
        meta = root.require_group("meta")
        root.attrs.update({
            "format_version": 2,
            "dataset_type": dataset_type,
            "image_size": self.image_size,
            "rows": 0,
            "split_names": [],
        })
        self.root = root
        self.images = data.create_dataset(
            "images",
            shape=(0, self.image_size, self.image_size, 3),
            chunks=(1, self.image_size, self.image_size, 3),
            dtype=np.uint8,
            compressor=compressor,
            overwrite=True,
        )
        self.generated_images = data.create_dataset(
            "generated_images",
            shape=(0, self.image_size, self.image_size, 3),
            chunks=(1, self.image_size, self.image_size, 3),
            dtype=np.uint8,
            compressor=compressor,
            overwrite=True,
        )
        chunks = (self.row_chunk_length,)
        self.row_arrays = {
            "image_index": data.create_dataset("image_index", shape=(0,), chunks=chunks, dtype=np.int64, compressor=compressor, overwrite=True),
            "generated_image_index": data.create_dataset("generated_image_index", shape=(0,), chunks=chunks, dtype=np.int64, compressor=compressor, overwrite=True),
            "split_id": meta.create_dataset("split_id", shape=(0,), chunks=chunks, dtype=np.int8, compressor=compressor, overwrite=True),
            "is_feedback": meta.create_dataset("is_feedback", shape=(0,), chunks=chunks, dtype=bool, compressor=compressor, overwrite=True),
            "metadata_index": meta.create_dataset("metadata_index", shape=(0,), chunks=chunks, dtype=np.int64, compressor=compressor, overwrite=True),
            "sample_index": meta.create_dataset("sample_index", shape=(0,), chunks=chunks, dtype=np.int64, compressor=compressor, overwrite=True),
            "feedback_index": meta.create_dataset("feedback_index", shape=(0,), chunks=chunks, dtype=np.int64, compressor=compressor, overwrite=True),
            "caption": meta.create_dataset("caption", shape=(0,), chunks=chunks, dtype=object, object_codec=VLenUTF8(), compressor=compressor, overwrite=True),
            "feedback": meta.create_dataset("feedback", shape=(0,), chunks=chunks, dtype=object, object_codec=VLenUTF8(), compressor=compressor, overwrite=True),
            "tuple_id": meta.create_dataset("tuple_id", shape=(0,), chunks=chunks, dtype=object, object_codec=VLenUTF8(), compressor=compressor, overwrite=True),
        }

    def _id_for(self, mapping, name):
        if name not in mapping:
            mapping[name] = len(mapping)
        return mapping[name]

    def _append_image(self, arr, mapping, path):
        key = str(path)
        if key in mapping:
            return mapping[key]
        idx = arr.shape[0]
        arr.resize((idx + 1, self.image_size, self.image_size, 3))
        arr[idx] = resize_square_array(path, self.image_size)
        mapping[key] = idx
        return idx

    def append_batch(self, rows):
        if not rows:
            return
        start_row = self.row_arrays["image_index"].shape[0]
        end_row = start_row + len(rows)

        values = {key: [] for key in self.row_arrays}
        for row in rows:
            image_path = resolve_path(self.dataset_root, row.get("gt_image_path", row.get("image_path")))
            generated_path = row.get("generated_image_path")
            generated_idx = -1
            if generated_path:
                generated_idx = self._append_image(
                    self.generated_images,
                    self.generated_image_to_id,
                    resolve_path(self.dataset_root, generated_path),
                )
            values["image_index"].append(self._append_image(self.images, self.image_to_id, image_path))
            values["generated_image_index"].append(generated_idx)
            values["split_id"].append(self._id_for(self.split_to_id, row.get("split", "train")))
            values["is_feedback"].append(row.get("dataset_type", self.dataset_type) == "feedback")
            values["metadata_index"].append(int(row.get("metadata_index", -1)))
            values["sample_index"].append(int(row["sample_index"]) if row.get("sample_index") is not None else -1)
            values["feedback_index"].append(int(row["feedback_index"]) if row.get("feedback_index") is not None else -1)
            values["caption"].append(row.get("caption", ""))
            values["feedback"].append(row.get("feedback", ""))
            values["tuple_id"].append(row.get("tuple_id", ""))
            self.rows.append(row)

        for key, arr in self.row_arrays.items():
            arr.resize((end_row,))
            arr[start_row:end_row] = np.asarray(values[key], dtype=arr.dtype if arr.dtype != object else object)

        self.root.attrs["rows"] = end_row
        self.root.attrs["split_names"] = [name for name, _ in sorted(self.split_to_id.items(), key=lambda item: item[1])]

    def stats(self):
        split_counts = Counter(row.get("split", "train") for row in self.rows)
        return {
            "dataset": str(self.dataset_root),
            "output": str(self.output),
            "dataset_type": self.dataset_type,
            "rows": len(self.rows),
            "image_size": self.image_size,
            "splits": dict(sorted(split_counts.items())),
            "split_names": list(self.root.attrs["split_names"]),
        }


class ContextTokenWriter:
    """Stage-3 writer: appends frozen-VLM context tokens to an existing stage-2 zarr."""

    def __init__(self, zarr_path, vlm_model, context_dim, max_context_len, dtype, overwrite=False, token_chunk_length=512, row_chunk_length=4096):
        self.root = zarr.open(str(zarr_path), mode="r+")
        data = self.root["data"]
        if "context_tokens" in data:
            if not overwrite:
                raise FileExistsError(f"{zarr_path} already has context tokens. Pass --overwrite to replace them.")
            del data["context_tokens"]
            del data["context_offsets"]
        self.context_dim = int(context_dim)
        self.dtype = dtype
        self.token_lengths = []
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        self.context_tokens = data.create_dataset(
            "context_tokens",
            shape=(0, self.context_dim),
            chunks=(int(token_chunk_length), self.context_dim),
            dtype=np.dtype(dtype),
            compressor=compressor,
            overwrite=True,
        )
        self.context_offsets = data.create_dataset(
            "context_offsets",
            shape=(1,),
            chunks=(int(row_chunk_length),),
            dtype=np.int64,
            compressor=None,
            overwrite=True,
        )
        self.context_offsets[0] = 0
        self.root.attrs.update({
            "vlm_model": vlm_model,
            "context_dim": self.context_dim,
            "context_dtype": dtype,
            "max_context_len": int(max_context_len),
        })

    def append_batch(self, tokens):
        if not tokens:
            return
        lengths = [int(token.shape[0]) for token in tokens]
        start_row = self.context_offsets.shape[0] - 1
        end_row = start_row + len(tokens)
        flat_start = int(self.context_offsets[start_row])
        flat_end = flat_start + sum(lengths)

        self.context_tokens.resize((flat_end, self.context_dim))
        self.context_tokens[flat_start:flat_end] = np.concatenate(
            [np.asarray(token, dtype=self.dtype) for token in tokens],
            axis=0,
        )
        self.context_offsets.resize((end_row + 1,))
        offsets = np.empty(len(tokens) + 1, dtype=np.int64)
        offsets[0] = flat_start
        np.cumsum(np.asarray(lengths, dtype=np.int64), out=offsets[1:])
        offsets[1:] += flat_start
        self.context_offsets[start_row:end_row + 1] = offsets
        self.token_lengths.extend(lengths)

    def stats(self):
        lengths = self.token_lengths
        return {
            "rows": len(lengths),
            "context_dim": self.context_dim,
            "dtype": self.dtype,
            "token_lengths": {
                "min": int(min(lengths)),
                "mean": float(sum(lengths) / len(lengths)),
                "p50": percentile(lengths, 50),
                "p95": percentile(lengths, 95),
                "p99": percentile(lengths, 99),
                "max": int(max(lengths)),
            },
        }
