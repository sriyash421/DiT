import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc, VLenUTF8
from PIL import Image


DEFAULT_TEMPLATES = ("chain", "order", "compact")


def resolve_path(root, path):
    path = Path(path)
    if path.is_absolute():
        return path
    rooted = Path(root) / path
    if rooted.exists():
        return rooted
    return path


def resize_square_array(path, image_size):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    image = image.crop((left, top, left + side, top + side))
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def percentile(values, q):
    if not values:
        return 0
    values = sorted(values)
    pos = (len(values) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    weight = pos - lo
    return values[lo] * (1 - weight) + values[hi] * weight


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def build_stats(dataset, output, dataset_type, vlm_model, rows, context_dim, max_context_len, dtype, image_size, token_lengths, split_names, template_names):
    split_counts = Counter(row.get("split", "unknown") for row in rows)
    template_counts = Counter(row.get("template", "unknown") for row in rows)
    stats = {
        "dataset": str(dataset),
        "output": str(output),
        "dataset_type": dataset_type,
        "vlm_model": vlm_model,
        "rows": len(rows),
        "context_dim": int(context_dim),
        "max_context_len": int(max_context_len),
        "dtype": dtype,
        "image_size": int(image_size),
        "splits": dict(sorted(split_counts.items())),
        "templates": dict(sorted(template_counts.items())),
        "split_names": list(split_names),
        "template_names": list(template_names),
        "token_lengths": {
            "min": int(min(token_lengths)),
            "mean": float(sum(token_lengths) / len(token_lengths)),
            "p50": percentile(token_lengths, 50),
            "p95": percentile(token_lengths, 95),
            "p99": percentile(token_lengths, 99),
            "max": int(max(token_lengths)),
        },
    }
    if dataset_type == "feedback":
        stats["feedback_rows"] = sum(1 for row in rows if row.get("dataset_type") == "feedback")
    return stats


class ClevrZarrWriter:
    def __init__(
        self,
        output,
        dataset_root,
        dataset_type,
        vlm_model,
        max_context_len,
        dtype,
        image_size,
        context_dim,
        overwrite=False,
        token_chunk_length=512,
        row_chunk_length=4096,
    ):
        self.output = Path(output)
        self.dataset_root = Path(dataset_root)
        self.dataset_type = dataset_type
        self.vlm_model = vlm_model
        self.max_context_len = int(max_context_len)
        self.dtype = dtype
        self.image_size = int(image_size)
        self.context_dim = int(context_dim)
        self.row_chunk_length = int(row_chunk_length)
        self.split_to_id = {}
        self.template_to_id = {}
        self.image_to_id = {}
        self.generated_image_to_id = {}
        self.rows = []
        self.token_lengths = []

        if self.output.exists():
            if not overwrite:
                raise FileExistsError(f"{self.output} already exists. Pass --overwrite to replace it.")
            shutil.rmtree(self.output)

        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        root = zarr.open(str(self.output), mode="w")
        data = root.require_group("data")
        meta = root.require_group("meta")
        root.attrs.update({
            "format_version": 1,
            "dataset_type": dataset_type,
            "vlm_model": vlm_model,
            "context_dim": self.context_dim,
            "context_dtype": dtype,
            "max_context_len": self.max_context_len,
            "image_size": self.image_size,
            "rows": 0,
        })
        self.root = root
        self.data = data
        self.meta = meta
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
            chunks=(self.row_chunk_length,),
            dtype=np.int64,
            compressor=None,
            overwrite=True,
        )
        self.context_offsets[0] = 0
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

        self.row_arrays = {
            "image_index": data.create_dataset("image_index", shape=(0,), chunks=(self.row_chunk_length,), dtype=np.int64, compressor=compressor, overwrite=True),
            "generated_image_index": data.create_dataset("generated_image_index", shape=(0,), chunks=(self.row_chunk_length,), dtype=np.int64, compressor=compressor, overwrite=True),
            "split_id": meta.create_dataset("split_id", shape=(0,), chunks=(self.row_chunk_length,), dtype=np.int8, compressor=compressor, overwrite=True),
            "template_id": meta.create_dataset("template_id", shape=(0,), chunks=(self.row_chunk_length,), dtype=np.int16, compressor=compressor, overwrite=True),
            "is_feedback": meta.create_dataset("is_feedback", shape=(0,), chunks=(self.row_chunk_length,), dtype=bool, compressor=compressor, overwrite=True),
            "metadata_index": meta.create_dataset("metadata_index", shape=(0,), chunks=(self.row_chunk_length,), dtype=np.int64, compressor=compressor, overwrite=True),
            "sample_index": meta.create_dataset("sample_index", shape=(0,), chunks=(self.row_chunk_length,), dtype=np.int64, compressor=compressor, overwrite=True),
            "feedback_index": meta.create_dataset("feedback_index", shape=(0,), chunks=(self.row_chunk_length,), dtype=np.int64, compressor=compressor, overwrite=True),
            "caption": meta.create_dataset("caption", shape=(0,), chunks=(self.row_chunk_length,), dtype=object, object_codec=VLenUTF8(), compressor=compressor, overwrite=True),
            "feedback": meta.create_dataset("feedback", shape=(0,), chunks=(self.row_chunk_length,), dtype=object, object_codec=VLenUTF8(), compressor=compressor, overwrite=True),
            "tuple_id": meta.create_dataset("tuple_id", shape=(0,), chunks=(self.row_chunk_length,), dtype=object, object_codec=VLenUTF8(), compressor=compressor, overwrite=True),
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

    def append_batch(self, rows, tokens):
        if len(rows) != len(tokens):
            raise ValueError(f"rows/tokens length mismatch: {len(rows)} != {len(tokens)}")
        if not rows:
            return

        lengths = [int(token.shape[0]) for token in tokens]
        if any(int(token.shape[-1]) != self.context_dim for token in tokens):
            raise ValueError("All context tokens must match context_dim.")

        start_row = self.row_arrays["image_index"].shape[0]
        end_row = start_row + len(rows)
        flat_start = int(self.context_offsets[start_row])
        flat_end = flat_start + sum(lengths)

        self.context_tokens.resize((flat_end, self.context_dim))
        self.context_tokens[flat_start:flat_end] = np.concatenate(
            [np.asarray(token, dtype=self.dtype) for token in tokens],
            axis=0,
        )
        self.context_offsets.resize((end_row + 1,))
        offsets = np.empty(len(rows) + 1, dtype=np.int64)
        offsets[0] = flat_start
        np.cumsum(np.asarray(lengths, dtype=np.int64), out=offsets[1:])
        offsets[1:] += flat_start
        self.context_offsets[start_row:end_row + 1] = offsets

        values = {key: [] for key in self.row_arrays}
        for row in rows:
            image_path = row.get("gt_image_path", row.get("image_path"))
            image_path = resolve_path(self.dataset_root, image_path)
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
            values["template_id"].append(self._id_for(self.template_to_id, row.get("template", "")))
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

        self.token_lengths.extend(lengths)
        self.root.attrs["rows"] = end_row
        self.root.attrs["split_names"] = [name for name, _ in sorted(self.split_to_id.items(), key=lambda item: item[1])]
        self.root.attrs["template_names"] = [name for name, _ in sorted(self.template_to_id.items(), key=lambda item: item[1])]

    def stats(self):
        return build_stats(
            dataset=self.dataset_root,
            output=self.output,
            dataset_type=self.dataset_type,
            vlm_model=self.vlm_model,
            rows=self.rows,
            context_dim=self.context_dim,
            max_context_len=self.max_context_len,
            dtype=self.dtype,
            image_size=self.image_size,
            token_lengths=self.token_lengths,
            split_names=self.root.attrs.get("split_names", []),
            template_names=self.root.attrs.get("template_names", []),
        )
