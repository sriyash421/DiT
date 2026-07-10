"""CLEVR zarr datasets, weighted distributed sampler, and context collation."""
import math
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import zarr
from PIL import Image
from torch.utils.data import Dataset, Sampler


class ClevrContextDataset(Dataset):
    """Reads one data.zarr store with images, captions, and cached frozen-Qwen context tokens."""

    def __init__(
        self,
        root,
        transform=None,
        split="train",
        use_disk=True,
        load_meta=True,
        load_images=False,
        load_context=False,
        max_dataset_size=None,
        return_generated_images=False,
    ):
        self.root = Path(root)
        self.transform = transform
        self.split = split
        self.return_generated_images = bool(return_generated_images)
        self.use_disk = bool(use_disk)
        self.load_meta = bool(load_meta)
        self.load_images = bool(load_images)
        self.load_context = bool(load_context)
        self.max_dataset_size = None if max_dataset_size is None else int(max_dataset_size)
        if self.max_dataset_size is not None and self.max_dataset_size <= 0:
            raise ValueError("max_dataset_size must be positive or None.")
        self._zarr = None
        self._data = None
        self._meta = None
        self._context_cache = None
        self._image_cache = None
        self._generated_image_cache = None
        self._open()

        self.dataset_type = self._zarr.attrs["dataset_type"]
        self.context_dim = int(self._zarr.attrs["context_dim"]) if self.has_context else None
        self.max_context_len = int(self._zarr.attrs["max_context_len"]) if self.has_context else 0
        self.split_names = list(self._zarr.attrs["split_names"])
        self.indices = self._select_indices(np.asarray(self._meta["split_id"][:]), self.split_names)

    def _select_indices(self, split_ids, split_names):
        if self.split is None:
            indices = np.arange(split_ids.shape[0], dtype=np.int64)
        elif self.split not in split_names:
            indices = np.zeros((0,), dtype=np.int64)
        else:
            indices = np.flatnonzero(split_ids == split_names.index(self.split)).astype(np.int64)
        if self.max_dataset_size is not None:
            indices = indices[:self.max_dataset_size]
        return indices

    def _open(self):
        root = zarr.open(str(self.root), mode="r")
        data = root["data"]
        meta = root["meta"]
        self._zarr = root
        self.has_context = "context_offsets" in data
        keep_context_on_disk = (self.use_disk and not self.load_context) or self.max_dataset_size is not None
        keep_images_on_disk = (self.use_disk and not self.load_images) or self.max_dataset_size is not None
        self._data = {
            "image_index": data["image_index"][:],
            "generated_image_index": data["generated_image_index"][:],
            "images": data["images"] if keep_images_on_disk else data["images"][:],
            "generated_images": data["generated_images"] if keep_images_on_disk else data["generated_images"][:],
        }
        if self.has_context:
            self._data["context_offsets"] = data["context_offsets"][:]
            self._data["context_tokens"] = data["context_tokens"] if keep_context_on_disk else data["context_tokens"][:]
        if self.load_meta:
            self._meta = {key: meta[key][:] for key in meta.keys()}
        else:
            self._meta = {key: meta[key] for key in meta.keys()}
        row_indices = self._select_indices(np.asarray(meta["split_id"][:]), list(root.attrs["split_names"]))
        self._build_limited_caches(data, row_indices)

    def _build_limited_caches(self, data, row_indices):
        if self.max_dataset_size is None:
            return
        if self.load_context and self.has_context:
            offsets = self._data["context_offsets"]
            self._context_cache = {}
            for row_idx in row_indices:
                start = int(offsets[row_idx])
                end = int(offsets[row_idx + 1])
                self._context_cache[int(row_idx)] = np.asarray(data["context_tokens"][start:end])
        if self.load_images:
            image_ids = np.asarray(self._data["image_index"][row_indices], dtype=np.int64)
            self._image_cache = {int(image_id): np.asarray(data["images"][int(image_id)]) for image_id in np.unique(image_ids)}
            generated_ids = np.asarray(self._data["generated_image_index"][row_indices], dtype=np.int64)
            generated_ids = generated_ids[generated_ids >= 0]
            self._generated_image_cache = {
                int(image_id): np.asarray(data["generated_images"][int(image_id)])
                for image_id in np.unique(generated_ids)
            }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_zarr"] = None
        state["_data"] = None
        state["_meta"] = None
        state["_context_cache"] = None
        state["_image_cache"] = None
        state["_generated_image_cache"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open()

    def __len__(self):
        return int(self.indices.shape[0])

    def set_transform(self, transform):
        self.transform = transform

    def set_return_generated_images(self, flag):
        self.return_generated_images = bool(flag)

    def _meta_value(self, key, row_idx):
        value = self._meta[key][row_idx]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if hasattr(value, "item"):
            value = value.item()
        return value

    def context_for_row(self, row_idx, dtype=None):
        """Context tokens for a row, kept in the stored dtype (float16) unless dtype is given."""
        if self._context_cache is not None and int(row_idx) in self._context_cache:
            arr = np.asarray(self._context_cache[int(row_idx)])
        else:
            start = int(self._data["context_offsets"][row_idx])
            end = int(self._data["context_offsets"][row_idx + 1])
            arr = np.asarray(self._data["context_tokens"][start:end])
        if dtype is not None:
            arr = arr.astype(dtype)
        return torch.from_numpy(arr)

    def image_for_row(self, row_idx):
        image_idx = int(self._data["image_index"][row_idx])
        if self._image_cache is not None and image_idx in self._image_cache:
            return Image.fromarray(np.asarray(self._image_cache[image_idx]), mode="RGB")
        return Image.fromarray(np.asarray(self._data["images"][image_idx]), mode="RGB")

    def generated_image_for_row(self, row_idx):
        generated_idx = int(self._data["generated_image_index"][row_idx])
        if generated_idx < 0:
            return None
        if self._generated_image_cache is not None and generated_idx in self._generated_image_cache:
            return Image.fromarray(np.asarray(self._generated_image_cache[generated_idx]), mode="RGB")
        return Image.fromarray(np.asarray(self._data["generated_images"][generated_idx]), mode="RGB")

    def image_for_index(self, idx):
        return self.image_for_row(int(self.indices[int(idx)]))

    def generated_image_for_index(self, idx):
        return self.generated_image_for_row(int(self.indices[int(idx)]))

    def record_for_index(self, idx):
        row_idx = int(self.indices[int(idx)])
        split_id = int(self._meta_value("split_id", row_idx))
        generated_idx = int(self._data["generated_image_index"][row_idx])
        return {
            "row_idx": row_idx,
            "split": self.split_names[split_id] if 0 <= split_id < len(self.split_names) else "",
            "caption": str(self._meta_value("caption", row_idx)),
            "feedback": str(self._meta_value("feedback", row_idx)),
            "metadata_index": int(self._meta_value("metadata_index", row_idx)),
            "sample_index": int(self._meta_value("sample_index", row_idx)),
            "feedback_index": int(self._meta_value("feedback_index", row_idx)),
            "tuple_id": str(self._meta_value("tuple_id", row_idx)),
            "image_path": f"images/{int(self._data['image_index'][row_idx])}",
            "generated_image_path": "" if generated_idx < 0 else f"generated_images/{generated_idx}",
        }

    def __getitem__(self, idx):
        row_idx = int(self.indices[int(idx)])
        image = self.image_for_row(row_idx)
        if self.transform is not None:
            image = self.transform(image)
        record = self.record_for_index(idx)
        is_feedback = bool(self._meta_value("is_feedback", row_idx))
        item = {
            "image": image,
            "caption": record["caption"],
            "feedback": record["feedback"],
            "metadata_index": record["metadata_index"],
            "sample_index": record["sample_index"],
            "feedback_index": record["feedback_index"],
            "tuple_id": record["tuple_id"],
            "image_path": record["image_path"],
            "generated_image_path": record["generated_image_path"],
            "is_feedback": is_feedback,
        }
        if self.has_context:
            item["context_tokens"] = self.context_for_row(row_idx)
        if self.return_generated_images:
            item["generated_image"] = self.generated_image_for_row(row_idx)
        return item


class ClevrContextMultiDataset(Dataset):
    """Concatenates ClevrContextDatasets with per-source sampling weights."""

    def __init__(
        self,
        entries,
        transform=None,
        split="train",
        use_disk=True,
        load_meta=True,
        load_images=False,
        load_context=False,
    ):
        if not entries:
            raise ValueError("entries must contain at least one dataset.")
        self.datasets = []
        self.names = []
        self.sampling_ratios = []
        for entry in entries:
            dataset = ClevrContextDataset(
                entry["path"],
                transform=transform,
                split=split,
                use_disk=use_disk,
                load_meta=load_meta,
                load_images=load_images,
                load_context=load_context,
                max_dataset_size=entry["max_dataset_size"],
            )
            if len(dataset) == 0:
                raise ValueError(f"Dataset has no rows for split={split}: {entry['path']}")
            self.datasets.append(dataset)
            self.names.append(entry["name"])
            self.sampling_ratios.append(float(entry["sampling_ratio"]))
        if any(ratio <= 0 for ratio in self.sampling_ratios):
            raise ValueError("All sampling_ratio values must be positive.")
        self.context_dim = self.datasets[0].context_dim
        self.max_context_len = max(dataset.max_context_len for dataset in self.datasets)
        if any(dataset.context_dim != self.context_dim for dataset in self.datasets):
            raise ValueError("All datasets must use the same context_dim.")
        lengths = [len(dataset) for dataset in self.datasets]
        self.cumulative_lengths = torch.tensor([0] + lengths, dtype=torch.long).cumsum(0)
        self.cumulative_lengths_list = self.cumulative_lengths.tolist()
        self.total_length = int(self.cumulative_lengths[-1].item())
        self.sample_weights = self._build_sample_weights()

    def _build_sample_weights(self):
        ratios = torch.tensor(self.sampling_ratios, dtype=torch.float64)
        ratios = ratios / ratios.sum()
        weights = []
        for ratio, dataset in zip(ratios, self.datasets):
            weights.append(torch.full((len(dataset),), float(ratio / len(dataset)), dtype=torch.float32))
        return torch.cat(weights)

    def __len__(self):
        return self.total_length

    def set_transform(self, transform):
        for dataset in self.datasets:
            dataset.set_transform(transform)

    def set_return_generated_images(self, flag):
        for dataset in self.datasets:
            dataset.set_return_generated_images(flag)

    def global_to_local(self, idx):
        idx = int(idx)
        dataset_idx = bisect_right(self.cumulative_lengths_list, idx) - 1
        local_idx = idx - int(self.cumulative_lengths[dataset_idx].item())
        return dataset_idx, local_idx

    def image_for_index(self, idx):
        dataset_idx, local_idx = self.global_to_local(idx)
        return self.datasets[dataset_idx].image_for_index(local_idx)

    def generated_image_for_index(self, idx):
        dataset_idx, local_idx = self.global_to_local(idx)
        return self.datasets[dataset_idx].generated_image_for_index(local_idx)

    def __getitem__(self, idx):
        dataset_idx, local_idx = self.global_to_local(idx)
        item = self.datasets[dataset_idx][local_idx]
        item["source_index"] = dataset_idx
        item["source_name"] = self.names[dataset_idx]
        return item


class DistributedWeightedSampler(Sampler):
    """Per-rank multinomial sampling from per-row weights."""

    def __init__(self, weights, num_replicas=None, rank=None, replacement=True, seed=0):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.weights) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=self.replacement,
            generator=generator,
        ).tolist()
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


def pad_contexts(contexts):
    """Pad variable-length context token sequences to (B, L, dim) plus a bool mask."""
    if not contexts:
        raise ValueError("Cannot pad an empty context batch.")
    max_len = max(int(context.shape[0]) for context in contexts)
    dim = int(contexts[0].shape[-1])
    dtype = contexts[0].dtype
    device = contexts[0].device
    tokens = torch.zeros(len(contexts), max_len, dim, dtype=dtype, device=device)
    mask = torch.zeros(len(contexts), max_len, dtype=torch.bool, device=device)
    for idx, context in enumerate(contexts):
        length = int(context.shape[0])
        if int(context.shape[-1]) != dim:
            raise ValueError(f"Context dim mismatch at item {idx}: expected {dim}, got {context.shape[-1]}")
        tokens[idx, :length] = context
        mask[idx, :length] = True
    return tokens, mask


def context_collate(batch):
    out = {
        "image": torch.stack([item["image"] for item in batch]),
        "is_feedback": torch.tensor([item["is_feedback"] for item in batch], dtype=torch.bool),
        "source_index": torch.tensor([item.get("source_index", 0) for item in batch], dtype=torch.long),
        "source_name": [item.get("source_name", "") for item in batch],
        "caption": [item["caption"] for item in batch],
        "feedback": [item["feedback"] for item in batch],
        "metadata_index": torch.tensor([item["metadata_index"] for item in batch], dtype=torch.long),
        "sample_index": torch.tensor([item["sample_index"] for item in batch], dtype=torch.long),
        "feedback_index": torch.tensor([item["feedback_index"] for item in batch], dtype=torch.long),
        "tuple_id": [item["tuple_id"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "generated_image_path": [item["generated_image_path"] for item in batch],
    }
    if "context_tokens" in batch[0]:
        out["context_tokens"], out["context_mask"] = pad_contexts([item["context_tokens"] for item in batch])
    if "generated_image" in batch[0]:
        out["generated_image"] = [item["generated_image"] for item in batch]
    return out
