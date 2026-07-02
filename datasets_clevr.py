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
    ):
        self.root = Path(root)
        self.transform = transform
        self.split = split
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
        self.context_dim = int(self._zarr.attrs["context_dim"])
        self.max_context_len = int(self._zarr.attrs["max_context_len"])
        self.split_names = list(self._zarr.attrs.get("split_names", []))
        self.template_names = list(self._zarr.attrs.get("template_names", []))

        split_ids = np.asarray(self._meta["split_id"])
        self.indices = self._select_indices(split_ids)

    def _select_indices(self, split_ids):
        if self.split is None:
            indices = np.arange(split_ids.shape[0], dtype=np.int64)
        else:
            if self.split not in self.split_names:
                indices = np.zeros((0,), dtype=np.int64)
            else:
                indices = np.flatnonzero(split_ids == self.split_names.index(self.split)).astype(np.int64)
        if self.max_dataset_size is not None:
            indices = indices[:self.max_dataset_size]
        return indices

    def _open(self):
        root = zarr.open(str(self.root), mode="r")
        data = root["data"]
        meta = root["meta"]

        self._zarr = root
        split_names = list(root.attrs.get("split_names", []))
        split_ids = np.asarray(meta["split_id"][:])
        row_indices = self._select_indices_from(split_ids, split_names)
        self._data = {
            "context_offsets": data["context_offsets"][:],
            "image_index": data["image_index"][:],
            "generated_image_index": data["generated_image_index"][:],
            "context_tokens": data["context_tokens"] if (self.use_disk and not self.load_context) or self.max_dataset_size is not None else data["context_tokens"][:],
            "images": data["images"] if (self.use_disk and not self.load_images) or self.max_dataset_size is not None else data["images"][:],
            "generated_images": data["generated_images"] if (self.use_disk and not self.load_images) or self.max_dataset_size is not None else data["generated_images"][:],
        }
        if self.load_meta:
            self._meta = {key: value[:] for key, value in meta.items()}
        else:
            self._meta = {key: value for key, value in meta.items()}
        self._build_limited_caches(data, row_indices)

    def _select_indices_from(self, split_ids, split_names):
        if self.split is None:
            indices = np.arange(split_ids.shape[0], dtype=np.int64)
        elif self.split not in split_names:
            indices = np.zeros((0,), dtype=np.int64)
        else:
            indices = np.flatnonzero(split_ids == split_names.index(self.split)).astype(np.int64)
        if self.max_dataset_size is not None:
            indices = indices[:self.max_dataset_size]
        return indices

    def _build_limited_caches(self, data, row_indices):
        if self.max_dataset_size is None:
            return
        if self.load_context:
            offsets = self._data["context_offsets"]
            self._context_cache = {}
            for row_idx in row_indices:
                start = int(offsets[row_idx])
                end = int(offsets[row_idx + 1])
                self._context_cache[int(row_idx)] = np.asarray(data["context_tokens"][start:end])
            self._data["context_tokens"] = data["context_tokens"]
        if self.load_images:
            image_ids = np.asarray(self._data["image_index"][row_indices], dtype=np.int64)
            self._image_cache = {int(image_id): np.asarray(data["images"][int(image_id)]) for image_id in np.unique(image_ids)}
            generated_ids = np.asarray(self._data["generated_image_index"][row_indices], dtype=np.int64)
            generated_ids = generated_ids[generated_ids >= 0]
            self._generated_image_cache = {
                int(image_id): np.asarray(data["generated_images"][int(image_id)])
                for image_id in np.unique(generated_ids)
            }
            self._data["images"] = data["images"]
            self._data["generated_images"] = data["generated_images"]

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

    def _meta_value(self, key, idx, default=""):
        arr = self._meta.get(key)
        if arr is None:
            return default
        value = arr[idx]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if hasattr(value, "item"):
            value = value.item()
        return value

    def context_for_row(self, row_idx, dtype=np.float32):
        if self._context_cache is not None and int(row_idx) in self._context_cache:
            return torch.from_numpy(np.asarray(self._context_cache[int(row_idx)], dtype=dtype))
        start = int(self._data["context_offsets"][row_idx])
        end = int(self._data["context_offsets"][row_idx + 1])
        return torch.from_numpy(np.asarray(self._data["context_tokens"][start:end], dtype=dtype))

    def image_for_row(self, row_idx):
        image_idx = int(self._data["image_index"][row_idx])
        if self._image_cache is not None and image_idx in self._image_cache:
            return Image.fromarray(np.asarray(self._image_cache[image_idx]), mode="RGB")
        return Image.fromarray(np.asarray(self._data["images"][image_idx]), mode="RGB")

    def record_for_index(self, idx):
        row_idx = int(self.indices[int(idx)])
        template_id = int(self._meta_value("template_id", row_idx, -1))
        split_id = int(self._meta_value("split_id", row_idx, -1))
        template = self.template_names[template_id] if 0 <= template_id < len(self.template_names) else ""
        split = self.split_names[split_id] if 0 <= split_id < len(self.split_names) else ""
        return {
            "row_idx": row_idx,
            "split": split,
            "caption": str(self._meta_value("caption", row_idx, "")),
            "feedback": str(self._meta_value("feedback", row_idx, "")),
            "metadata_index": int(self._meta_value("metadata_index", row_idx, -1)),
            "template": template,
            "sample_index": int(self._meta_value("sample_index", row_idx, -1)),
            "feedback_index": int(self._meta_value("feedback_index", row_idx, -1)),
            "tuple_id": str(self._meta_value("tuple_id", row_idx, "")),
            "image_path": f"images/{int(self._data['image_index'][row_idx])}",
            "generated_image_path": "" if int(self._data["generated_image_index"][row_idx]) < 0 else f"generated_images/{int(self._data['generated_image_index'][row_idx])}",
        }

    def __getitem__(self, idx):
        row_idx = int(self.indices[int(idx)])
        context_tokens = self.context_for_row(row_idx)
        image = self.image_for_row(row_idx)
        if self.transform is not None:
            image = self.transform(image)

        generated_idx = int(self._data["generated_image_index"][row_idx])
        dataset_type = "feedback" if bool(self._meta_value("is_feedback", row_idx, False)) else "base"
        record = self.record_for_index(idx)
        return {
            "image": image,
            "context_tokens": context_tokens,
            "caption": record["caption"],
            "feedback": record["feedback"],
            "metadata_index": record["metadata_index"],
            "template": record["template"],
            "sample_index": record["sample_index"],
            "feedback_index": record["feedback_index"],
            "tuple_id": record["tuple_id"],
            "image_path": record["image_path"],
            "generated_image_path": "" if generated_idx < 0 else f"generated_images/{generated_idx}",
            "dataset_type": dataset_type,
            "is_feedback": dataset_type == "feedback",
        }


class ClevrContextMultiDataset(Dataset):
    def __init__(
        self,
        dataset_config,
        transform=None,
        split="train",
        use_disk=True,
        load_meta=True,
        load_images=False,
        load_context=False,
        max_dataset_size=None,
    ):
        if not dataset_config:
            raise ValueError("dataset_config must contain at least one dataset.")
        self.dataset_config = [dict(entry) for entry in dataset_config]
        self.datasets = []
        self.names = []
        self.sampling_ratios = []
        for idx, entry in enumerate(self.dataset_config):
            path = entry.get("dataset_path")
            if path is None:
                raise ValueError(f"dataset_config[{idx}] is missing dataset_path")
            dataset = ClevrContextDataset(
                path,
                transform=transform,
                split=split,
                use_disk=use_disk,
                load_meta=load_meta,
                load_images=load_images,
                load_context=load_context,
                max_dataset_size=entry.get("max_dataset_size", max_dataset_size),
            )
            if len(dataset) == 0:
                raise ValueError(f"dataset_config[{idx}] has no rows for split={split}: {path}")
            self.datasets.append(dataset)
            self.names.append(entry.get("name", f"dataset_{idx}"))
            self.sampling_ratios.append(float(entry.get("sampling_ratio", 1.0)))
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

    def global_to_local(self, idx):
        idx = int(idx)
        dataset_idx = bisect_right(self.cumulative_lengths_list, idx) - 1
        local_idx = idx - int(self.cumulative_lengths[dataset_idx].item())
        return dataset_idx, local_idx

    def __getitem__(self, idx):
        dataset_idx, local_idx = self.global_to_local(idx)
        item = self.datasets[dataset_idx][local_idx]
        item["source_index"] = dataset_idx
        item["source_name"] = self.names[dataset_idx]
        return item


class DistributedWeightedSampler(Sampler):
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
    context_tokens, context_mask = pad_contexts([item["context_tokens"] for item in batch])
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "context_tokens": context_tokens,
        "context_mask": context_mask,
        "is_feedback": torch.tensor([item["is_feedback"] for item in batch], dtype=torch.bool),
        "source_index": torch.tensor([item.get("source_index", 0) for item in batch], dtype=torch.long),
        "source_name": [item.get("source_name", f"dataset_{item.get('source_index', 0)}") for item in batch],
        "caption": [item["caption"] for item in batch],
        "feedback": [item["feedback"] for item in batch],
        "metadata_index": torch.tensor([item.get("metadata_index", -1) for item in batch], dtype=torch.long),
        "template": [item.get("template", "") for item in batch],
        "sample_index": torch.tensor([item.get("sample_index", -1) for item in batch], dtype=torch.long),
        "feedback_index": torch.tensor([item.get("feedback_index", -1) for item in batch], dtype=torch.long),
        "tuple_id": [item.get("tuple_id", "") for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "generated_image_path": [item["generated_image_path"] for item in batch],
    }
