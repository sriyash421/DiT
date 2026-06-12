import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


def resolve_path(root, path):
    path = Path(path)
    if path.is_absolute():
        return path
    rooted = Path(root) / path
    if rooted.exists():
        return rooted
    return path


class ClevrTextEmbeddingDataset(Dataset):
    def __init__(self, root, transform=None, split="train"):
        self.root = Path(root)
        self.transform = transform
        self.embedding_root = self.root / "text_embeddings"
        with (self.embedding_root / "index.json").open() as f:
            index = json.load(f)
        self.max_length = index["max_length"]
        self.embedding_dim = index["embedding_dim"]
        self.dataset_type = index.get("dataset_type", "base")
        self.records = [record for record in index["records"] if split is None or record.get("split") == split]
        self._shard_cache = {}

    def __len__(self):
        return len(self.records)

    def load_shard(self, shard_name):
        if shard_name not in self._shard_cache:
            self._shard_cache[shard_name] = torch.load(self.embedding_root / shard_name, map_location="cpu", weights_only=False)
        return self._shard_cache[shard_name]

    def __getitem__(self, idx):
        record = self.records[idx]
        shard = self.load_shard(record["shard"])
        offset = int(record["offset"])
        dataset_type = record.get("dataset_type", self.dataset_type)
        image_path = record.get("gt_image_path", record.get("image_path"))
        image = Image.open(resolve_path(self.root, image_path)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        out = {
            "image": image,
            "text_tokens": shard["text_tokens"][offset].float(),
            "text_mask": shard["text_mask"][offset].bool(),
            "text_pooled": shard["text_pooled"][offset].float(),
            "caption": record.get("caption", ""),
            "image_path": str(image_path),
            "dataset_type": dataset_type,
            "has_feedback": dataset_type == "feedback",
            "has_image_context": dataset_type == "feedback",
        }
        if dataset_type == "feedback":
            out.update({
                "feedback_tokens": shard["feedback_tokens"][offset].float(),
                "feedback_mask": shard["feedback_mask"][offset].bool(),
                "feedback_pooled": shard["feedback_pooled"][offset].float(),
                "attempt_latent": shard["attempt_latent"][offset].float(),
                "feedback": record.get("feedback", ""),
                "generated_image_path": record.get("generated_image_path", ""),
            })
        return out


class WeightedDatasetList(Dataset):
    def __init__(self, datasets, sampling_ratios, virtual_epoch_size=None, seed=0):
        assert len(datasets) == len(sampling_ratios)
        self.datasets = datasets
        ratios = torch.tensor(sampling_ratios, dtype=torch.float64)
        self.probs = (ratios / ratios.sum()).tolist()
        if virtual_epoch_size is None:
            virtual_epoch_size = max(len(dataset) for dataset in datasets)
        self.virtual_epoch_size = int(virtual_epoch_size)
        self.seed = int(seed)
        self._map = []
        rng = random.Random(self.seed)
        for _ in range(self.virtual_epoch_size):
            source = rng.choices(range(len(self.datasets)), weights=self.probs, k=1)[0]
            index = rng.randrange(len(self.datasets[source]))
            self._map.append((source, index))

    def __len__(self):
        return self.virtual_epoch_size

    def __getitem__(self, idx):
        source, inner_idx = self._map[idx]
        item = self.datasets[source][inner_idx]
        item["source_index"] = source
        return item


def adaptive_collate(batch):
    base_items = [item for item in batch if not item.get("has_feedback", False)]
    feedback_items = [item for item in batch if item.get("has_feedback", False)]

    def stack_common(items):
        if not items:
            return None
        return {
            "image": torch.stack([item["image"] for item in items]),
            "text_tokens": torch.stack([item["text_tokens"] for item in items]),
            "text_mask": torch.stack([item["text_mask"] for item in items]),
            "text_pooled": torch.stack([item["text_pooled"] for item in items]),
            "source_index": torch.tensor([item.get("source_index", 0) for item in items], dtype=torch.long),
        }

    out = {"base": stack_common(base_items), "feedback": stack_common(feedback_items)}
    if feedback_items:
        out["feedback"].update({
            "feedback_tokens": torch.stack([item["feedback_tokens"] for item in feedback_items]),
            "feedback_mask": torch.stack([item["feedback_mask"] for item in feedback_items]),
            "feedback_pooled": torch.stack([item["feedback_pooled"] for item in feedback_items]),
            "attempt_latent": torch.stack([item["attempt_latent"] for item in feedback_items]),
        })
    return out
