import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class ClevrTextEmbeddingDataset(Dataset):
    def __init__(self, root, transform=None, split="train"):
        self.root = Path(root)
        self.transform = transform
        self.embedding_root = self.root / "text_embeddings"
        with (self.embedding_root / "index.json").open() as f:
            index = json.load(f)
        self.max_length = index["max_length"]
        self.embedding_dim = index["embedding_dim"]
        self.records = [record for record in index["records"] if split is None or record["split"] == split]
        self._shard_cache = {}

    def __len__(self):
        return len(self.records)

    def load_shard(self, shard_name):
        if shard_name not in self._shard_cache:
            self._shard_cache[shard_name] = torch.load(self.embedding_root / shard_name, map_location="cpu")
        return self._shard_cache[shard_name]

    def __getitem__(self, idx):
        record = self.records[idx]
        image = Image.open(self.root / record["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        shard = self.load_shard(record["shard"])
        offset = record["offset"]
        return {
            "image": image,
            "text_tokens": shard["text_tokens"][offset].float(),
            "text_mask": shard["text_mask"][offset],
            "text_pooled": shard["text_pooled"][offset].float(),
            "caption": record["caption"],
            "image_path": record["image_path"],
        }
