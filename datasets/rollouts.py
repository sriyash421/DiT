"""Online rollout dataset: per-rank record shards written by the on-policy RolloutCollector."""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.clevr.dataset import pad_contexts


class RolloutBuffer(Dataset):
    """Merges per-rank records.pt shards written by RolloutCollector."""

    def __init__(self, path):
        self.path = Path(path)
        payloads = self._load_payloads(self.path)
        self.records = []
        self.stats = {"attempted": 0, "success": 0, "failed": 0, "gemini_tokens": 0}
        for payload in payloads:
            self.records.extend(payload["records"])
            for key, value in payload.get("stats", {}).items():
                if key in self.stats:
                    self.stats[key] += value
        self.stats["success"] = len(self.records)

    @staticmethod
    def _load_payloads(path):
        if (path / "records.pt").exists():
            return [torch.load(path / "records.pt", map_location="cpu", weights_only=False)]
        record_paths = sorted(path.glob("rank_*/records.pt"))
        if not record_paths:
            raise FileNotFoundError(f"No rollout records found under {path}")
        return [torch.load(record_path, map_location="cpu", weights_only=False) for record_path in record_paths]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[int(idx)]
        if "history_attempt_images" not in record:
            record["history_attempt_images"] = [
                Image.open(path).convert("RGB") for path in record["history_attempt_paths"]
            ]
        return record


def rollout_collate(batch):
    out = {
        "x_latent": torch.stack([item["x_latent"] for item in batch]),
        "caption": [item["caption"] for item in batch],
        "feedback": [item["feedback"] for item in batch],
        "feedback_history": [item["feedback_history"] for item in batch],
        "history_attempt_paths": [item["history_attempt_paths"] for item in batch],
        "history_attempt_images": [item["history_attempt_images"] for item in batch],
        "attempt_path": [item["attempt_path"] for item in batch],
    }
    if "context_tokens" in batch[0]:
        out["context_tokens"], out["context_mask"] = pad_contexts([item["context_tokens"] for item in batch])
    return out
