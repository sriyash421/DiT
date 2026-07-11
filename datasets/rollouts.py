"""Online rollout dataset: per-rank record shards written by the on-policy RolloutCollector."""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.clevr.dataset import pad_contexts


class RolloutBuffer(Dataset):
    """Merges per-rank records.pt shards written by RolloutCollector.

    Frozen-encoder records already carry their context embeddings. For a LoRA encoder,
    prepare_fn tokenizes each record's history once (reused across updates); the VLM
    forward runs per update in the trainer.
    """

    def __init__(self, path, prepare_fn=None):
        self.path = Path(path)
        self.prepare_fn = prepare_fn
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
        if "context_tokens" not in record and "context_inputs" not in record and self.prepare_fn is not None:
            images = [Image.open(path).convert("RGB") for path in record["history_attempt_paths"]]
            record["context_inputs"] = self.prepare_fn(record["caption"], record["feedback_history"], images)
        return record


def rollout_collate(batch):
    out = {"x_latent": torch.stack([item["x_latent"] for item in batch])}
    if "context_tokens" in batch[0]:
        out["context_tokens"], out["context_mask"] = pad_contexts([item["context_tokens"] for item in batch])
    elif "context_inputs" in batch[0]:
        out["context_inputs"] = [item["context_inputs"] for item in batch]
    return out
