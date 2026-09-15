"""Online rollout dataset: per-rank record shards written by the on-policy RolloutCollector."""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class RolloutBuffer(Dataset):
    """Merges per-rank records.pt shards written by RolloutCollector.

    Each record stores the RAW interleaved history that produced an accepted attempt: the caption,
    the feedbacks and attempt-image paths of all prior attempts, plus the GT path. The model turns
    this history into a loss (QwenDiT encodes it; OmniGen builds a multi-image edit example), so the
    buffer is model-agnostic and stores no encoded context.
    """

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
        return {
            "gt_image": Image.open(record["gt_path"]).convert("RGB"),
            "caption": record["caption"],
            "feedback_history": list(record["feedback_history"]),
            "attempt_images": [Image.open(path).convert("RGB") for path in record["attempt_paths"]],
            "attempt_paths": list(record["attempt_paths"]),
            # passthrough for logging (log_rollout_samples reads paths, not the PILs)
            "gt_path": record["gt_path"],
            "attempt_path": record["attempt_path"],
            "feedback": record["feedback"],
            # initial sampling latent that produced this attempt (None for older buffers)
            "init_latent": (torch.load(record["latent_path"], map_location="cpu", weights_only=False)
                            if record.get("latent_path") else None),
        }


def rollout_collate(batch):
    """Group rollout rows into per-field lists; the model's rollout_loss does tensor conversion."""
    keys = ("gt_image", "caption", "feedback_history", "attempt_images", "attempt_paths")
    out = {key: [item[key] for item in batch] for key in keys}
    latents = [item.get("init_latent") for item in batch]
    out["init_latents"] = latents if any(l is not None for l in latents) else None
    # Position of each row within its chain: 0 is the first draft (caption only), t>0 are the
    # feedback-conditioned repairs. The curriculum weights the loss by this, and caption dropout
    # keys off it. Derived here rather than stored, so the record schema is unchanged.
    out["chain_pos"] = [len(item["feedback_history"]) for item in batch]
    return out
