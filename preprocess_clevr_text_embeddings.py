"""
Precompute frozen FLAN-T5 embeddings for CLEVR caption and feedback datasets.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers.models import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, T5EncoderModel

from clevr_captions import render_caption


def resolve_path(root, path):
    path = Path(path)
    if path.is_absolute():
        return path
    rooted = Path(root) / path
    if rooted.exists():
        return rooted
    return path


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def image_to_tensor(path, image_size):
    image = Image.open(path).convert("RGB")
    image = center_crop_arr(image, image_size)
    arr = np.asarray(image).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)


class CaptionRows(Dataset):
    def __init__(self, metadata_path, templates):
        self.rows = []
        with Path(metadata_path).open() as f:
            metadata = [json.loads(line) for line in f]
        for metadata_index, row in enumerate(metadata):
            for template in templates:
                caption = render_caption(row, template=template)
                self.rows.append({
                    "dataset_type": "base",
                    "metadata_index": metadata_index,
                    "image_path": row["image_path"],
                    "split": row["split"],
                    "template": template,
                    "caption": caption,
                })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class FeedbackRows(Dataset):
    def __init__(self, dataset_root):
        self.root = Path(dataset_root)
        path = self.root / "feedback_dataset.jsonl"
        self.rows = []
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                self.rows.append({
                    "dataset_type": "feedback",
                    "metadata_index": row["metadata_index"],
                    "gt_image_path": row["gt_image_path"],
                    "generated_image_path": row["generated_image_path"],
                    "source_image_path": row.get("source_image_path", row["gt_image_path"]),
                    "split": "train",
                    "template": row.get("template", "feedback"),
                    "caption": row["caption"],
                    "feedback": row["feedback"],
                    "metadata": row.get("metadata"),
                    "sample_index": row.get("sample_index"),
                    "feedback_index": row.get("feedback_index"),
                    "tuple_id": row.get("tuple_id"),
                })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def collate_rows(rows):
    return rows


def masked_mean(hidden, mask):
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


@torch.no_grad()
def encode_texts(texts, tokenizer, encoder, max_length, device, dtype):
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    hidden = encoder(**encoded).last_hidden_state
    mask = encoded["attention_mask"]
    hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)
    pooled = masked_mean(hidden, mask)
    return hidden.cpu().to(dtype), mask.cpu().to(torch.bool), pooled.cpu().to(dtype)


@torch.no_grad()
def encode_attempt_latents(rows, dataset_root, vae, image_size, device, dtype):
    images = []
    for row in rows:
        path = resolve_path(dataset_root, row["generated_image_path"])
        images.append(image_to_tensor(path, image_size))
    images = torch.stack(images).to(device)
    latents = vae.encode(images).latent_dist.mode().mul_(vae.config.scaling_factor)
    return latents.cpu().to(dtype)


def main(args):
    dataset_root = Path(args.dataset)
    out_dir = dataset_root / "text_embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "base":
        rows = CaptionRows(dataset_root / "metadata.jsonl", args.templates)
    else:
        rows = FeedbackRows(dataset_root)
    loader = DataLoader(rows, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_rows)

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    encoder = T5EncoderModel.from_pretrained(args.encoder).to(device)
    encoder.eval()
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    vae = None
    if args.mode == "feedback":
        vae = AutoencoderKL.from_pretrained(args.vae).to(device)
        vae.eval()

    index = {
        "dataset_type": args.mode,
        "encoder": args.encoder,
        "max_length": args.max_length,
        "templates": args.templates if args.mode == "base" else [],
        "embedding_dim": encoder.config.d_model,
        "vae": args.vae if args.mode == "feedback" else None,
        "vae_scaling_factor": float(vae.config.scaling_factor) if vae is not None else None,
        "image_size": args.image_size,
        "records": [],
    }
    shard_rows = []
    shard_caption_tokens = []
    shard_caption_masks = []
    shard_caption_pooled = []
    shard_feedback_tokens = []
    shard_feedback_masks = []
    shard_feedback_pooled = []
    shard_attempt_latents = []
    shard_id = 0

    def flush_shard():
        nonlocal shard_id, shard_rows, shard_caption_tokens, shard_caption_masks, shard_caption_pooled
        nonlocal shard_feedback_tokens, shard_feedback_masks, shard_feedback_pooled, shard_attempt_latents
        if not shard_rows:
            return
        shard_name = f"shard_{shard_id:05d}.pt"
        shard_path = out_dir / shard_name
        payload = {
            "rows": shard_rows,
            "text_tokens": torch.cat(shard_caption_tokens, dim=0),
            "text_mask": torch.cat(shard_caption_masks, dim=0),
            "text_pooled": torch.cat(shard_caption_pooled, dim=0),
        }
        if args.mode == "feedback":
            payload.update({
                "feedback_tokens": torch.cat(shard_feedback_tokens, dim=0),
                "feedback_mask": torch.cat(shard_feedback_masks, dim=0),
                "feedback_pooled": torch.cat(shard_feedback_pooled, dim=0),
                "attempt_latent": torch.cat(shard_attempt_latents, dim=0),
            })
        torch.save(payload, shard_path)
        for offset, row in enumerate(shard_rows):
            index["records"].append({**row, "shard": shard_name, "offset": offset})
        print(f"Wrote {shard_path}")
        shard_id += 1
        shard_rows = []
        shard_caption_tokens = []
        shard_caption_masks = []
        shard_caption_pooled = []
        shard_feedback_tokens = []
        shard_feedback_masks = []
        shard_feedback_pooled = []
        shard_attempt_latents = []

    with torch.no_grad():
        for batch in loader:
            captions = [row["caption"] for row in batch]
            tokens, masks, pooled = encode_texts(captions, tokenizer, encoder, args.max_length, device, dtype)
            shard_rows.extend(batch)
            shard_caption_tokens.append(tokens)
            shard_caption_masks.append(masks)
            shard_caption_pooled.append(pooled)

            if args.mode == "feedback":
                feedbacks = [row["feedback"] for row in batch]
                ftokens, fmasks, fpooled = encode_texts(feedbacks, tokenizer, encoder, args.max_length, device, dtype)
                attempt_latents = encode_attempt_latents(batch, dataset_root, vae, args.image_size, device, dtype)
                shard_feedback_tokens.append(ftokens)
                shard_feedback_masks.append(fmasks)
                shard_feedback_pooled.append(fpooled)
                shard_attempt_latents.append(attempt_latents)

            if len(shard_rows) >= args.shard_size:
                flush_shard()

    flush_shard()
    with (out_dir / "index.json").open("w") as f:
        json.dump(index, f, indent=2)
    print(f"Wrote {len(index['records'])} embedding records to {out_dir / 'index.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--mode", choices=["base", "feedback"], default="base")
    parser.add_argument("--encoder", type=str, default="google/flan-t5-large")
    parser.add_argument("--templates", nargs="+", default=["chain", "order", "compact"])
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vae", type=str, default="stabilityai/sdxl-vae")
    parser.add_argument("--image-size", type=int, default=256)
    main(parser.parse_args())
