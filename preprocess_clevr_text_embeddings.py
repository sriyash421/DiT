"""
Precompute frozen FLAN-T5 caption embeddings for the CLEVR DiT dataset.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, T5EncoderModel

from clevr_captions import render_caption


class CaptionRows(Dataset):
    def __init__(self, metadata_path, templates):
        self.rows = []
        with Path(metadata_path).open() as f:
            metadata = [json.loads(line) for line in f]
        for metadata_index, row in enumerate(metadata):
            for template in templates:
                caption = render_caption(row, template=template)
                self.rows.append({
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


def collate_rows(rows):
    return rows


def masked_mean(hidden, mask):
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def main(args):
    dataset_root = Path(args.dataset)
    metadata_path = dataset_root / "metadata.jsonl"
    out_dir = dataset_root / "text_embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = CaptionRows(metadata_path, args.templates)
    loader = DataLoader(rows, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_rows)

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    encoder = T5EncoderModel.from_pretrained(args.encoder).to(device)
    encoder.eval()
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    index = {
        "encoder": args.encoder,
        "max_length": args.max_length,
        "templates": args.templates,
        "embedding_dim": encoder.config.d_model,
        "records": [],
    }
    shard_rows = []
    shard_tokens = []
    shard_masks = []
    shard_pooled = []
    shard_id = 0

    def flush_shard():
        nonlocal shard_id, shard_rows, shard_tokens, shard_masks, shard_pooled
        if not shard_rows:
            return
        shard_name = f"shard_{shard_id:05d}.pt"
        shard_path = out_dir / shard_name
        torch.save({
            "rows": shard_rows,
            "text_tokens": torch.cat(shard_tokens, dim=0),
            "text_mask": torch.cat(shard_masks, dim=0),
            "text_pooled": torch.cat(shard_pooled, dim=0),
        }, shard_path)
        for offset, row in enumerate(shard_rows):
            index["records"].append({
                **row,
                "shard": shard_name,
                "offset": offset,
            })
        print(f"Wrote {shard_path}")
        shard_id += 1
        shard_rows = []
        shard_tokens = []
        shard_masks = []
        shard_pooled = []

    with torch.no_grad():
        for batch in loader:
            captions = [row["caption"] for row in batch]
            encoded = tokenizer(
                captions,
                padding="max_length",
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = encoder(**encoded).last_hidden_state
            mask = encoded["attention_mask"]
            hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)
            pooled = masked_mean(hidden, mask)

            shard_rows.extend(batch)
            shard_tokens.append(hidden.cpu().to(dtype))
            shard_masks.append(mask.cpu().to(torch.bool))
            shard_pooled.append(pooled.cpu().to(dtype))

            if len(shard_rows) >= args.shard_size:
                flush_shard()

    flush_shard()
    with (out_dir / "index.json").open("w") as f:
        json.dump(index, f, indent=2)
    print(f"Wrote {len(index['records'])} embedding records to {out_dir / 'index.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--encoder", type=str, default="google/flan-t5-large")
    parser.add_argument("--templates", nargs="+", default=["chain", "order", "compact"])
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    main(parser.parse_args())
