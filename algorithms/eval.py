"""Evaluation helpers: validation loss, batch selection, and result grids."""
import random
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import wandb
from algorithms.utils import dataloader_kwargs, rank_is_zero
from datasets.clevr.dataset import ClevrContextDataset, pad_contexts


def build_eval_datasets(dataset, split):
    """One eval dataset per source of a ClevrContextMultiDataset; empty splits are skipped."""
    eval_datasets = []
    eval_names = []
    for source, name in zip(dataset.datasets, dataset.names):
        eval_dataset = ClevrContextDataset(
            source.root,
            transform=source.transform,
            split=split,
            use_disk=source.use_disk,
            load_meta=source.load_meta,
            load_images=source.load_images,
            load_context=source.load_context,
        )
        if len(eval_dataset) == 0:
            continue
        eval_datasets.append(eval_dataset)
        eval_names.append(name)
    return eval_datasets, eval_names


@torch.no_grad()
def evaluate_dataset(model, eval_dataset, collate, eval_cfg, device):
    """Mean per-sample training loss over an eval split, all-reduced across ranks."""
    sampler = DistributedSampler(
        eval_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=False,
        drop_last=False,
    )
    loader = DataLoader(
        eval_dataset,
        **dataloader_kwargs(
            eval_cfg.dataloader,
            batch_size=eval_cfg.batch_size,
            shuffle=False,
            sampler=sampler,
            drop_last=False,
            collate_fn=collate,
        ),
    )
    total_loss = torch.tensor(0.0, device=device)
    total_count = torch.tensor(0.0, device=device)
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= eval_cfg.max_batches_per_dataset:
            break
        batch_size = int(batch["source_index"].shape[0])
        total_loss += model.loss(batch) * batch_size
        total_count += batch_size
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
    if total_count.item() == 0:
        return None
    return (total_loss / total_count).item()


def run_checkpoint_eval(model, eval_datasets, eval_names, eval_cfg, device, logger, train_steps):
    net = model.net
    was_training = net.training
    net.eval()
    encoder = getattr(model, "encoder", None)
    if encoder is not None:
        encoder.eval()
    metrics = {}
    for name, eval_dataset in zip(eval_names, eval_datasets):
        wrapped, collate = model.prepare_dataset(eval_dataset)
        loss = evaluate_dataset(model, wrapped, collate, eval_cfg, device)
        if loss is None:
            continue
        metrics[f"eval_loss/{name}"] = loss
    if rank_is_zero() and metrics:
        wandb.log(metrics, step=train_steps)
        logger.info("Eval losses: " + ", ".join(f"{key}={value:.4f}" for key, value in metrics.items()))
    if was_training:
        net.train()
        if encoder is not None:
            encoder.train()


def select_eval_batch(dataset, seed, count):
    """Randomly pick rows and return their contexts, captions, and GT images.

    Works on both ClevrContextDataset and ClevrContextMultiDataset.
    """
    if len(dataset) < count:
        raise RuntimeError(f"Only found {len(dataset)} rows; need {count}.")
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), count)
    contexts = []
    captions = []
    gt_images = []
    for idx in indices:
        item = dataset[idx]
        if dataset.context_dim is not None:
            contexts.append(item["context_tokens"].float())
        captions.append(item["caption"])
        gt_images.append(dataset.image_for_index(idx).convert("RGB"))
    if contexts:
        context_tokens, context_mask = pad_contexts(contexts)
    else:
        context_tokens, context_mask = None, None
    return {
        "indices": indices,
        "context_tokens": context_tokens,
        "context_mask": context_mask,
        "caption": captions,
        "gt_images": gt_images,
    }


@torch.no_grad()
def adaptive_eval(model, verifier, captions, gt_images, steps, seed, scorer, sampler_cfg):
    """Model-agnostic adaptive rollout for eval: iteratively `model.generate`, score against GT,
    and refine with verifier feedback, conditioning each attempt on the full prior history.

    Makes `steps` predictions and asks for feedback only steps-1 times. Returns
    (traces, histories, token_count): per-item lists of {image, feedback_used, distance} steps, the
    feedback given to each item, and the verifier tokens spent. Attempts are written to a temp dir
    so image-conditioned models (OmniGen) can reference prior attempts by path.
    """
    import tempfile

    count = len(captions)
    histories = [[] for _ in range(count)]
    attempt_image_history = [[] for _ in range(count)]
    attempt_path_history = [[] for _ in range(count)]
    traces = [[] for _ in range(count)]
    active = list(range(count))
    token_count = 0
    tmp_dir = Path(tempfile.mkdtemp(prefix="adaptive_eval_"))

    for step in range(steps):
        if not active:
            break
        context_batch = {
            "caption": [captions[idx] for idx in active],
            "feedback_history": [list(histories[idx]) for idx in active],
            "attempt_images": [list(attempt_image_history[idx]) for idx in active],
            "attempt_paths": [list(attempt_path_history[idx]) for idx in active],
        }
        attempt_images = model.generate(
            context_batch,
            num_sampling_steps=int(sampler_cfg.num_sampling_steps),
            cfg_scale=float(sampler_cfg.cfg_scale),
            ddim_eta=float(sampler_cfg.ddim_eta),
            seed=seed + step,
        )
        active_captions = [captions[idx] for idx in active]
        active_gt_images = [gt_images[idx] for idx in active]
        attempt_paths = []
        for pos, batch_idx in enumerate(active):
            path = tmp_dir / f"idx_{batch_idx:04d}_step_{step:02d}.png"
            attempt_images[pos].save(path)
            attempt_paths.append(str(path))
        score_results = scorer.score_distance(active_captions, active_gt_images, attempt_images)
        token_count += sum(int(result.token_count) for result in score_results)
        for pos, batch_idx in enumerate(active):
            distance = None
            if score_results and pos < len(score_results) and score_results[pos].ok:
                distance = score_results[pos].score
            traces[batch_idx].append({
                "image": attempt_images[pos],
                "feedback_used": histories[batch_idx][-1] if histories[batch_idx] else "",
                "distance": distance,
            })

        if step + 1 >= steps:
            break
        results = verifier.verify(
            active_captions,
            active_gt_images,
            attempt_images,
            [list(histories[idx]) for idx in active],
        )
        token_count += sum(int(result.token_count) for result in results)
        success_positions = [pos for pos, result in enumerate(results) if result.ok]
        if not success_positions:
            break
        next_active = []
        for pos in success_positions:
            batch_idx = active[pos]
            histories[batch_idx].append(results[pos].feedback)
            attempt_image_history[batch_idx].append(attempt_images[pos])
            attempt_path_history[batch_idx].append(attempt_paths[pos])
            next_active.append(batch_idx)
        active = next_active

    return traces, histories, token_count


def distance_metrics(traces):
    """Per-step mean distance plus best distance across steps, averaged over items."""
    metrics = {}
    by_step = {}
    for trace in traces:
        for step, entry in enumerate(trace):
            if entry["distance"] is not None:
                by_step.setdefault(step, []).append(float(entry["distance"]))
    for step, values in by_step.items():
        metrics[f"distance_step_{step}"] = sum(values) / len(values)
    best_scores = []
    for trace in traces:
        scores = [entry["distance"] for entry in trace if entry["distance"] is not None]
        if scores:
            best_scores.append(min(scores))
    if best_scores:
        metrics["best_distance"] = sum(best_scores) / len(best_scores)
        metrics["best_aligned_score"] = sum(1.0 / (1.0 + score) for score in best_scores) / len(best_scores)
    return metrics


def _wrap_text(draw, text, font, max_width):
    """Greedy word-wrap `text` to `max_width` pixels; returns a list of lines (>=1)."""
    words = str(text).split()
    if not words:
        return [""]
    lines, current = [], words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if draw.textbbox((0, 0), trial, font=font)[2] <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def render_adaptive_trace(trace, gt_image, caption, tile=160):
    """Render one adaptive-eval rollout as a strip image:

    a caption banner on top, then attempt_0 --feedback_0--> attempt_1 --feedback_1--> ... across the
    row, and finally the GT image (thick black border) at the far right. `trace` is the per-item list
    of {"image", "feedback_used", ...} dicts in step order (as returned by adaptive_eval).
    """
    font = ImageFont.load_default()
    measure = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    entries = list(trace) if trace else []
    attempts = [entry["image"].convert("RGB").resize((tile, tile)) for entry in entries]
    if not attempts:
        attempts = [Image.new("RGB", (tile, tile), (230, 230, 230))]
    # Feedback shown between attempt k and k+1 is the feedback that produced attempt k+1.
    feedbacks = [str(entries[k + 1].get("feedback_used", "")) for k in range(len(entries) - 1)]
    n = len(attempts)

    conn_w, gap, pad, label_h = 168, 26, 12, 16
    step = tile + conn_w
    line_h = font.getbbox("Ay")[3] + 3
    gt_x = pad + (n - 1) * step + tile + gap
    width = gt_x + tile + pad

    caption_lines = _wrap_text(measure, f"Caption: {caption}", font, width - 2 * pad)
    caption_h = pad + line_h * len(caption_lines) + pad
    height = caption_h + tile + label_h + pad

    out = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    for i, line in enumerate(caption_lines):
        draw.text((pad, pad + i * line_h), line, fill=(0, 0, 0), font=font)

    top = caption_h
    for k, img in enumerate(attempts):
        x = pad + k * step
        out.paste(img, (x, top))
        draw.rectangle((x, top, x + tile - 1, top + tile - 1), outline=(0, 0, 0), width=1)
        draw.text((x + 4, top + tile + 2), f"attempt {k}", fill=(0, 0, 0), font=font)
        if k < n - 1:
            cx = x + tile
            mid_y = top + tile // 2
            fb_lines = _wrap_text(measure, feedbacks[k], font, conn_w - 16)[:4]
            for i, line in enumerate(fb_lines):
                draw.text((cx + 8, mid_y - line_h * len(fb_lines) - 4 + i * line_h), line, fill=(0, 0, 0), font=font)
            draw.line((cx + 8, mid_y, cx + conn_w - 10, mid_y), fill=(0, 0, 0), width=2)
            draw.polygon(
                [(cx + conn_w - 4, mid_y), (cx + conn_w - 12, mid_y - 5), (cx + conn_w - 12, mid_y + 5)],
                fill=(0, 0, 0),
            )

    gt = gt_image.convert("RGB").resize((tile, tile))
    out.paste(gt, (gt_x, top))
    draw.rectangle((gt_x, top, gt_x + tile - 1, top + tile - 1), outline=(0, 0, 0), width=max(4, tile // 32))
    draw.text((gt_x + 4, top + tile + 2), "GT Image", fill=(0, 0, 0), font=font)
    return out


def save_adaptive_trace_grid(path, traces, gt_images, captions, tile=160):
    """Stack one render_adaptive_trace strip per sample into a single grid image. Returns the path,
    or None if no trace had any steps."""
    rows = [
        render_adaptive_trace(trace, gt_image, caption, tile=tile)
        for trace, gt_image, caption in zip(traces, gt_images, captions)
        if trace
    ]
    if not rows:
        return None
    gap = 18
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows) + gap * (len(rows) - 1)
    out = Image.new("RGB", (width, height), (255, 255, 255))
    y = 0
    for row in rows:
        out.paste(row, (0, y))
        y += row.height + gap
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
    return path


def save_gt_pred_grid(path, gt_images, pred_images, cols=4, gap=12, max_images=16):
    """Save a grid of GT | prediction pairs, plotting at most `max_images` pairs."""
    if max_images is not None:
        gt_images = gt_images[:max_images]
        pred_images = pred_images[:max_images]
    tile = 128
    pair_w = tile * 2
    label_h = 24
    cell_h = tile + label_h
    rows = (len(gt_images) + cols - 1) // cols
    out_w = cols * pair_w + max(cols - 1, 0) * gap
    out_h = rows * cell_h + max(rows - 1, 0) * gap
    out = Image.new("RGB", (out_w, out_h), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    for idx, (gt, pred) in enumerate(zip(gt_images, pred_images)):
        row = idx // cols
        col = idx % cols
        x = col * (pair_w + gap)
        y = row * (cell_h + gap)
        draw.text((x + 8, y + 4), "GT", fill=(20, 20, 20))
        draw.text((x + tile + 8, y + 4), "Pred", fill=(20, 20, 20))
        out.paste(gt.resize((tile, tile), Image.Resampling.LANCZOS), (x, y + label_h))
        out.paste(pred.resize((tile, tile), Image.Resampling.LANCZOS), (x + tile, y + label_h))
        # Thick black border around GT to mark it as the reference; thin border on the prediction.
        draw.rectangle((x, y + label_h, x + tile - 1, y + label_h + tile - 1), outline=(0, 0, 0), width=max(3, tile // 32))
        draw.rectangle((x + tile, y + label_h, x + pair_w - 1, y + label_h + tile - 1), outline=(0, 0, 0))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
