"""
Controlled CLEVR overfit experiments for debugging DiT training/sampling.

All artifacts are written under results/debug by default:
  data/       fixed 50-image subset metadata and reference images
  cache/      precomputed SDXL VAE latents and cached text embeddings
  runs/       per-experiment checkpoints, samples, CSV logs, plots
  reports/    human-readable summaries and next-step notes
"""
import argparse
import csv
import json
import math
import os
import random
import shutil
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from time import time

import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from diffusers.models import AutoencoderKL
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from diffusion import create_diffusion
from models import DiT_models


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


@torch.no_grad()
def update_ema(ema_model, model, decay):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag):
    for param in model.parameters():
        param.requires_grad = flag


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_name(value):
    return str(value).replace("/", "-").replace(".", "p")


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path, data):
    with Path(path).open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def write_markdown(path, text):
    Path(path).write_text(text)


def load_metadata_rows(dataset_root):
    metadata_path = Path(dataset_root) / "metadata.jsonl"
    with metadata_path.open() as f:
        return [json.loads(line) for line in f]


def select_subset(args):
    debug_root = Path(args.debug_root)
    data_dir = debug_root / "data" / f"subset_{args.subset_size:03d}_seed_{args.seed}"
    ensure_dir(data_dir)

    rows = load_metadata_rows(args.dataset_root)
    candidates = []
    dataset_root = Path(args.dataset_root)
    for row in rows:
        if row["split"] != args.split:
            continue
        image_path = dataset_root / row["image_path"]
        if image_path.is_file():
            candidates.append(row)
    if len(candidates) < args.subset_size:
        raise RuntimeError(f"Only found {len(candidates)} readable {args.split} rows")

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    subset = candidates[:args.subset_size]
    with (data_dir / "metadata.jsonl").open("w") as f:
        for row in subset:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    image_link_dir = data_dir / "images"
    ensure_dir(image_link_dir)
    for idx, row in enumerate(subset):
        src = dataset_root / row["image_path"]
        dst = image_link_dir / f"{idx:03d}_{src.name}"
        if not dst.exists():
            os.symlink(os.path.relpath(src, dst.parent), dst)

    write_json(data_dir / "subset_config.json", vars(args))
    return subset, data_dir


def load_text_embedding_map(dataset_root, template):
    index_path = Path(dataset_root) / "text_embeddings" / "index.json"
    with index_path.open() as f:
        index = json.load(f)
    records = {}
    for record in index["records"]:
        if record["template"] == template:
            records[record["image_path"]] = record
    return index, records


def tensor_to_grid(image_tensors, nrow, out_path):
    save_image(image_tensors, out_path, nrow=nrow, normalize=True, value_range=(-1, 1))


@torch.no_grad()
def prepare_cache(args):
    seed_all(args.seed)
    subset, data_dir = select_subset(args)
    debug_root = Path(args.debug_root)
    cache_dir = debug_root / "cache"
    ref_dir = debug_root / "references"
    ensure_dir(cache_dir)
    ensure_dir(ref_dir)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    images = []
    captions = []
    for row in subset:
        image = Image.open(Path(args.dataset_root) / row["image_path"]).convert("RGB")
        images.append(transform(image))
        captions.append(row["image_path"])
    image_tensor = torch.stack(images)
    tensor_to_grid(image_tensor, nrow=10, out_path=ref_dir / "subset_images.png")

    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    vae.eval()
    scaling_factor = float(vae.config.scaling_factor)
    latents = []
    recon = []
    batch_size = args.vae_batch_size
    for start in range(0, len(image_tensor), batch_size):
        batch = image_tensor[start:start + batch_size].to(device)
        encoded = vae.encode(batch).latent_dist
        latent = encoded.mean if args.latent_mode == "mean" else encoded.sample()
        latent = latent * scaling_factor
        decoded = vae.decode(latent / scaling_factor).sample
        latents.append(latent.cpu())
        recon.append(decoded.cpu())
    latents = torch.cat(latents)
    recon = torch.cat(recon)
    tensor_to_grid(recon, nrow=10, out_path=ref_dir / "vae_reconstructions.png")

    text_index, text_records = load_text_embedding_map(args.dataset_root, args.template)
    shard_cache = {}
    text_tokens = []
    text_masks = []
    text_pooled = []
    rendered_captions = []
    for row in subset:
        record = text_records[row["image_path"]]
        if record["shard"] not in shard_cache:
            shard_cache[record["shard"]] = torch.load(
                Path(args.dataset_root) / "text_embeddings" / record["shard"],
                map_location="cpu",
            )
        shard = shard_cache[record["shard"]]
        offset = record["offset"]
        text_tokens.append(shard["text_tokens"][offset].float())
        text_masks.append(shard["text_mask"][offset].bool())
        text_pooled.append(shard["text_pooled"][offset].float())
        rendered_captions.append(record["caption"])

    cache = {
        "latents": latents,
        "images": image_tensor,
        "text_tokens": torch.stack(text_tokens),
        "text_mask": torch.stack(text_masks),
        "text_pooled": torch.stack(text_pooled),
        "captions": rendered_captions,
        "rows": subset,
        "vae": args.vae,
        "vae_scaling_factor": scaling_factor,
        "latent_mode": args.latent_mode,
        "image_size": args.image_size,
        "text_embed_dim": text_index["embedding_dim"],
        "max_text_len": text_index["max_length"],
        "template": args.template,
    }
    cache_path = cache_dir / f"clevr_subset_{args.subset_size:03d}_{safe_name(Path(args.vae).name)}_{args.latent_mode}.pt"
    torch.save(cache, cache_path)

    with (ref_dir / "captions.txt").open("w") as f:
        for idx, caption in enumerate(rendered_captions):
            f.write(f"{idx:03d}: {caption}\n")

    report = (
        "# Debug Subset\n\n"
        f"- Dataset root: `{args.dataset_root}`\n"
        f"- Split: `{args.split}`\n"
        f"- Subset size: `{args.subset_size}`\n"
        f"- Seed: `{args.seed}`\n"
        f"- VAE: `{args.vae}`\n"
        f"- VAE scaling factor: `{scaling_factor}`\n"
        f"- Latent mode: `{args.latent_mode}`\n"
        f"- Cache: `{cache_path}`\n"
        f"- References: `references/subset_images.png`, `references/vae_reconstructions.png`\n"
    )
    ensure_dir(debug_root / "reports")
    write_markdown(debug_root / "reports" / "subset.md", report)
    print(f"Prepared cache at {cache_path}")
    return cache_path


def load_cache(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def make_model(args, cache, text_conditioning):
    latent_size = args.image_size // 8
    kwargs = {
        "input_size": latent_size,
        "num_classes": args.num_classes,
        "class_dropout_prob": args.class_dropout_prob,
    }
    if text_conditioning:
        kwargs.update({
            "text_conditioning": True,
            "text_embed_dim": cache["text_embed_dim"],
            "max_text_len": cache["max_text_len"],
        })
    return DiT_models[args.model](**kwargs)


def grad_norm(parameters):
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        value = param.grad.detach().data.norm(2).item()
        total += value * value
    return math.sqrt(total)


def append_csv(path, row, header):
    exists = Path(path).exists()
    with Path(path).open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def plot_loss(csv_path, out_path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Using PIL loss plot because matplotlib is unavailable: {exc}")
        plot_loss_pil(csv_path, out_path, title)
        return
    steps, losses, grad_norms = [], [], []
    with Path(csv_path).open() as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
            grad_norms.append(float(row["grad_norm"]))
    if not steps:
        return
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(steps, losses, color="#1f77b4", label="loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("train loss")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(steps, grad_norms, color="#d62728", alpha=0.55, label="grad norm")
    ax2.set_ylabel("grad norm")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_loss_pil(csv_path, out_path, title):
    rows = []
    with Path(csv_path).open() as f:
        for row in csv.DictReader(f):
            rows.append((int(row["step"]), float(row["loss"]), float(row["grad_norm"])))
    if not rows:
        return

    width, height = 1000, 560
    margin_l, margin_r, margin_t, margin_b = 78, 42, 54, 70
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    image = Image.new("RGB", (width, height), "white")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    draw.text((margin_l, 18), title[:110], fill=(20, 20, 20))

    steps = [row[0] for row in rows]
    losses = [row[1] for row in rows]
    grads = [row[2] for row in rows]
    x_min, x_max = min(steps), max(steps)
    loss_min, loss_max = min(losses), max(losses)
    grad_min, grad_max = min(grads), max(grads)
    loss_pad = max((loss_max - loss_min) * 0.08, 1e-4)
    grad_pad = max((grad_max - grad_min) * 0.08, 1e-4)
    loss_min, loss_max = loss_min - loss_pad, loss_max + loss_pad
    grad_min, grad_max = grad_min - grad_pad, grad_max + grad_pad

    def x_of(step):
        if x_max == x_min:
            return margin_l
        return margin_l + int((step - x_min) / (x_max - x_min) * plot_w)

    def y_of(value, v_min, v_max):
        if v_max == v_min:
            return margin_t + plot_h // 2
        return margin_t + int((v_max - value) / (v_max - v_min) * plot_h)

    # Grid and axes.
    draw.rectangle((margin_l, margin_t, margin_l + plot_w, margin_t + plot_h), outline=(40, 40, 40))
    for i in range(1, 5):
        y = margin_t + i * plot_h // 5
        draw.line((margin_l, y, margin_l + plot_w, y), fill=(225, 225, 225))
    for i in range(1, 5):
        x = margin_l + i * plot_w // 5
        draw.line((x, margin_t, x, margin_t + plot_h), fill=(235, 235, 235))

    loss_points = [(x_of(s), y_of(v, loss_min, loss_max)) for s, v, _ in rows]
    grad_points = [(x_of(s), y_of(v, grad_min, grad_max)) for s, _, v in rows]
    if len(loss_points) > 1:
        draw.line(loss_points, fill=(35, 102, 178), width=3)
        draw.line(grad_points, fill=(190, 65, 55), width=2)
    draw.text((margin_l, height - 42), f"step {x_min} to {x_max}", fill=(20, 20, 20))
    draw.text((margin_l, height - 24), f"loss range {loss_min:.4f} to {loss_max:.4f}", fill=(35, 102, 178))
    draw.text((width - 300, height - 24), f"grad range {grad_min:.3f} to {grad_max:.3f}", fill=(190, 65, 55))
    image.save(out_path)


@torch.no_grad()
def decode_latents(latents, vae, scaling_factor, batch_size):
    images = []
    for start in range(0, latents.shape[0], batch_size):
        batch = latents[start:start + batch_size]
        images.append(vae.decode(batch / scaling_factor).sample.cpu())
    return torch.cat(images)


@torch.no_grad()
def sample_model(model, ema, args, cache, run_dir, step, mode, device, vae):
    sample_dir = run_dir / "samples"
    ensure_dir(sample_dir)
    diffusion = create_diffusion(str(args.sample_steps))
    latent_size = args.image_size // 8
    scaling_factor = cache["vae_scaling_factor"]
    z = torch.randn(args.num_samples, 4, latent_size, latent_size, device=device)

    models = [("raw", model)]
    if ema is not None:
        models.append(("ema", ema))
    for name, active_model in models:
        active_model.eval()
        if mode == "uncond":
            model_kwargs = {}
            forward_fn = active_model.forward
        else:
            caption_indices = torch.arange(args.num_samples, device=device) % cache["text_tokens"].shape[0]
            tokens = cache["text_tokens"][caption_indices.cpu()].to(device)
            mask = cache["text_mask"][caption_indices.cpu()].to(device)
            pooled = cache["text_pooled"][caption_indices.cpu()].to(device)
            model_kwargs = {"text_tokens": tokens, "text_mask": mask, "text_pooled": pooled}
            forward_fn = active_model.forward
        samples = diffusion.p_sample_loop(
            forward_fn,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=device,
        )
        decoded = decode_latents(samples, vae, scaling_factor, args.vae_batch_size)
        out_path = sample_dir / f"step_{step:07d}_{name}.png"
        save_image(decoded, out_path, nrow=args.nrow, normalize=True, value_range=(-1, 1))
        active_model.train()
    model.train()


def train_experiment(args):
    seed_all(args.seed)
    if args.cache is None:
        cache_path = prepare_cache(args)
    else:
        cache_path = Path(args.cache)
    cache = load_cache(cache_path)
    debug_root = Path(args.debug_root)
    mode = args.mode
    run_name = args.run_name or (
        f"{mode}_{safe_name(args.model)}_lr{safe_name(args.lr)}_ema{safe_name(args.ema_decay)}"
        f"_steps{args.steps}_bs{args.batch_size}"
    )
    run_dir = debug_root / "runs" / run_name
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "plots")
    ensure_dir(run_dir / "samples")
    write_json(run_dir / "config.json", {**vars(args), "cache": str(cache_path)})

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    latents = cache["latents"].to(device)
    model = make_model(args, cache, text_conditioning=(mode == "text")).to(device)
    ema = deepcopy(model).to(device) if args.use_ema else None
    if ema is not None:
        requires_grad(ema, False)
        update_ema(ema, model, decay=0.0)
        ema.eval()
    model.train()
    diffusion = create_diffusion(timestep_respacing="")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    vae = AutoencoderKL.from_pretrained(cache["vae"]).to(device)
    vae.eval()
    requires_grad(vae, False)

    log_path = run_dir / "train_log.csv"
    header = ["step", "loss", "grad_norm", "lr", "steps_per_sec"]
    start_time = time()
    running_loss = 0.0
    running_grad = 0.0
    running_steps = 0
    print(f"Starting {mode} run {run_name} on {device}")
    print(f"Writing artifacts to {run_dir}")

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, latents.shape[0], (args.batch_size,), device=device)
        x = latents[idx]
        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
        if mode == "uncond":
            model_kwargs = {}
        else:
            cpu_idx = idx.cpu()
            model_kwargs = {
                "text_tokens": cache["text_tokens"][cpu_idx].to(device),
                "text_mask": cache["text_mask"][cpu_idx].to(device),
                "text_pooled": cache["text_pooled"][cpu_idx].to(device),
            }
        loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
        loss = loss_dict["loss"].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        before_clip = grad_norm(model.parameters())
        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if ema is not None:
            update_ema(ema, model, decay=args.ema_decay)

        running_loss += float(loss.item())
        running_grad += before_clip
        running_steps += 1

        if step % args.log_every == 0 or step == 1:
            torch.cuda.synchronize() if device == "cuda" else None
            elapsed = max(time() - start_time, 1e-6)
            row = {
                "step": step,
                "loss": running_loss / running_steps,
                "grad_norm": running_grad / running_steps,
                "lr": args.lr,
                "steps_per_sec": running_steps / elapsed,
            }
            append_csv(log_path, row, header)
            print(
                f"step={step:07d} loss={row['loss']:.5f} "
                f"grad_norm={row['grad_norm']:.3f} steps_per_sec={row['steps_per_sec']:.2f}"
            )
            running_loss = 0.0
            running_grad = 0.0
            running_steps = 0
            start_time = time()
            plot_loss(log_path, run_dir / "plots" / "loss.png", run_name)

        if step % args.sample_every == 0 or step in args.sample_at:
            sample_model(model, ema, args, cache, run_dir, step, mode, device, vae)

        if step % args.ckpt_every == 0 or step == args.steps:
            checkpoint = {
                "model": model.state_dict(),
                "ema": ema.state_dict() if ema is not None else None,
                "opt": opt.state_dict(),
                "args": vars(args),
                "cache": str(cache_path),
                "step": step,
            }
            torch.save(checkpoint, run_dir / "checkpoints" / f"{step:07d}.pt")

    sample_model(model, ema, args, cache, run_dir, args.steps, mode, device, vae)
    plot_loss(log_path, run_dir / "plots" / "loss.png", run_name)
    notes = (
        f"# {run_name}\n\n"
        f"- Mode: `{mode}`\n"
        f"- Model: `{args.model}` from scratch\n"
        f"- Steps: `{args.steps}`\n"
        f"- Batch size: `{args.batch_size}`\n"
        f"- LR: `{args.lr}`\n"
        f"- EMA decay: `{args.ema_decay}`\n"
        f"- Grad clip: `{args.grad_clip}`\n"
        f"- Cache: `{cache_path}`\n"
        f"- Loss CSV: `train_log.csv`\n"
        f"- Loss plot: `plots/loss.png`\n"
        f"- Samples: `samples/`\n"
    )
    write_markdown(run_dir / "README.md", notes)
    print(f"Finished {run_name}")


def main(args):
    ensure_dir(args.debug_root)
    ensure_dir(Path(args.debug_root) / "reports")
    if args.action == "prepare":
        prepare_cache(args)
    elif args.action == "train":
        train_experiment(args)
    else:
        raise ValueError(args.action)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["prepare", "train"], default="train")
    parser.add_argument("--mode", choices=["uncond", "text"], default="uncond")
    parser.add_argument("--debug-root", type=str, default="results/debug")
    parser.add_argument("--dataset-root", type=str, default="/gpfs/scrubbed/sriyash/clevr_dit_dataset")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--subset-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", type=str, default=None)
    parser.add_argument("--template", type=str, default="chain")
    parser.add_argument("--vae", type=str, default="stabilityai/sdxl-vae")
    parser.add_argument("--latent-mode", choices=["mean", "sample"], default="mean")
    parser.add_argument("--vae-batch-size", type=int, default=16)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/4")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--class-dropout-prob", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema-decay", type=float, default=0.9)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--sample-at", type=int, nargs="*", default=[100, 250])
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--nrow", type=int, default=4)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
