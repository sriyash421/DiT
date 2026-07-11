"""Evaluate the adaptive feedback rollout: sample, get verifier feedback, re-encode, and resample.

The model and dataset are rebuilt strictly from the training run's saved config.yaml (found one
level above the checkpoint's `checkpoints/` dir). The verifier is picked from the shared
`configs/verifier/*.yaml` files. Everything else is plain argparse."""
import argparse
from pathlib import Path

import hydra
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from omegaconf import OmegaConf

import wandb
from algorithms.eval import adaptive_rollout, distance_metrics, select_eval_batch
from algorithms.on_policy import PolicySampler
from algorithms.utils import save_trace_grid, write_json
from diffusion import create_diffusion
from verifiers.eval_metrics import make_scorer


def resolve_run(run_dir, step):
    """Resolve a run dir + step into (training config, full checkpoint path); step=-1 = latest."""
    run_dir = Path(run_dir)
    cfg_path = run_dir / "config.yaml"
    assert cfg_path.exists(), f"No training config at {cfg_path}; is {run_dir} a training run dir?"
    ckpt_dir = run_dir / "checkpoints"
    if step < 0:
        steps = sorted(int(p.stem) for p in ckpt_dir.glob("*.pt") if not p.stem.endswith("-ema"))
        assert steps, f"No checkpoints found in {ckpt_dir}."
        step = steps[-1]
    ckpt = ckpt_dir / f"{step:07d}.pt"
    assert ckpt.exists(), f"Checkpoint {ckpt} does not exist."
    print(f"Run {run_dir.name}: config={cfg_path.name}, ckpt={ckpt.name}")
    return OmegaConf.load(cfg_path), str(ckpt)


def build_verifier(name, api_url=None):
    verifier_cfg_path = Path(__file__).parent / "configs" / "verifier" / f"{name}.yaml"
    assert verifier_cfg_path.exists(), f"No verifier config at {verifier_cfg_path}."
    verifier_cfg = OmegaConf.load(verifier_cfg_path)
    if api_url is not None:
        verifier_cfg.api_url = api_url
    return hydra.utils.instantiate(verifier_cfg)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True, help="Training run dir (holds config.yaml and checkpoints/).")
    p.add_argument("--step", type=int, default=-1, help="Checkpoint step to load; -1 for the latest.")
    p.add_argument("--out_dir", default="results/eval_adaptive")
    p.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--verifier", default="open_router", help="Name under configs/verifier/ (no .yaml).")
    p.add_argument("--verifier_api_url", default=None, help="Override the verifier api_url (e.g. vllm node).")
    p.add_argument("--split", default="val")
    p.add_argument("--num_captions", type=int, default=8)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--caption_seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_sampling_steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--ddim_eta", type=float, default=0.0)
    p.add_argument("--wandb_project", default="DiT-qwen-clevr")
    p.add_argument("--wandb_name", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = 0
    torch.cuda.set_device(device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cfg, ckpt = resolve_run(args.run_dir, args.step)
    args.ckpt = ckpt
    dataset = hydra.utils.instantiate(train_cfg.dataset, split=args.split)
    model = hydra.utils.instantiate(train_cfg.model, context_dim=dataset.context_dim, device=device)
    model.load(ckpt, use_ema=args.use_ema)
    model.net.eval()
    verifier = build_verifier(args.verifier, api_url=args.verifier_api_url)
    context_encoder = model.get_encoder()
    context_encoder.eval()
    scorer = make_scorer()
    sampler = PolicySampler(
        create_diffusion(str(args.num_sampling_steps)),
        latent_size=model.latent_size,
        vae_scaling_factor=model.vae.config.scaling_factor,
        cfg_scale=args.cfg_scale,
        ddim_eta=args.ddim_eta,
    )

    batch = select_eval_batch(dataset, args.caption_seed, args.num_captions)
    with torch.no_grad():
        batch["context_tokens"], batch["context_mask"] = context_encoder.encode_history(
            batch["caption"], [[] for _ in batch["caption"]], [[] for _ in batch["caption"]]
        )

    traces, histories, token_count = adaptive_rollout(
        model.net,
        model.vae,
        sampler,
        verifier,
        context_encoder,
        batch,
        batch["gt_images"],
        steps=args.steps,
        seed=args.seed,
        scorer=scorer,
    )

    for idx, trace in enumerate(traces):
        if not trace:
            continue
        save_trace_grid(
            out_dir / f"trace_{idx:03d}.png",
            batch["gt_images"][idx],
            trace[0]["image"],
            "\n".join(histories[idx]),
            trace[-1]["image"],
        )
        for step, entry in enumerate(trace):
            entry["image"].save(out_dir / f"caption_{idx:03d}_step_{step:02d}.png")

    metrics = distance_metrics(traces)
    results = {
        "ckpt": args.ckpt,
        "split": args.split,
        "steps": args.steps,
        "num_captions": args.num_captions,
        "verifier_tokens": token_count,
        "metrics": metrics,
        "captions": batch["caption"],
        "feedback_histories": histories,
        "distances": [[entry["distance"] for entry in trace] for trace in traces],
    }
    write_json(out_dir / "results.json", results)
    print(f"Adaptive eval over {args.num_captions} captions x {args.steps} steps: {metrics}")

    if args.wandb_name is not None:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
        payload = {f"eval/{key}": value for key, value in metrics.items()}
        for idx in range(len(traces)):
            grid_path = out_dir / f"trace_{idx:03d}.png"
            if grid_path.exists():
                payload[f"eval/trace_{idx:03d}"] = wandb.Image(str(grid_path))
        wandb.log(payload)
        wandb.finish()


if __name__ == "__main__":
    main()
