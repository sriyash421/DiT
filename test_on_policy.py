import json
import os
import time
from pathlib import Path

import pytest
import torch
from diffusers.models import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from clevr_transforms import build_clevr_transform
from datasets_clevr import ClevrContextDataset, context_collate
from diffusion import create_diffusion
from feedback_verifiers import DEFAULT_GEMINI_MODEL, GeminiVerifier, VerificationResult
from models import DiT_models
from on_policy import (
    PolicySampler,
    QwenContextEncoder,
    RolloutBuffer,
    RolloutCollector,
    diffusion_loss,
    load_metadata_for_zarr,
    metadata_by_index,
    rollout_collate,
    save_trace_grid,
    unwrap_model,
)
from train_text import load_checkpoint
from train_on_policy import all_reduce_rollout_stats, exact_update_batches, validate_startup_config
from train_utils import requires_grad, update_ema


def env(name, default=None):
    return os.getenv(name, default)


def latest_checkpoint(root):
    root = Path(root)
    if not root.exists():
        return None
    candidates = sorted(root.glob("**/*-ema.pt"))
    return str(candidates[-1]) if candidates else None


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def trainable_snapshot(model):
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def any_trainable_param_changed(model, before):
    return any(
        name in before and not torch.equal(before[name], param.detach())
        for name, param in model.named_parameters()
        if param.requires_grad
    )


def grad_l2_norm(model):
    terms = [
        param.grad.detach().float().pow(2).sum()
        for param in model.parameters()
        if param.requires_grad and param.grad is not None
    ]
    if not terms:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return torch.sqrt(torch.stack(terms).sum())


class FakeSampler:
    def __init__(self):
        self.models = []

    def sample(self, model, vae, context_tokens, context_mask, device, seed=None):
        del vae, context_mask, device, seed
        self.models.append(model)
        images = [
            Image.new("RGB", (8, 8), (idx * 40, 20, 10))
            for idx in range(int(context_tokens.shape[0]))
        ]
        return torch.zeros(len(images), 3, 8, 8), images


class FakeVerifier:
    def verify_batch(self, captions, metadata, gt_images, attempt_images):
        del metadata, gt_images, attempt_images
        return [
            VerificationResult(ok=True, feedback=f"fix {caption}", token_count=11)
            for caption in captions
        ]

    def verify_history_batch(self, captions, metadata, gt_images, attempt_images, feedback_histories):
        del metadata, gt_images, attempt_images
        return [
            VerificationResult(ok=True, feedback=f"fix {caption} step {len(history)}", token_count=11)
            for caption, history in zip(captions, feedback_histories)
        ]


class FakeContextEncoder:
    def encode_feedback(self, captions, metadata, feedback, attempt_images):
        del metadata, feedback, attempt_images
        tokens = torch.arange(len(captions) * 6, dtype=torch.float32).reshape(len(captions), 2, 3)
        return tokens, torch.ones(len(captions), 2, dtype=torch.bool)

    def encode_history(self, captions, feedback_histories, attempt_image_histories):
        del feedback_histories, attempt_image_histories
        tokens = torch.arange(len(captions) * 9, dtype=torch.float32).reshape(len(captions), 3, 3)
        return tokens, torch.ones(len(captions), 3, dtype=torch.bool)


class FakeVAE:
    class Config:
        scaling_factor = 1.0

    class LatentDist:
        def __init__(self, batch_size):
            self.batch_size = batch_size

        def sample(self):
            return torch.ones(self.batch_size, 4, 2, 2)

    class Encoded:
        def __init__(self, batch_size):
            self.latent_dist = FakeVAE.LatentDist(batch_size)

    config = Config()

    def encode(self, x):
        return self.Encoded(int(x.shape[0]))


class RecordingDistributedSampler(DistributedSampler):
    def __init__(self, dataset):
        super().__init__(dataset, num_replicas=1, rank=0, shuffle=False)
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)
        super().set_epoch(epoch)


def make_rollout_record(idx, attempt_path=None):
    return {
        "x_latent": torch.full((4, 2, 2), float(idx)),
        "base_context_tokens": torch.ones(2, 3) * idx,
        "adaptive_context_tokens": torch.ones(3, 3) * (idx + 10),
        "caption": f"caption {idx}",
        "feedback": f"feedback {idx}",
        "metadata_index": idx,
        "gt_path": f"gt-{idx}.png",
        "attempt_path": str(attempt_path or f"attempt-{idx}.png"),
    }


def save_rollout_payload(path, start, count, tokens=0):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    records = [make_rollout_record(start + idx) for idx in range(count)]
    torch.save(
        {
            "records": records,
            "stats": {
                "attempted": count,
                "success": count,
                "failed": 0,
                "gemini_tokens": tokens,
            },
        },
        path / "records.pt",
    )


def test_rollout_buffer_merges_rank_shards(tmp_path):
    step_dir = tmp_path / "step_000001"
    save_rollout_payload(step_dir / "rank_000", 0, 2, tokens=5)
    save_rollout_payload(step_dir / "rank_001", 2, 3, tokens=7)

    dataset = RolloutBuffer(step_dir)

    assert len(dataset) == 5
    assert [dataset[idx]["metadata_index"] for idx in range(len(dataset))] == [0, 1, 2, 3, 4]
    assert dataset.stats["attempted"] == 5
    assert dataset.stats["success"] == 5
    assert dataset.stats["gemini_tokens"] == 12


def test_rollout_collector_writes_rank_shard_records_and_attempt_pngs(tmp_path):
    batch = {
        "image": torch.zeros(2, 3, 8, 8),
        "context_tokens": torch.ones(2, 2, 3),
        "context_mask": torch.ones(2, 2, dtype=torch.bool),
        "caption": ["a", "b"],
        "metadata_index": torch.tensor([0, 1]),
    }
    data_sampler = RecordingDistributedSampler(TensorDataset(torch.arange(2)))
    collector = RolloutCollector(
        model="ema-model",
        vae=FakeVAE(),
        sampler=FakeSampler(),
        verifier=FakeVerifier(),
        context_encoder=FakeContextEncoder(),
        metadata_rows=[{"id": 0}, {"id": 1}],
    )

    stats = collector.collect(
        [batch],
        tmp_path / "step_000001" / "rank_000",
        sample_count=2,
        device=torch.device("cpu"),
        data_sampler=data_sampler,
        epoch=3,
    )
    dataset = RolloutBuffer(tmp_path / "step_000001")

    assert stats == {"attempted": 2, "success": 2, "failed": 0, "gemini_tokens": 22}
    assert len(dataset) == 2
    assert data_sampler.epochs == [3]
    assert (tmp_path / "step_000001" / "rank_000" / "records.pt").exists()
    assert (tmp_path / "step_000001" / "rank_000" / "gt" / "000000.png").exists()
    assert (tmp_path / "step_000001" / "rank_000" / "gt" / "000001.png").exists()
    assert (tmp_path / "step_000001" / "rank_000" / "attempts" / "000000.png").exists()
    assert (tmp_path / "step_000001" / "rank_000" / "attempts" / "000001.png").exists()
    assert dataset[0]["gt_path"].endswith("gt/000000.png")


def test_rollout_collector_multi_step_writes_step_records_and_histories(tmp_path):
    batch = {
        "image": torch.zeros(2, 3, 8, 8),
        "context_tokens": torch.ones(2, 2, 3),
        "context_mask": torch.ones(2, 2, dtype=torch.bool),
        "caption": ["a", "b"],
        "metadata_index": torch.tensor([0, 1]),
    }
    collector = RolloutCollector(
        model="ema-model",
        vae=FakeVAE(),
        sampler=FakeSampler(),
        verifier=FakeVerifier(),
        context_encoder=FakeContextEncoder(),
        metadata_rows=[{"id": 0}, {"id": 1}],
        rollout_length=3,
    )

    stats = collector.collect(
        [batch],
        tmp_path / "step_000001" / "rank_000",
        sample_count=2,
        device=torch.device("cpu"),
    )
    dataset = RolloutBuffer(tmp_path / "step_000001")

    assert stats == {"attempted": 6, "success": 6, "failed": 0, "gemini_tokens": 66}
    assert len(dataset) == 6
    assert [dataset[idx]["step_index"] for idx in range(len(dataset))] == [0, 0, 1, 1, 2, 2]
    assert dataset[0]["feedback_history"] == []
    assert dataset[2]["feedback_history"] == ["fix a step 0"]
    assert len(dataset[4]["history_attempt_paths"]) == 2
    assert Path(dataset[4]["attempt_path"]).exists()


def test_distributed_sampler_partitions_merged_rollout_buffer(tmp_path):
    step_dir = tmp_path / "step_000001"
    save_rollout_payload(step_dir / "rank_000", 0, 2)
    save_rollout_payload(step_dir / "rank_001", 2, 2)
    dataset = RolloutBuffer(step_dir)

    rank0 = list(DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=False))
    rank1 = list(DistributedSampler(dataset, num_replicas=2, rank=1, shuffle=False))

    assert set(rank0).isdisjoint(rank1)
    assert sorted(rank0 + rank1) == [0, 1, 2, 3]


def test_exact_update_batches_restarts_epochs_and_stops_exactly():
    dataset = TensorDataset(torch.arange(3))
    sampler = RecordingDistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=2, sampler=sampler)

    batches = list(exact_update_batches(loader, sampler, updates_per_step=5))

    assert len(batches) == 5
    assert sampler.epochs == [0, 1, 2]


def test_rollout_use_ema_model_selection_with_collector(tmp_path):
    batch = {
        "image": torch.zeros(1, 3, 8, 8),
        "context_tokens": torch.ones(1, 2, 3),
        "context_mask": torch.ones(1, 2, dtype=torch.bool),
        "caption": ["a"],
        "metadata_index": torch.tensor([0]),
    }
    for use_ema, expected_model in ((True, "ema"), (False, "raw")):
        fake_sampler = FakeSampler()
        collector = RolloutCollector(
            model="ema" if use_ema else "raw",
            vae=FakeVAE(),
            sampler=fake_sampler,
            verifier=FakeVerifier(),
            context_encoder=FakeContextEncoder(),
            metadata_rows=[{}],
        )
        collector.collect(
            [batch],
            tmp_path / f"use_ema_{use_ema}" / "rank_000",
            sample_count=1,
            device=torch.device("cpu"),
        )
        assert fake_sampler.models == [expected_model]


def test_all_reduce_rollout_stats_keeps_gemini_token_total_without_dist():
    stats = {"attempted": 2, "success": 1, "failed": 1, "gemini_tokens": 33}

    assert all_reduce_rollout_stats(stats, torch.device("cpu")) == stats


def test_rollout_collate_pads_records_from_disk_buffer(tmp_path):
    step_dir = tmp_path / "step_000001"
    save_rollout_payload(step_dir / "rank_000", 0, 2)
    dataset = RolloutBuffer(step_dir)
    batch = rollout_collate([dataset[0], dataset[1]])

    assert batch["x_latent"].shape == (2, 4, 2, 2)
    assert batch["base_context_tokens"].shape[0] == 2
    assert batch["adaptive_context_tokens"].shape[0] == 2


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def test_startup_config_allows_checkpoint_without_pretraining():
    validate_startup_config(AttrDict(train=AttrDict(ckpt="checkpoint.pt", pretraining_steps=0)))


def test_startup_config_allows_pretraining_without_checkpoint():
    validate_startup_config(AttrDict(train=AttrDict(ckpt=None, pretraining_steps=1)))


def test_startup_config_rejects_no_checkpoint_and_no_pretraining():
    with pytest.raises(AssertionError):
        validate_startup_config(AttrDict(train=AttrDict(ckpt=None, pretraining_steps=0)))


@pytest.fixture(scope="session")
def runtime_config():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real on-policy integration test.")
    api_key_env = env("ON_POLICY_TEST_API_KEY_ENV", "GEMINI_API_KEY")
    api_key = os.getenv(api_key_env)
    if not api_key:
        pytest.skip(f"{api_key_env} is required for Gemini verification.")

    dataset_path = env("ON_POLICY_TEST_DATASET", "/gpfs/scrubbed/sriyash/clevr_dit_dataset/data.zarr")
    if not Path(dataset_path).exists():
        pytest.skip(f"Dataset not found: {dataset_path}")

    ckpt = env("ON_POLICY_TEST_CKPT")
    if ckpt is None:
        ckpt = latest_checkpoint(env("ON_POLICY_TEST_CKPT_ROOT", "/gpfs/scrubbed/sriyash/DiT-qwen-clevr-base"))
    if ckpt is None or not Path(ckpt).exists():
        pytest.skip("Set ON_POLICY_TEST_CKPT to a valid DiT checkpoint.")

    out_root = Path(env("ON_POLICY_TEST_OUT", "results/on_policy_test"))
    out_dir = out_root / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "api_key": api_key,
        "dataset_path": dataset_path,
        "ckpt": ckpt,
        "out_dir": out_dir,
        "device": torch.device("cuda"),
        "model_name": env("ON_POLICY_TEST_MODEL", "DiT-S/4"),
        "image_size": int(env("ON_POLICY_TEST_IMAGE_SIZE", "256")),
        "batch_size": int(env("ON_POLICY_TEST_BATCH_SIZE", "1")),
        "sample_steps": int(env("ON_POLICY_TEST_SAMPLE_STEPS", "100")),
        "cfg_scale": float(env("ON_POLICY_TEST_CFG_SCALE", "1.0")),
        "split": env("ON_POLICY_TEST_SPLIT", "val"),
    }


@pytest.fixture(scope="session")
def base_batch(runtime_config):
    transform = build_clevr_transform(runtime_config["image_size"])
    dataset = ClevrContextDataset(
        runtime_config["dataset_path"],
        transform=transform,
        split=runtime_config["split"],
    )
    assert len(dataset) >= runtime_config["batch_size"]
    items = [dataset[idx] for idx in range(runtime_config["batch_size"])]
    batch = context_collate(items)
    metadata_rows = load_metadata_for_zarr(runtime_config["dataset_path"])
    metadata = metadata_by_index(metadata_rows, batch["metadata_index"].tolist())
    gt_images = [dataset.image_for_row(dataset.indices[idx]) for idx in range(runtime_config["batch_size"])]
    return {
        "dataset": dataset,
        "batch": batch,
        "metadata": metadata,
        "gt_images": gt_images,
    }


@pytest.fixture(scope="session")
def model_stack(runtime_config, base_batch):
    device = runtime_config["device"]
    dataset = base_batch["dataset"]
    model = DiT_models[runtime_config["model_name"]](
        input_size=runtime_config["image_size"] // 8,
        num_classes=1000,
        text_conditioning=True,
        context_dim=dataset.context_dim,
        class_dropout_prob=0.1,
    ).to(device)
    model.load_state_dict(load_checkpoint(runtime_config["ckpt"]), strict=False)
    model.train()
    ema = DiT_models[runtime_config["model_name"]](
        input_size=runtime_config["image_size"] // 8,
        num_classes=1000,
        text_conditioning=True,
        context_dim=dataset.context_dim,
        class_dropout_prob=0.1,
    ).to(device)
    ema.load_state_dict(model.state_dict())
    requires_grad(ema, False)

    vae = AutoencoderKL.from_pretrained(env("ON_POLICY_TEST_VAE", "stabilityai/sdxl-vae")).to(device)
    vae.eval()
    train_diffusion = create_diffusion(timestep_respacing="")
    sample_diffusion = create_diffusion(str(runtime_config["sample_steps"]))
    sampler = PolicySampler(
        sample_diffusion,
        latent_size=runtime_config["image_size"] // 8,
        vae_scaling_factor=vae.config.scaling_factor,
        cfg_scale=runtime_config["cfg_scale"],
        sampler="ddim",
    )
    return {
        "model": model,
        "ema": ema,
        "vae": vae,
        "train_diffusion": train_diffusion,
        "sampler": sampler,
    }


@pytest.fixture(scope="session")
def verifier(runtime_config):
    return GeminiVerifier(
        api_key=runtime_config["api_key"],
        model=env("ON_POLICY_TEST_GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        api_url=env("ON_POLICY_TEST_GEMINI_API_URL"),
        workers=int(env("ON_POLICY_TEST_VERIFIER_WORKERS", "4")),
        max_tokens=int(env("ON_POLICY_TEST_MAX_FEEDBACK_TOKENS", "96")),
        temperature=float(env("ON_POLICY_TEST_FEEDBACK_TEMPERATURE", "0.0")),
    )


@pytest.fixture(scope="session")
def context_encoder(runtime_config):
    return QwenContextEncoder(
        env("ON_POLICY_TEST_QWEN_MODEL", "Qwen/Qwen3.5-4B"),
        device="cuda",
        vlm_dtype=env("ON_POLICY_TEST_QWEN_DTYPE", "bfloat16"),
        device_map=env("ON_POLICY_TEST_QWEN_DEVICE_MAP", "auto"),
        max_length=int(env("ON_POLICY_TEST_MAX_CONTEXT_LEN", "1024")),
    )


@pytest.fixture(scope="session")
def attempt1(runtime_config, base_batch, model_stack):
    batch = base_batch["batch"]
    device = runtime_config["device"]
    _, images = model_stack["sampler"].sample(
        model_stack["model"],
        model_stack["vae"],
        batch["context_tokens"].to(device),
        batch["context_mask"].to(device),
        device,
        seed=0,
    )
    return images


@pytest.fixture(scope="session")
def feedback(runtime_config, base_batch, verifier, attempt1):
    batch = base_batch["batch"]
    results = verifier.verify_batch(
        batch["caption"],
        base_batch["metadata"],
        base_batch["gt_images"],
        attempt1,
    )
    success_indices = [idx for idx, result in enumerate(results) if result.ok]
    if not success_indices:
        errors = [result.error for result in results]
        save_json(runtime_config["out_dir"] / "verifier_failures.json", {
            "errors": errors,
            "captions": batch["caption"],
            "gemini_model": env("ON_POLICY_TEST_GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        })
        pytest.fail(f"Gemini verifier returned no successful feedback: {errors}")
    return {
        "results": results,
        "success_indices": success_indices,
        "texts": [results[idx].feedback for idx in success_indices],
    }


@pytest.fixture(scope="session")
def adaptive_context(runtime_config, base_batch, context_encoder, attempt1, feedback):
    batch = base_batch["batch"]
    success_indices = feedback["success_indices"]
    tokens, mask = context_encoder.encode_feedback(
        [batch["caption"][idx] for idx in success_indices],
        [base_batch["metadata"][idx] for idx in success_indices],
        feedback["texts"],
        [attempt1[idx] for idx in success_indices],
    )
    return {"tokens": tokens, "mask": mask}


@pytest.fixture(scope="session")
def one_step_update(runtime_config, base_batch, model_stack, adaptive_context, feedback):
    device = runtime_config["device"]
    model = model_stack["model"]
    batch = base_batch["batch"]
    success_indices = feedback["success_indices"]
    x_img = batch["image"].to(device)
    base_tokens = batch["context_tokens"].to(device)
    base_mask = batch["context_mask"].to(device)

    with torch.no_grad():
        x_latent = model_stack["vae"].encode(x_img).latent_dist.sample().mul_(model_stack["vae"].config.scaling_factor)

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-6)
    before = trainable_snapshot(model)
    base_loss = diffusion_loss(
        model,
        model_stack["train_diffusion"],
        x_latent,
        base_tokens,
        base_mask,
    )
    feedback_loss = diffusion_loss(
        model,
        model_stack["train_diffusion"],
        x_latent[success_indices],
        adaptive_context["tokens"].to(device),
        adaptive_context["mask"].to(device),
    )
    loss = feedback_loss + base_loss
    opt.zero_grad()
    loss.backward()
    grad_norm = grad_l2_norm(model)
    opt.step()
    update_ema(model_stack["ema"], unwrap_model(model), decay=0.999)
    return {
        "loss": loss.detach(),
        "feedback_loss": feedback_loss.detach(),
        "base_loss": base_loss.detach(),
        "grad_norm": grad_norm.detach(),
        "params_changed": any_trainable_param_changed(model, before),
    }


@pytest.mark.integration
def test_loads_base_batch(base_batch, runtime_config):
    batch = base_batch["batch"]
    assert batch["image"].shape[0] == runtime_config["batch_size"]
    assert batch["context_tokens"].ndim == 3
    assert batch["context_mask"].ndim == 2
    assert len(batch["caption"]) == runtime_config["batch_size"]


@pytest.mark.integration
def test_generates_attempt1(attempt1, runtime_config):
    assert len(attempt1) == runtime_config["batch_size"]
    assert all(image.size == (runtime_config["image_size"], runtime_config["image_size"]) for image in attempt1)


@pytest.mark.integration
def test_gets_gemini_feedback(feedback):
    assert feedback["success_indices"]
    assert all(text.strip() for text in feedback["texts"])


@pytest.mark.integration
def test_encodes_adaptive_context(adaptive_context, feedback):
    assert adaptive_context["tokens"].ndim == 3
    assert adaptive_context["mask"].ndim == 2
    assert adaptive_context["tokens"].shape[0] == len(feedback["success_indices"])


@pytest.mark.integration
def test_one_step_update_changes_parameters(one_step_update):
    assert torch.isfinite(one_step_update["loss"])
    assert torch.isfinite(one_step_update["feedback_loss"])
    assert torch.isfinite(one_step_update["base_loss"])
    assert torch.isfinite(one_step_update["grad_norm"])
    assert one_step_update["grad_norm"].item() > 0
    assert one_step_update["params_changed"]


@pytest.mark.integration
def test_generates_attempt2_and_logs_outputs(runtime_config, base_batch, model_stack, adaptive_context, attempt1, feedback, one_step_update):
    device = runtime_config["device"]
    _, attempt2_images = model_stack["sampler"].sample(
        model_stack["model"],
        model_stack["vae"],
        adaptive_context["tokens"][:1].to(device),
        adaptive_context["mask"][:1].to(device),
        device,
        seed=1,
    )

    idx = feedback["success_indices"][0]
    out_dir = runtime_config["out_dir"]
    gt_path = out_dir / "gt.png"
    attempt1_path = out_dir / "attempt1.png"
    attempt2_path = out_dir / "attempt2.png"
    grid_path = out_dir / "trace_grid.png"
    base_batch["gt_images"][idx].save(gt_path)
    attempt1[idx].save(attempt1_path)
    attempt2_images[0].save(attempt2_path)
    save_trace_grid(grid_path, base_batch["gt_images"][idx], attempt1[idx], feedback["texts"][0], attempt2_images[0])

    total_tokens = sum(result.token_count for result in feedback["results"])
    metrics = {
        "checkpoint": runtime_config["ckpt"],
        "dataset": runtime_config["dataset_path"],
        "model": runtime_config["model_name"],
        "batch_size": runtime_config["batch_size"],
        "success_count": len(feedback["success_indices"]),
        "failure_count": len(feedback["results"]) - len(feedback["success_indices"]),
        "feedback_failure_rate": (len(feedback["results"]) - len(feedback["success_indices"])) / max(len(feedback["results"]), 1),
        "loss": float(one_step_update["loss"].item()),
        "feedback_loss": float(one_step_update["feedback_loss"].item()),
        "base_loss": float(one_step_update["base_loss"].item()),
        "grad_norm": float(one_step_update["grad_norm"].item()),
        "gemini_total_tokens": int(total_tokens),
    }
    trace = {
        "caption": base_batch["batch"]["caption"][idx],
        "feedback": feedback["texts"][0],
        "gt_image": str(gt_path),
        "attempt1_image": str(attempt1_path),
        "attempt2_image": str(attempt2_path),
        "grid": str(grid_path),
    }
    save_json(out_dir / "metrics.json", metrics)
    save_json(out_dir / "trace.json", trace)

    assert gt_path.exists()
    assert attempt1_path.exists()
    assert attempt2_path.exists()
    assert grid_path.exists()
