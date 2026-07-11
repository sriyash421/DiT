"""Tests for datasets, models, verifiers, and trainers. Run: pytest test.py -m "not integration" -q"""
import hashlib
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from algorithms.on_policy import (
    OnPolicyTrainer,
    PolicySampler,
    RolloutCollector,
    all_reduce_rollout_stats,
    exact_update_batches,
)
from algorithms.utils import (
    build_optimizer_scheduler,
    diffusion_loss,
    load_checkpoint,
    requires_grad,
    save_trace_grid,
    unwrap_model,
    update_ema,
    write_json,
)
from datasets import build_clevr_dataset
from datasets.clevr.dataset import ClevrContextDataset, context_collate, pad_contexts
from datasets.clevr.utils import (
    ClevrZarrWriter,
    ContextTokenWriter,
    build_clevr_transform,
    render_caption,
    zarr_has_context,
)
from datasets.rollouts import RolloutBuffer, rollout_collate
from diffusion import create_diffusion
from models.omni_gen import training_losses as omni_training_losses
from models.qwen_dit import DiT, DiT_models
from models.qwen_vlm import build_history_messages
from verifiers import build_verifier
from verifiers.base import (
    FeedbackVerifier,
    VerificationResult,
    build_distance_prompt,
    build_feedback_prompt,
    clean_feedback_text,
    normalize_chat_url,
    parse_distance_score,
)
from verifiers.gemini import DEFAULT_GEMINI_MODEL, GeminiVerifier
from verifiers.open_router import OpenRouterVerifier
from verifiers.vllm_qwen import VLLMQwenVerifier


def env(name, default=None):
    return os.getenv(name, default)


def latest_checkpoint(root):
    root = Path(root)
    if not root.exists():
        return None
    candidates = sorted(root.glob("**/*-ema.pt"))
    return str(candidates[-1]) if candidates else None


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


# ---------------------------------------------------------------------------
# Fakes.
# ---------------------------------------------------------------------------

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
    def __init__(self):
        self.calls = 0

    def verify(self, captions, gt_images, attempt_images, feedback_histories=None):
        del gt_images, attempt_images
        self.calls += 1
        if feedback_histories is None:
            feedback_histories = [[] for _ in captions]
        return [
            VerificationResult(ok=True, feedback=f"fix {caption} step {len(history)}", token_count=11)
            for caption, history in zip(captions, feedback_histories)
        ]


class FakeContextEncoder:
    """Encodes each row to 3 tokens filled with its history length, so tests can check
    which history depth a context was encoded from."""

    freeze = True

    def train(self):
        pass

    def eval(self):
        pass

    def prepare_row(self, caption, feedback_history, attempt_image_history):
        return {"depth": len(feedback_history)}

    def forward(self, rows):
        tokens = torch.stack([torch.full((3, 3), float(row["depth"])) for row in rows])
        return tokens, torch.ones(len(rows), 3, dtype=torch.bool)

    def encode_history(self, captions, feedback_histories, attempt_image_histories):
        assert len(captions) == len(feedback_histories) == len(attempt_image_histories)
        return self.forward([self.prepare_row(c, h, i) for c, h, i in zip(captions, feedback_histories, attempt_image_histories)])


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


def make_rollout_record(idx, attempt_path=None, with_context=True):
    record = {
        "x_latent": torch.full((4, 2, 2), float(idx)),
        "caption": f"caption {idx}",
        "feedback": f"feedback {idx}",
        "feedback_history": [],
        "history_attempt_paths": [],
        "gt_path": f"gt-{idx}.png",
        "attempt_path": str(attempt_path or f"attempt-{idx}.png"),
    }
    if with_context:
        record["context_tokens"] = torch.ones(3, 3) * (idx + 10)
    return record


def save_rollout_payload(path, start, count, tokens=0, with_context=True):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    records = [make_rollout_record(start + idx, with_context=with_context) for idx in range(count)]
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


# ---------------------------------------------------------------------------
# Rollout buffer / collector.
# ---------------------------------------------------------------------------

def test_rollout_buffer_merges_rank_shards(tmp_path):
    step_dir = tmp_path / "step_000001"
    save_rollout_payload(step_dir / "rank_000", 0, 2, tokens=5)
    save_rollout_payload(step_dir / "rank_001", 2, 3, tokens=7)

    dataset = RolloutBuffer(step_dir)

    assert len(dataset) == 5
    assert [dataset[idx]["caption"] for idx in range(len(dataset))] == [f"caption {idx}" for idx in range(5)]
    assert dataset.stats["attempted"] == 5
    assert dataset.stats["success"] == 5
    assert dataset.stats["gemini_tokens"] == 12


def test_rollout_collector_length_one_never_verifies(tmp_path):
    batch = {
        "image": torch.zeros(2, 3, 8, 8),
        "caption": ["a", "b"],
    }
    data_sampler = RecordingDistributedSampler(TensorDataset(torch.arange(2)))
    verifier = FakeVerifier()
    collector = RolloutCollector(
        model="ema-model",
        vae=FakeVAE(),
        sampler=FakeSampler(),
        verifier=verifier,
        encoder=FakeContextEncoder(),
        preprocess_context=True,
        rollout_length=1,
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

    assert stats == {"attempted": 2, "success": 2, "failed": 0, "gemini_tokens": 0}
    assert verifier.calls == 0
    assert len(dataset) == 2
    assert data_sampler.epochs == [3]
    assert (tmp_path / "step_000001" / "rank_000" / "records.pt").exists()
    assert (tmp_path / "step_000001" / "rank_000" / "gt" / "000000.png").exists()
    assert (tmp_path / "step_000001" / "rank_000" / "attempts" / "000001_step_00.png").exists()
    assert dataset[0]["gt_path"].endswith("gt/000000.png")
    assert dataset[0]["step_index"] == 0
    assert dataset[0]["feedback"] == ""
    # Depth-0 records store the caption-only context (history length 0).
    assert torch.equal(dataset[0]["context_tokens"], torch.zeros(3, 3, dtype=torch.float16))


def test_rollout_collector_multi_step_verifies_k_minus_one_times(tmp_path):
    batch = {
        "image": torch.zeros(2, 3, 8, 8),
        "caption": ["a", "b"],
    }
    verifier = FakeVerifier()
    collector = RolloutCollector(
        model="ema-model",
        vae=FakeVAE(),
        sampler=FakeSampler(),
        verifier=verifier,
        encoder=FakeContextEncoder(),
        preprocess_context=True,
        rollout_length=3,
    )

    stats = collector.collect(
        [batch],
        tmp_path / "step_000001" / "rank_000",
        sample_count=2,
        device=torch.device("cpu"),
    )
    dataset = RolloutBuffer(tmp_path / "step_000001")

    # K=3 predictions per sample, K-1=2 verifier rounds, K records per sample.
    assert stats == {"attempted": 6, "success": 6, "failed": 0, "gemini_tokens": 44}
    assert verifier.calls == 2
    assert len(dataset) == 6
    assert [dataset[idx]["step_index"] for idx in range(len(dataset))] == [0, 0, 1, 1, 2, 2]
    assert dataset[0]["feedback_history"] == []
    assert dataset[0]["feedback"] == "fix a step 0"
    assert dataset[2]["feedback_history"] == ["fix a step 0"]
    assert len(dataset[4]["history_attempt_paths"]) == 2
    assert dataset[4]["feedback"] == ""
    assert Path(dataset[4]["attempt_path"]).exists()
    # A depth-k record stores the context encoded from its pre-attempt history of length k.
    assert torch.equal(dataset[0]["context_tokens"], torch.zeros(3, 3, dtype=torch.float16))
    assert torch.equal(dataset[2]["context_tokens"], torch.full((3, 3), 1.0, dtype=torch.float16))
    assert torch.equal(dataset[4]["context_tokens"], torch.full((3, 3), 2.0, dtype=torch.float16))


def test_rollout_collector_live_records_carry_no_tokens(tmp_path):
    batch = {
        "image": torch.zeros(1, 3, 8, 8),
        "caption": ["a"],
    }
    collector = RolloutCollector(
        model="raw-model",
        vae=FakeVAE(),
        sampler=FakeSampler(),
        verifier=FakeVerifier(),
        encoder=FakeContextEncoder(),
        preprocess_context=False,
        rollout_length=2,
    )

    collector.collect(
        [batch],
        tmp_path / "step_000001" / "rank_000",
        sample_count=1,
        device=torch.device("cpu"),
    )
    dataset = RolloutBuffer(tmp_path / "step_000001", prepare_fn=FakeContextEncoder().prepare_row)

    assert len(dataset) == 2
    assert all("context_tokens" not in record for record in dataset.records)
    # __getitem__ tokenizes and caches the context inputs on the record once.
    dataset[1]
    assert "context_inputs" in dataset.records[1]
    batch_out = rollout_collate([dataset[0], dataset[1]])
    assert "context_tokens" not in batch_out
    assert len(batch_out["context_inputs"]) == 2


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
        "caption": ["a"],
    }
    for use_ema, expected_model in ((True, "ema"), (False, "raw")):
        fake_sampler = FakeSampler()
        collector = RolloutCollector(
            model="ema" if use_ema else "raw",
            vae=FakeVAE(),
            sampler=fake_sampler,
            verifier=FakeVerifier(),
            encoder=FakeContextEncoder(),
            preprocess_context=True,
        )
        collector.collect(
            [batch],
            tmp_path / f"use_ema_{use_ema}" / "rank_000",
            sample_count=1,
            device=torch.device("cpu"),
        )
        assert fake_sampler.models == [expected_model]


def test_all_reduce_rollout_stats_keeps_token_total_without_dist():
    stats = {"attempted": 2, "success": 1, "failed": 1, "gemini_tokens": 33}

    assert all_reduce_rollout_stats(stats, torch.device("cpu")) == stats


def test_rollout_collate_pads_records_from_disk_buffer(tmp_path):
    step_dir = tmp_path / "step_000001"
    save_rollout_payload(step_dir / "rank_000", 0, 2)
    dataset = RolloutBuffer(step_dir)
    batch = rollout_collate([dataset[0], dataset[1]])

    assert batch["x_latent"].shape == (2, 4, 2, 2)
    assert batch["context_tokens"].shape == (2, 3, 3)
    assert batch["context_mask"].all()


# ---------------------------------------------------------------------------
# Models.
# ---------------------------------------------------------------------------

def tiny_dit():
    torch.manual_seed(0)
    return DiT(input_size=8, patch_size=4, hidden_size=32, depth=1, num_heads=2, context_dim=16)


def test_dit_forward_shapes():
    model = tiny_dit()
    x = torch.randn(2, 4, 8, 8)
    t = torch.randint(0, 1000, (2,))
    tokens = torch.randn(2, 5, 16)
    mask = torch.ones(2, 5, dtype=torch.bool)
    out = model(x, t, context_tokens=tokens, context_mask=mask)
    assert out.shape == (2, 8, 8, 8)


def test_dit_forward_with_cfg_shapes():
    model = tiny_dit()
    x = torch.randn(4, 4, 8, 8)
    t = torch.randint(0, 1000, (4,))
    tokens = torch.randn(4, 5, 16)
    mask = torch.ones(4, 5, dtype=torch.bool)
    out = model.forward_with_cfg(x, t, cfg_scale=2.0, context_tokens=tokens, context_mask=mask)
    assert out.shape == (4, 8, 8, 8)
    eps = out[:, :3]
    assert torch.allclose(eps[:2], eps[2:])


def test_dit_param_names_match_pre_refactor_checkpoints():
    # Fingerprint of DiT-S/8 parameter names. If this changes, existing
    # checkpoints will no longer load with strict=True.
    model = DiT_models["DiT-S/8"](input_size=8, context_dim=16)
    names = "\n".join(sorted(name for name, _ in model.named_parameters()))
    fingerprint = hashlib.sha256(names.encode()).hexdigest()
    assert fingerprint == PARAM_NAME_FINGERPRINT, "DiT parameter names changed; old checkpoints will break."


def test_diffusion_loss_and_compute_loss():
    model = tiny_dit()
    diffusion = create_diffusion(timestep_respacing="")
    x_latent = torch.randn(2, 4, 8, 8)
    tokens = torch.randn(2, 5, 16)
    mask = torch.ones(2, 5, dtype=torch.bool)

    torch.manual_seed(0)
    loss = diffusion_loss(model, diffusion, x_latent, tokens, mask)
    assert torch.isfinite(loss)

    trainer = OnPolicyTrainer.__new__(OnPolicyTrainer)
    trainer.model = SimpleNamespace(net=model, diffusion=diffusion)
    trainer.encoder = FakeContextEncoder()
    trainer.device = torch.device("cpu")
    batch = {"x_latent": x_latent, "context_tokens": tokens, "context_mask": mask}
    total, stats = trainer.compute_loss(batch)
    assert torch.isfinite(total)
    assert stats["loss"] == pytest.approx(float(total.item()))


def test_omni_vendored_training_losses():
    model = lambda xt, t, **kwargs: xt
    x1 = torch.randn(3, 4, 8, 8)
    loss = omni_training_losses(model, x1, {})["loss"]
    assert loss.shape == (3,)
    assert torch.isfinite(loss).all()


def test_omni_module_never_imports_omnigen_train_helper():
    import sys

    import models.omni_gen

    assert "OmniGen.train_helper" not in sys.modules


# ---------------------------------------------------------------------------
# Optimizer / scheduler.
# ---------------------------------------------------------------------------

def test_build_optimizer_scheduler_fixed_lr():
    params = [torch.nn.Parameter(torch.zeros(2))]
    opt, scheduler = build_optimizer_scheduler(params, lr=1e-3, weight_decay=0.0)
    assert scheduler is None
    assert opt.param_groups[0]["lr"] == 1e-3


def test_build_optimizer_scheduler_cosine_warmup():
    params = [torch.nn.Parameter(torch.zeros(2))]
    opt, scheduler = build_optimizer_scheduler(
        params, lr=1.0, weight_decay=0.0, schedule="cosine", total_steps=100, warmup_steps=10, min_lr=0.1
    )
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        scheduler.step()
    assert lrs[0] == pytest.approx(0.1, abs=1e-6)
    assert lrs[10] == pytest.approx(1.0, abs=1e-6)
    assert lrs[-1] == pytest.approx(0.1, abs=1e-2)


# ---------------------------------------------------------------------------
# Datasets: captions, zarr pipeline, collate.
# ---------------------------------------------------------------------------

CAPTION_ROW = {
    "objects": [
        {"size": "large", "color": "red", "material": "rubber", "shape": "cube"},
        {"size": "small", "color": "blue", "material": "metal", "shape": "sphere"},
    ],
    "orders": {"left_to_right": [0, 1], "front_to_back": [1, 0]},
}


def test_render_caption_chain_template():
    caption = render_caption(CAPTION_ROW)
    assert caption == (
        "objects: large red rubber cube, small blue metal sphere. "
        "horizontal: blue sphere is right of red cube. "
        "depth: red cube is behind blue sphere."
    )


def test_render_caption_single_object_drops_empty_chains():
    row = {
        "objects": [{"size": "large", "color": "red", "material": "rubber", "shape": "cube"}],
        "orders": {"left_to_right": [0], "front_to_back": [0]},
    }
    assert render_caption(row) == "objects: large red rubber cube."


def make_tiny_zarr(tmp_path, rows=4, image_size=16, context_dim=8):
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for idx in range(rows):
        image_path = image_dir / f"{idx}.png"
        Image.new("RGB", (32, 32), (idx * 20, 100, 50)).save(image_path)
        records.append({
            "dataset_type": "base",
            "metadata_index": idx,
            "image_path": f"images/{idx}.png",
            "split": "train" if idx < rows - 1 else "val",
            "caption": f"caption {idx}",
        })
    writer = ClevrZarrWriter(
        output=tmp_path / "data.zarr",
        dataset_root=tmp_path,
        dataset_type="base",
        image_size=image_size,
    )
    writer.append_batch(records)

    token_writer = ContextTokenWriter(
        zarr_path=tmp_path / "data.zarr",
        vlm_model="test-vlm",
        context_dim=context_dim,
        max_context_len=32,
        dtype="float16",
    )
    torch.manual_seed(0)
    token_writer.append_batch([torch.randn(3 + idx, context_dim) for idx in range(rows)])
    return tmp_path / "data.zarr"


def test_zarr_pipeline_roundtrip(tmp_path):
    zarr_path = make_tiny_zarr(tmp_path)
    dataset = ClevrContextDataset(zarr_path, transform=build_clevr_transform(16), split="train")

    assert len(dataset) == 3
    assert dataset.context_dim == 8
    assert zarr_has_context(zarr_path)
    item = dataset[0]
    assert item["image"].shape == (3, 16, 16)
    assert item["context_tokens"].shape == (3, 8)
    assert item["caption"] == "caption 0"
    assert not item["is_feedback"]

    val_dataset = ClevrContextDataset(zarr_path, split="val")
    assert len(val_dataset) == 1

    batch = context_collate([dataset[0], dataset[1]])
    assert batch["image"].shape == (2, 3, 16, 16)
    assert batch["context_tokens"].shape == (2, 4, 8)
    assert batch["context_mask"].tolist() == [[True, True, True, False], [True, True, True, True]]


def test_stage2_only_zarr_has_no_context(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (10, 20, 30)).save(image_dir / "0.png")
    writer = ClevrZarrWriter(
        output=tmp_path / "data.zarr",
        dataset_root=tmp_path,
        dataset_type="base",
        image_size=16,
    )
    writer.append_batch([{
        "dataset_type": "base",
        "metadata_index": 0,
        "image_path": "images/0.png",
        "split": "train",
        "caption": "caption 0",
    }])

    dataset = ClevrContextDataset(tmp_path / "data.zarr", transform=build_clevr_transform(16), split="train")
    assert not dataset.has_context
    assert not zarr_has_context(tmp_path / "data.zarr")
    assert dataset.context_dim is None
    item = dataset[0]
    assert "context_tokens" not in item
    batch = context_collate([item])
    assert "context_tokens" not in batch
    assert batch["caption"] == ["caption 0"]


def test_context_tokens_stay_float16_through_collate(tmp_path):
    zarr_path = make_tiny_zarr(tmp_path)
    dataset = ClevrContextDataset(zarr_path, transform=build_clevr_transform(16), split="train")
    item = dataset[0]
    assert item["context_tokens"].dtype == torch.float16
    batch = context_collate([dataset[0], dataset[1]])
    assert batch["context_tokens"].dtype == torch.float16


def test_build_clevr_dataset_factory(tmp_path):
    zarr_path = make_tiny_zarr(tmp_path)
    dataset = build_clevr_dataset(
        datasets=[{"name": "base", "path": str(zarr_path), "sampling_ratio": 1.0, "max_dataset_size": None}],
        split="train",
        image_size=16,
    )
    assert len(dataset) == 3
    assert dataset.context_dim == 8
    assert dataset.sample_weights.shape == (3,)
    assert dataset.image_for_index(0).size == (16, 16)


def test_pad_contexts_shapes_and_mask():
    tokens, mask = pad_contexts([torch.ones(2, 4), torch.ones(5, 4)])
    assert tokens.shape == (2, 5, 4)
    assert mask.sum().item() == 7


# ---------------------------------------------------------------------------
# Verifiers.
# ---------------------------------------------------------------------------

def test_feedback_prompt_mentions_priority_and_word_limit():
    prompt = build_feedback_prompt("a red cube")
    assert "Caption: a red cube" in prompt
    assert "missing/extra object > shape > color > size > material > position/depth > background" in prompt
    assert "under 12 words" in prompt
    assert "return exactly: no update" in prompt
    assert "Previous feedback already given:" not in prompt


def test_feedback_prompt_with_history_includes_past_feedback():
    prompt = build_feedback_prompt("a red cube", feedback_history=["add a sphere"])
    assert "Caption: a red cube" in prompt
    assert "Previous feedback already given:" in prompt
    assert "- add a sphere" in prompt
    assert "Do not repeat a previous command" in prompt
    assert "return exactly: no update" in prompt


def test_distance_prompt_asks_for_single_digit():
    prompt = build_distance_prompt("a red cube")
    assert "Caption: a red cube" in prompt
    assert "Return only one integer from 0 to 9." in prompt


def test_parse_distance_score():
    assert parse_distance_score("3") == 3.0
    assert parse_distance_score("The answer is 7.") == 7.0
    assert parse_distance_score("Image 2 needs 3 edits") == 3.0
    with pytest.raises(ValueError):
        parse_distance_score("no digits here")


def test_clean_feedback_text():
    assert clean_feedback_text("<think>hmm</think>Add a red cube") == "Add a red cube"
    assert clean_feedback_text("blah blah FINAL: move the sphere left") == "move the sphere left"
    assert clean_feedback_text("- change cube to sphere") == "change cube to sphere"


def test_normalize_chat_url():
    assert normalize_chat_url("http://host:8000/v1") == "http://host:8000/v1/chat/completions"
    assert normalize_chat_url("http://host:8000/v1/chat/completions") == "http://host:8000/v1/chat/completions"
    assert normalize_chat_url("http://host:8000") == "http://host:8000/v1/chat/completions"


def test_build_verifier_dispatch(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    assert isinstance(build_verifier("gemini"), GeminiVerifier)
    assert isinstance(build_verifier("vllm-qwen", api_url="http://host:8000/v1", model="qwen"), VLLMQwenVerifier)
    assert isinstance(build_verifier("open-router", model="google/gemini-3.1-flash-lite"), OpenRouterVerifier)
    with pytest.raises(ValueError):
        build_verifier("unknown")


def test_vllm_verifier_sets_thinking_payload(monkeypatch):
    verifier = VLLMQwenVerifier(api_url="http://host:8000/v1", model="qwen", enable_thinking=True)
    assert verifier.extra_payload == {"chat_template_kwargs": {"enable_thinking": True}}


def test_threaded_verify_preserves_order():
    class SlowVerifier(FeedbackVerifier):
        workers = 4

        def _verify_row(self, caption, gt_image, attempt_image, feedback_history):
            time.sleep(0.01 * (3 - int(caption)))
            return VerificationResult(ok=True, feedback=caption)

    verifier = SlowVerifier()
    results = verifier.verify(["0", "1", "2"], [None] * 3, [None] * 3)
    assert [result.feedback for result in results] == ["0", "1", "2"]


def test_base_verify_threads_history_and_distance_raises():
    class Plain(FeedbackVerifier):
        def _verify_row(self, caption, gt_image, attempt_image, feedback_history):
            return VerificationResult(ok=True, feedback=f"{caption}|{len(feedback_history)}")

    assert Plain().verify(["a"], [None], [None])[0].feedback == "a|0"
    assert Plain().verify(["a"], [None], [None], [["old feedback"]])[0].feedback == "a|1"
    with pytest.raises(NotImplementedError):
        Plain().score_distance(["a"], [None], [None])


def test_build_history_messages_asserts_matching_lengths():
    messages = build_history_messages(["c"], [[]], [[]])
    assert messages[0][0]["content"] == [{"type": "text", "text": "Caption: c"}]
    with pytest.raises(AssertionError):
        build_history_messages(["a"], [["f"]], [[]])
    with pytest.raises(AssertionError):
        build_history_messages(["a", "b"], [[]], [[]])


# ---------------------------------------------------------------------------
# Integration tests (CUDA + GEMINI_API_KEY + real dataset/checkpoint).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def runtime_config():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the on-policy integration tests.")
    api_key_env = env("ON_POLICY_TEST_API_KEY_ENV", "GEMINI_API_KEY")
    if not os.getenv(api_key_env):
        pytest.skip(f"{api_key_env} is required for Gemini verification.")

    dataset_path = env("ON_POLICY_TEST_DATASET", "/gscratch/scrubbed/sriyash/clevr_dit_dataset/data.zarr")
    if not Path(dataset_path).exists():
        pytest.skip(f"Dataset not found: {dataset_path}")

    ckpt = env("ON_POLICY_TEST_CKPT")
    if ckpt is None:
        ckpt = latest_checkpoint(env("ON_POLICY_TEST_CKPT_ROOT", "/gscratch/scrubbed/sriyash/DiT-qwen-clevr-base"))
    if ckpt is None or not Path(ckpt).exists():
        pytest.skip("Set ON_POLICY_TEST_CKPT to a valid DiT checkpoint.")

    out_dir = Path(env("ON_POLICY_TEST_OUT", "results/on_policy_test")) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "api_key_env": api_key_env,
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
    gt_images = [dataset.image_for_index(idx) for idx in range(runtime_config["batch_size"])]
    return {
        "dataset": dataset,
        "batch": batch,
        "gt_images": gt_images,
    }


@pytest.fixture(scope="session")
def model_stack(runtime_config, base_batch):
    from diffusers.models import AutoencoderKL

    device = runtime_config["device"]
    dataset = base_batch["dataset"]
    model = DiT_models[runtime_config["model_name"]](
        input_size=runtime_config["image_size"] // 8,
        context_dim=dataset.context_dim,
        context_dropout_prob=0.1,
    ).to(device)
    model.load_state_dict(load_checkpoint(runtime_config["ckpt"]), strict=False)
    model.train()
    ema = DiT_models[runtime_config["model_name"]](
        input_size=runtime_config["image_size"] // 8,
        context_dim=dataset.context_dim,
        context_dropout_prob=0.1,
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
def gemini_verifier(runtime_config):
    return GeminiVerifier(
        api_key_env=runtime_config["api_key_env"],
        model=env("ON_POLICY_TEST_GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        workers=int(env("ON_POLICY_TEST_VERIFIER_WORKERS", "4")),
        max_tokens=int(env("ON_POLICY_TEST_MAX_FEEDBACK_TOKENS", "96")),
        temperature=float(env("ON_POLICY_TEST_FEEDBACK_TEMPERATURE", "0.0")),
    )


@pytest.fixture(scope="session")
def context_encoder(runtime_config):
    from models.qwen_vlm import QwenEncoder

    return QwenEncoder(
        env("ON_POLICY_TEST_QWEN_MODEL", "Qwen/Qwen3.5-4B"),
        device="cuda",
        dtype=env("ON_POLICY_TEST_QWEN_DTYPE", "bfloat16"),
        device_map=env("ON_POLICY_TEST_QWEN_DEVICE_MAP", "auto"),
        max_length=int(env("ON_POLICY_TEST_MAX_CONTEXT_LEN", "1024")),
        freeze=True,
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
def feedback(runtime_config, base_batch, gemini_verifier, attempt1):
    batch = base_batch["batch"]
    results = gemini_verifier.verify(
        batch["caption"],
        base_batch["gt_images"],
        attempt1,
    )
    success_indices = [idx for idx, result in enumerate(results) if result.ok]
    if not success_indices:
        errors = [result.error for result in results]
        write_json(runtime_config["out_dir"] / "verifier_failures.json", {
            "errors": errors,
            "captions": batch["caption"],
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
    tokens, mask = context_encoder.encode_history(
        [batch["caption"][idx] for idx in success_indices],
        [[text] for text in feedback["texts"]],
        [[attempt1[idx]] for idx in success_indices],
    )
    return {"tokens": tokens.detach().cpu().to(torch.float16), "mask": mask.detach().cpu()}


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
    base_loss = diffusion_loss(model, model_stack["train_diffusion"], x_latent, base_tokens, base_mask)
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
    grid_path = out_dir / "trace_grid.png"
    save_trace_grid(grid_path, base_batch["gt_images"][idx], attempt1[idx], feedback["texts"][0], attempt2_images[0])
    write_json(out_dir / "metrics.json", {
        "checkpoint": runtime_config["ckpt"],
        "success_count": len(feedback["success_indices"]),
        "loss": float(one_step_update["loss"].item()),
        "grad_norm": float(one_step_update["grad_norm"].item()),
        "verifier_total_tokens": int(sum(result.token_count for result in feedback["results"])),
    })
    assert grid_path.exists()


PARAM_NAME_FINGERPRINT = "b5392d18f092f1ddc774156d1cbddc06446ccd42db136a634e05119a1882e20f"
