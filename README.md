# DiT CLEVR

Text-conditioned DiT trained on CLEVR in SDXL-VAE latent space, conditioned on frozen (or LoRA-finetuned)
`Qwen/Qwen3.5-4B` context tokens through cross-attention, with a VLM-feedback on-policy training loop.

## Layout

```
train.py                  # single hydra entry point for all training (config picks the trainer)
eval_generation.py        # GT|pred grids + Gemini distance-to-GT on train/val
eval_adaptive.py          # iterative feedback rollout eval
test.py                   # pytest suite (unit + integration)
algorithms/
  offline.py              # OfflineTrainer: supervised training/finetuning (QwenDiT and OmniGen)
  on_policy.py            # OnPolicyTrainer: rollout -> verify -> re-encode -> distill
  eval.py                 # val loss, adaptive rollout, batch selection, grids
  utils.py                # optimizer/scheduler, diffusion loss, EMA, checkpointing, logging
models/
  qwen_dit.py             # DiT + registry + QwenDiT wrapper (unified model interface)
  qwen_vlm.py             # Qwen VLM loading + context encoding (frozen or LoRA)
  omni_gen.py             # OmniGen wrapper behind the same interface (.venv-omni only)
datasets/
  __init__.py             # build_clevr_dataset factory used by configs
  clevr/                  # dataset classes + 3-stage preprocessing pipeline (see its README)
verifiers/
  base.py                 # prompts, parsers, FeedbackVerifier + OpenAI-chat backend
  gemini.py / vllm_qwen.py / open_router.py / local_qwen.py
  eval_metrics.py         # Gemini (OpenRouter) 0-9 distance-to-GT scorer
configs/                  # hydra configs; dataset/ and verifier/ groups
helper_scripts/           # one-off utilities (feedback dataset generation, dedup, grids, VAE recon)
diffusion/                # vendored gaussian diffusion library
```

Everything is built with `hydra.utils.instantiate` from `_target_` blocks in `configs/`. Which trainer
runs is just `trainer._target_` in the config.

## Setup

```bash
uv sync                 # creates .venv from pyproject.toml/uv.lock
source .venv/bin/activate
```

OmniGen training uses a separate env (`.venv-omni`) with the external OmniGen package installed
(`git clone https://github.com/VectorSpaceLab/OmniGen && pip install -e OmniGen`). Never install
HuggingFace `datasets` into `.venv` — the local `datasets/` package shadows it (see NOTES.md).

API keys come from environment variables: `GEMINI_API_KEY` (native Gemini verifier),
`OPENROUTER_API_KEY` (OpenRouter verifier + eval scorer).

## Data

Three stages turn raw CLEVR into a training-ready zarr (details in `datasets/clevr/README.md`):

```bash
python -m datasets.clevr.preprocess_clevr --clevr-root <CLEVR_v1.0> --out <dataset_dir> --num-objects 3
python -m datasets.clevr.convert_to_zarr --dataset <dataset_dir>
python -m datasets.clevr.encode_context --zarr <dataset_dir>/data.zarr
```

Captions use the single `chain` template. Stage 3 (frozen-Qwen context tokens) is optional if you train
with `model.context_encoder.freeze_encoder: false`, but precomputing is much faster for a frozen encoder.

## Training

```bash
torchrun --nproc_per_node=2 train.py --config-name train_base        # base DiT
torchrun --nproc_per_node=2 train.py --config-name train_adaptive    # base + feedback data mix
torchrun --nproc_per_node=1 train.py --config-name train_omni        # OmniGen LoRA/full finetune (.venv-omni)
torchrun --nproc_per_node=1 train.py --config-name train_on_policy   # on-policy feedback distillation
```

- Offline training uses cosine LR with warmup; on-policy uses fixed LRs (`pretrain_lr`, `lr`).
- All metrics log to wandb keyed by the global optimizer step.
- Checkpoints: `{model, ema, opt, scheduler?, context_encoder?}` at `<results_dir>/<experiment_name>/checkpoints/`,
  plus a standalone `-ema.pt`.
- Set `model.context_encoder.freeze_encoder: false` to LoRA-finetune the Qwen context encoder end-to-end;
  set `model.lora_finetune: false` in `train_omni` for full OmniGen finetuning.

The on-policy loop: sample attempts from the EMA policy, verify each against the GT with the configured
verifier (`configs/verifier/`), re-encode caption+feedback history with the frozen Qwen encoder, and run
`updates_per_rollout` diffusion updates on the accepted records. Step-0 records train caption-only
conditioning; later steps train feedback conditioning (`rollout.length` controls depth).

## Evaluation

```bash
python eval_generation.py ckpt=<ckpt>                 # GT|pred grids + mean distance on train/val
python eval_adaptive.py ckpt=<ckpt> verifier=vllm_qwen  # iterative feedback rollouts + trace grids
```

The distance metric is a Gemini (`google/gemini-3.1-flash-lite` via OpenRouter) estimate of the number of
object edits needed to turn the prediction into the GT (0 = match, 9 = worst).

## Tests

```bash
pytest test.py -m "not integration" -q     # CPU unit suite
pytest test.py -q                          # + integration (CUDA, GEMINI_API_KEY, real dataset/ckpt)
```

## Cluster

`job_*.sh` are the slurm launchers (train, on-policy, feedback generation, evals, and the vLLM server
used by the `vllm_qwen` verifier). See `NOTES.md` for the post-refactor review list.
