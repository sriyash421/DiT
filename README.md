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

Download raw CLEVR first:

```bash
wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip && unzip CLEVR_v1.0.zip
```

Three stages turn raw CLEVR into a training-ready zarr (details in `datasets/clevr/README.md`):

```bash
python -m datasets.clevr.preprocess_clevr --clevr-root CLEVR_v1.0 --out <dataset_dir> --max-objects 3
python -m datasets.clevr.convert_to_zarr --dataset <dataset_dir>
python -m datasets.clevr.encode_context --zarr <dataset_dir>/data.zarr
```

Captions use the single `chain` template; stage 1 keeps every scene with at most `--max-objects`
objects. Stage 3 (frozen-Qwen context tokens) is only consumed by frozen-encoder offline training
(`train_adaptive`) and runs automatically at training startup when the zarr lacks tokens; running
it up front just moves the cost. On-policy training and the evals encode contexts on the fly.

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

The on-policy loop: sample attempts from the EMA policy, verify against the GT with the configured
verifier (`configs/verifier/`), encode the interleaved caption/attempt/feedback history with the Qwen
encoder, and run `updates_per_rollout` diffusion updates on the accepted records. `rollout.length` is
the number of predictions per rollout: length K makes K predictions and asks for feedback K-1 times
(feedback on the final attempt would never be used). A depth-k record trains on the history that
generated attempt k — caption-only at depth 0, caption + k feedback/attempt pairs after that.
On-policy launches also need `OPENROUTER_API_KEY` for the hardcoded eval distance scorer.

### On-policy LR: use the offline LR, not PPO's

The numeracy on-policy run was first configured at `lr: 1e-5`, chosen to sit inside the PPO
convention band (1e-5..5e-5). That was the wrong reference class and it cost ~270 iterations:

| | offline run (learned) | on-policy run (near-frozen) |
| --- | --- | --- |
| lr | **1e-4** | **1e-5** |
| global_batch_size | 32 | 64 |
| lora rank / alpha | 128 / 128 | 128 / 128 |

Identical adapter, 10x lower LR — and the result was +0.01 dense score over 272 iterations, with
the fixed-prompt/fixed-noise drift grid showing visually frozen generations.

PPO uses small LRs because the policy-gradient estimator is high-variance. This loop is **supervised
distillation**: the target is the ground-truth image, so the objective is the same flow-matching loss
as offline training. Anchor the LR to the offline run that worked, not to RL convention.

Current recommendation for on-policy configs:

- `lr: 1e-4` — match the offline stage.
- `updates_per_rollout: 64` — 2 epochs over the rollout buffer. Staleness is safe here for the same
  reason: the target is the GT image, so reusing rollouts does not bias the objective the way it
  would in off-policy RL (no importance weighting needed).
- Keep `stored_noise_prob: 1.0` and `rollout.length >= 4`. With fresh noise per step the model
  resamples instead of editing; the length-2 / fresh-noise run *declined* across rollout steps
  (0.714 -> 0.685), whereas fixed within-sequence noise at length 4 gives +0.009 from step 0 -> 3.

Before reaching for full finetuning to fix a stalled run, check the LR first, then run a capacity
probe (overfit ~32 prompts to convergence). Rank-128 LoRA on qkv/o/gate_up/down is 201M trainable
params — 5.6% of the transformer matrices — and already moved the base model to 0.742 dense offline,
so adapter capacity is rarely the binding constraint.

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
used by the `vllm_qwen` verifier). See `NOTES.md` for remaining oddities and intentional decisions.


now run a big test: so run this test: use the original prompt v1, best prompt v1, and v2 and v3. for models : qwen3.5-9B (with and without thinking), qwen3.5-27B (with and without thinking), qwen3.6-27B
  with and without thinking. so 4 prompts x 6 models. host the three models on 3x2 l40s simultaneously and run the 24 evals on ckpt partition! for ground truth verify each models predictions by using anthropic claude fable 5 on open router to get the ground truth! run the eval on 64 images each! generate images using the 100k chgeckpoint for this model: /gscratch/scrubbed/sriyash/OmniGen-clevr/omni_lora_finetune_five_objects on the 64 images in the val set. get the ground truth response using the claude model. and then compare the responses to the 24 images and present evereyrhing in a verifier_report.MD . the verifier should contain tables, explainations and some images to point behavior that i should look at! and also plots and figures to make me understand the comparison better. the goal is to compare performance in terms of accuracy, speed, throuhgput, output tokens, and other metrics useful for knowing which model to use for on-policy feedback generartion. given below are the recomended sampling parameters. strictly use them. also for thiking models set a higher token output length to avoid truncating the output! 
  for qwen3.6 model family:
  Sampling Parameters:

We suggest using the following sets of sampling parameters depending on the mode and task type:
Thinking mode for general tasks:
temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0
Instruct (or non-thinking) mode:
temperature=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0


for qwen3.5 model family:
Sampling Parameters:

We suggest using the following sets of sampling parameters depending on the mode and task type:
Thinking mode for general tasks:
temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0
Instruct (or non-thinking) mode for general tasks:
temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0