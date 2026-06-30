# DiT CLEVR

This fork trains DiT on CLEVR latents from the SDXL VAE. The conditioned model uses frozen `Qwen/Qwen3.5-4B` context tokens and cross-attention inside DiT.

The VLM is frozen. Preprocessing caches Qwen hidden states and resized images in `data.zarr`; training does not run Qwen.
Caption templates are not selected separately in the scripts. Base context preprocessing writes one cached record per standard caption rendering, and training, feedback generation, and eval treat those rendered captions as ordinary data points.
Training image preprocessing is deterministic: each RGB image is resized to `model.image_size x model.image_size` with bicubic interpolation, converted to a tensor, then normalized to `[-1, 1]` for the SDXL VAE. There is no random crop, center crop, flip, color jitter, or other image augmentation in `train_text.py`.

## Files

- `preprocess_clevr_dit_dataset.py`: filter CLEVR scenes and write `metadata.jsonl`.
- `preprocess_clevr_context.py`: cache frozen Qwen context tokens and images to `data.zarr`.
- `scripts/convert_clevr_context_to_zarr.py`: convert older context shard datasets to `data.zarr`.
- `train_text.py`: train the Qwen-conditioned DiT.
- `configs/train_base.yaml`: Hydra config for base context training.
- `configs/train_adaptive.yaml`: Hydra config for base + feedback training.
- `clevr_transforms.py`: deterministic CLEVR resize/normalize preprocessing.
- `datasets_clevr.py`: context datasets, collate, and distributed weighted sampler.
- `vlm_utils.py`: VLM context text, metadata, loading, and encoding helpers.
- `generate_vlm_feedback_dataset.py`: sample images and collect Gemini feedback.
- `scripts/sample_clevr_eval_grid.py`: make fixed caption grids from cached context.
- `scripts/eval_adaptive_feedback_loop.py`: run iterative Gemini feedback eval.
- `scripts/sample_text.py`: sample from a raw caption by running Qwen online.

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## 1. Build CLEVR Rows

```bash
python preprocess_clevr_dit_dataset.py \
  --clevr-root /gpfs/scrubbed/sriyash/CLEVRDataset/CLEVR_v1.0 \
  --out /gpfs/scrubbed/sriyash/clevr_dit_dataset \
  --num-objects 3
```

## 2. Cache Base Qwen Context

```bash
python preprocess_clevr_context.py \
  --dataset /gpfs/scrubbed/sriyash/clevr_dit_dataset \
  --mode base \
  --vlm-model Qwen/Qwen3.5-4B \
  --max-context-len 1024 \
  --batch-size 4 \
  --overwrite
```

## 3. Train Base Model

```bash
torchrun --nnodes=1 --nproc_per_node=4 train_text.py \
  --config-name train_base
```

Edit `configs/train_base.yaml` or pass Hydra overrides for paths and hyperparameters, for example:

```bash
torchrun --nnodes=1 --nproc_per_node=1 train_text.py \
  --config-name train_base \
  train.experiment_name=base_s4_debug \
  train.max_train_steps=1000 \
  eval.max_batches_per_dataset=1
```

## 4. Generate Gemini Feedback Data

```bash
python generate_vlm_feedback_dataset.py \
  --ckpt /gpfs/scrubbed/sriyash/DiT-qwen-clevr-base/base_s4/checkpoints/0100000-ema.pt \
  --model DiT-S/4 \
  --vae stabilityai/sdxl-vae \
  --dataset-root /gpfs/scrubbed/sriyash/clevr_dit_dataset/data.zarr \
  --out-dir /gpfs/scrubbed/sriyash/vlm_feedback_dataset_qwen \
  --max-images 1000 \
  --samples-per-caption 5 \
  --feedbacks-per-image 5 \
  --cfg-scale 2.0 \
  --num-sampling-steps 100 \
  --vlm gemini \
  --use-caption
```

Set `OPENROUTER_API_KEY` before running Gemini feedback generation.

## 5. Cache Feedback Qwen Context

```bash
python preprocess_clevr_context.py \
  --dataset /gpfs/scrubbed/sriyash/vlm_feedback_dataset_qwen \
  --mode feedback \
  --vlm-model Qwen/Qwen3.5-4B \
  --max-context-len 1024 \
  --batch-size 4 \
  --overwrite
```

## 6. Finetune With Base + Feedback

```bash
torchrun --nnodes=1 --nproc_per_node=4 train_text.py \
  --config-name train_adaptive
```

`configs/train_adaptive.yaml` keeps the dataset schema as:

```yaml
data:
  dataset_config:
    - name: base
      dataset_path: /path/to/base/data.zarr
      sampling_ratio: 0.5
    - name: feedback
      dataset_path: /path/to/feedback/data.zarr
      sampling_ratio: 0.5
```

The sampler applies these ratios globally across the concatenated datasets.

## Dataset Regeneration

Changing the training resize preprocessing does not require regenerating `data.zarr` if `model.image_size` stays fixed. The cached dataset stores context tokens and resized RGB images. Regenerate when rows, splits, VLM model, prompt/context construction, or image size changes.

## 7. Evaluate

Cached validation grid:

```bash
python scripts/sample_clevr_eval_grid.py \
  --mode text \
  --ckpt /gpfs/scrubbed/sriyash/DiT-qwen-clevr-feedback/feedback_s4/checkpoints/0100000-ema.pt \
  --model DiT-S/4 \
  --vae stabilityai/sdxl-vae \
  --dataset-root /gpfs/scrubbed/sriyash/clevr_dit_dataset/data.zarr \
  --split val \
  --num-captions 5 \
  --samples-per-caption 4 \
  --cfg-scale 2.0 \
  --num-sampling-steps 100 \
  --out results/eval_samples/qwen_val_grid.png
```

Iterative feedback loop:

```bash
python scripts/eval_adaptive_feedback_loop.py \
  --ckpt /gpfs/scrubbed/sriyash/DiT-qwen-clevr-feedback/feedback_s4/checkpoints/0100000-ema.pt \
  --model DiT-S/4 \
  --vae stabilityai/sdxl-vae \
  --dataset-root /gpfs/scrubbed/sriyash/clevr_dit_dataset/data.zarr \
  --split val \
  --num-captions 5 \
  --steps 8 \
  --cfg-scale 2.0 \
  --num-sampling-steps 100 \
  --vlm-model Qwen/Qwen3.5-4B \
  --out-dir results/adaptive_feedback_loop_eval
```

Raw caption sample:

```bash
python scripts/sample_text.py \
  --ckpt /gpfs/scrubbed/sriyash/DiT-qwen-clevr-base/base_s4/checkpoints/0100000-ema.pt \
  --model DiT-S/4 \
  --caption "objects: small gray metal sphere, large yellow metal sphere, large blue rubber cube." \
  --cfg-scale 2.0 \
  --num-samples 4 \
  --out sample_text.png
```


current best ckpt: /gpfs/scrubbed/sriyash/DiT-qwen-clevr-base/base_l4_lr1e-4_minlr1e-5_warmup5k_ema999