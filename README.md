## Scalable Diffusion Models with Transformers (DiT)<br>

## Setup

```bash
uv venv
uv pip install -r requirements.txt
```

## To debug env
```bash
python sample.py --image-size 512 --seed 1
```

## CLEVR Text-Conditioned Fine-Tuning

This fork also supports fine-tuning `DiT-XL/2` on 3-object CLEVR images with frozen FLAN-T5 caption embeddings.

```bash
python preprocess_clevr_dit_dataset.py \
  --clevr-root /gpfs/scrubbed/sriyash/CLEVRDataset/CLEVR_v1.0 \
  --out /gpfs/scrubbed/sriyash/clevr_dit_dataset \
  --num-objects 3
```

```bash
python preprocess_clevr_text_embeddings.py \
  --dataset /gpfs/scrubbed/sriyash/clevr_dit_dataset \
  --encoder google/flan-t5-large \
  --templates chain order compact \
  --max-length 128 \
  --batch-size 32
```

```bash
torchrun --nnodes=1 --nproc_per_node=2 train_text.py \
  --data-path /gpfs/scrubbed/sriyash/clevr_dit_dataset \
  --results-dir /gpfs/scrubbed/sriyash/DiT-clevr-text-final \
  --model DiT-S/4 \
  --image-size 256 \
  --vae stabilityai/sdxl-vae \
  --from-scratch \
  --global-batch-size 128 \
  --num-workers 8 \
  --lr 3e-4 \
  --grad-clip 1.0 \
  --ema-decay 0.999 \
  --log-every 100 \
  --ckpt-every 2000 \
  --max-train-steps 100000 \
  --wandb-project DiT-clevr-text-final
```

```bash
python sample_text.py \
  --ckpt /tmp/dit_0030000_ema_sample.pt \
  --caption "objects: small gray metal sphere, large yellow metal sphere, large blue rubber cube. horizontal: yellow sphere is right of blue cube, gray sphere is right of yellow sphere. depth: gray sphere is behind yellow sphere, blue cube is behind gray sphere."
```


## Trained models
results/slurm-dit_clevr_text-134848.out - pretrained finetune / low lr
results/slurm-scratch-unconditional-dit_clevr_uncond-134867.out
134888 -- direct conditioning/text/scratch small model
https://wandb.ai/sriyash-uw-team/DiT-clevr-text-final/runs/ohbhe9n6
</br>

## Evaluation
1. Pretrained text fine-tune, train captions, 5x4

python sample_clevr_eval_grid.py \
  --mode text \
  --ckpt /gpfs/scrubbed/sriyash/DiT-clevr-results/003-DiT-XL-2-text/checkpoints/0029000-ema.pt \
  --model DiT-XL/2 \
  --vae stabilityai/sd-vae-ft-mse \
  --split train \
  --num-captions 5 \
  --samples-per-caption 4 \
  --caption-seed 0 \
  --seed 0 \
  --cfg-scale 1.0 \
  --num-sampling-steps 250 \
  --out results/eval_samples/pretrained_text_xl2_0029000_train_5x4.png

2. Pretrained text fine-tune, val captions, 5x4

python sample_clevr_eval_grid.py \
  --mode text \
  --ckpt /gpfs/scrubbed/sriyash/DiT-clevr-results/003-DiT-XL-2-text/checkpoints/0029000-ema.pt \
  --model DiT-XL/2 \
  --vae stabilityai/sd-vae-ft-mse \
  --split val \
  --num-captions 5 \
  --samples-per-caption 4 \
  --caption-seed 0 \
  --seed 0 \
  --cfg-scale 5.0 \
  --num-sampling-steps 250 \
  --out results/eval_samples/pretrained_text_xl2_0029000_val_5x4_5.0.png

3. Scratch text S/4, train captions, 5x4

python sample_clevr_eval_grid.py \
  --mode text \
  --ckpt /gpfs/scrubbed/sriyash/DiT-clevr-text-final/000-DiT-S-4-text/checkpoints/0100000-ema.pt \
  --model DiT-S/4 \
  --vae stabilityai/sdxl-vae \
  --split train \
  --num-captions 5 \
  --samples-per-caption 4 \
  --caption-seed 0 \
  --seed 0 \
  --cfg-scale 1.0 \
  --num-sampling-steps 250 \
  --out results/eval_samples/scratch_text_s4_0100000_train_5x4.png

4. Scratch text S/4, val captions, 5x4

python sample_clevr_eval_grid.py \
  --mode text \
  --ckpt /gpfs/scrubbed/sriyash/DiT-clevr-text-final/000-DiT-S-4-text/checkpoints/0100000-ema.pt \
  --model DiT-S/4 \
  --vae stabilityai/sdxl-vae \
  --split val \
  --num-captions 5 \
  --samples-per-caption 4 \
  --caption-seed 0 \
  --seed 0 \
  --cfg-scale 1.0 \
  --num-sampling-steps 250 \
  --out results/eval_samples/scratch_text_s4_0100000_val_5x4.png

5. Scratch unconditional S/4, random 5x4

python sample_clevr_eval_grid.py \
  --mode uncond \
  --ckpt /gpfs/scrubbed/sriyash/DiT-clevr-uncond-results/002-DiT-S-4/checkpoints/0096000.pt \
  --model DiT-S/4 \
  --vae stabilityai/sdxl-vae \
  --num-captions 5 \
  --samples-per-caption 4 \
  --seed 0 \
  --num-sampling-steps 250 \
  --out results/eval_samples/scratch_uncond_s4_0096000_5x4.png

</br>

# Evaluation metrics

</br>


# Hosting VLM Feedback servers

Run these on the GPU node that will host the VLM. The feedback script can run on a different node and call the server with `--vlm-server-url`.

## Qwen2.5-VL-72B via vLLM

```bash
vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --limit-mm-per-prompt '{"image": 2}' \
  --max-model-len 8192
```

Call it from the feedback script:

```bash
python scripts/vlm_feedback_clevr.py \
  --vlm qwen70b \
  --vlm-server-url http://g001:4000/v1 \
  --ckpt /gpfs/scrubbed/sriyash/DiT-clevr-text-final/000-DiT-S-4-text/checkpoints/0100000-ema.pt \
  --model DiT-S/4 \
  --vae stabilityai/sdxl-vae \
  --dataset-root /gpfs/scrubbed/sriyash/clevr_dit_dataset \
  --split val \
  --template chain \
  --num-examples 10 \
  --cfg-scale 2.0 \
  --num-sampling-steps 100 \
  --sampler ddim \
  --out-dir results/vlm_feedback/qwen70b_hosted
```

## InternVL3-78B via vLLM

```bash
vllm serve OpenGVLab/InternVL3-78B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --limit-mm-per-prompt '{"image": 2}' \
  --max-model-len 8192 \
  --trust-remote-code
```

Call it from the feedback script:

```bash
python scripts/vlm_feedback_clevr.py \
  --vlm intern78b \
  --vlm-server-url http://HOSTNAME:8000/v1 \
  --ckpt /gpfs/scrubbed/sriyash/DiT-clevr-text-final/000-DiT-S-4-text/checkpoints/0100000-ema.pt \
  --model DiT-S/4 \
  --vae stabilityai/sdxl-vae \
  --dataset-root /gpfs/scrubbed/sriyash/clevr_dit_dataset \
  --split val \
  --template chain \
  --num-examples 4 \
  --cfg-scale 1.0 \
  --num-sampling-steps 50 \
  --sampler ddim \
  --out-dir results/vlm_feedback/intern78b_hosted
```

Replace `HOSTNAME` with the node name or IP where vLLM is running. If the server uses an API key, set `VLLM_API_KEY` or pass `--vlm-server-api-key`.

# Step2: Generating and Training feedback conditioned models


## Debugging language feedback pipeline

- Using qwen2.5-VL-7b
```bash
python scripts/vlm_feedback_clevr.py \
  --ckpt /gpfs/scrubbed/sriyash/DiT-clevr-text-final/000-DiT-S-4-text/checkpoints/0100000-ema.pt \
  --model DiT-S/4 \
  --vae stabilityai/sdxl-vae \
  --dataset-root /gpfs/scrubbed/sriyash/clevr_dit_dataset \
  --split val \
  --template chain \
  --num-examples 4 \
  --cfg-scale 1.0 \
  --num-sampling-steps 50 \
  --sampler ddim \
  --vlm-model Qwen/Qwen2.5-VL-7B-Instruct \
  --out-dir results/vlm_feedback/scratch_text_s4
```

- using gemini

```bash
python scripts/vlm_feedback_clevr.py \
  --vlm gemini \
  --ckpt /gpfs/scrubbed/sriyash/DiT-clevr-text-final/000-DiT-S-4-text/checkpoints/0100000-ema.pt \
  --model DiT-S/4 \
  --vae stabilityai/sdxl-vae \
  --dataset-root /gpfs/scrubbed/sriyash/clevr_dit_dataset \
  --split val \
  --num-examples 10 \
  --cfg-scale 2.0 \
  --num-sampling-steps 100 \
  --use-caption \
  --out-dir results/vlm_feedback/gemini_metadata
```

