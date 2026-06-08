## Scalable Diffusion Models with Transformers (DiT)<br><sub>Official PyTorch Implementation</sub>

### [Paper](http://arxiv.org/abs/2212.09748) | [Project Page](https://www.wpeebles.com/DiT) | Run DiT-XL/2 [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/wpeebles/DiT) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/facebookresearch/DiT/blob/main/run_DiT.ipynb) <a href="https://replicate.com/arielreplicate/scalable_diffusion_with_transformers"><img src="https://replicate.com/arielreplicate/scalable_diffusion_with_transformers/badge"></a>

![DiT samples](visuals/sample_grid_0.png)

This repo contains PyTorch model definitions, pre-trained weights and training/sampling code for our paper exploring 
diffusion models with transformers (DiTs). You can find more visualizations on our [project page](https://www.wpeebles.com/DiT).

> [**Scalable Diffusion Models with Transformers**](https://www.wpeebles.com/DiT)<br>
> [William Peebles](https://www.wpeebles.com), [Saining Xie](https://www.sainingxie.com)
> <br>UC Berkeley, New York University<br>

We train latent diffusion models, replacing the commonly-used U-Net backbone with a transformer that operates on 
latent patches. We analyze the scalability of our Diffusion Transformers (DiTs) through the lens of forward pass 
complexity as measured by Gflops. We find that DiTs with higher Gflops---through increased transformer depth/width or
increased number of input tokens---consistently have lower FID. In addition to good scalability properties, our 
DiT-XL/2 models outperform all prior diffusion models on the class-conditional ImageNet 512×512 and 256×256 benchmarks, 
achieving a state-of-the-art FID of 2.27 on the latter.

This repository contains:

* 🪐 A simple PyTorch [implementation](models.py) of DiT
* ⚡️ Pre-trained class-conditional DiT models trained on ImageNet (512x512 and 256x256)
* 💥 A self-contained [Hugging Face Space](https://huggingface.co/spaces/wpeebles/DiT) and [Colab notebook](http://colab.research.google.com/github/facebookresearch/DiT/blob/main/run_DiT.ipynb) for running pre-trained DiT-XL/2 models
* 🛸 A DiT [training script](train.py) using PyTorch DDP

An implementation of DiT directly in Hugging Face `diffusers` can also be found [here](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/dit.mdx).


## Setup

First, download and set up the repo:

```bash
git clone https://github.com/facebookresearch/DiT.git
cd DiT
```

We provide an [`environment.yml`](environment.yml) file that can be used to create a Conda environment. If you only want 
to run pre-trained models locally on CPU, you can remove the `cudatoolkit` and `pytorch-cuda` requirements from the file.

```bash
conda env create -f environment.yml
conda activate DiT
```


## Sampling [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/wpeebles/DiT) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/facebookresearch/DiT/blob/main/run_DiT.ipynb)
![More DiT samples](visuals/sample_grid_1.png)

**Pre-trained DiT checkpoints.** You can sample from our pre-trained DiT models with [`sample.py`](sample.py). Weights for our pre-trained DiT model will be 
automatically downloaded depending on the model you use. The script has various arguments to switch between the 256x256
and 512x512 models, adjust sampling steps, change the classifier-free guidance scale, etc. For example, to sample from
our 512x512 DiT-XL/2 model, you can use:

```bash
python sample.py --image-size 512 --seed 1
```

For convenience, our pre-trained DiT models can be downloaded directly here as well:

| DiT Model     | Image Resolution | FID-50K | Inception Score | Gflops | 
|---------------|------------------|---------|-----------------|--------|
| [XL/2](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt) | 256x256          | 2.27    | 278.24          | 119    |
| [XL/2](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-512x512.pt) | 512x512          | 3.04    | 240.82          | 525    |


**Custom DiT checkpoints.** If you've trained a new DiT model with [`train.py`](train.py) (see [below](#training-dit)), you can add the `--ckpt`
argument to use your own checkpoint instead. For example, to sample from the EMA weights of a custom 
256x256 DiT-L/4 model, run:

```bash
python sample.py --model DiT-L/4 --image-size 256 --ckpt /path/to/model.pt
```


## Training DiT

We provide a training script for DiT in [`train.py`](train.py). This script can be used to train class-conditional 
DiT models, but it can be easily modified to support other types of conditioning. To launch DiT-XL/2 (256x256) training with `N` GPUs on 
one node:

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py --model DiT-XL/2 --data-path /path/to/imagenet/train
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
torchrun --nnodes=1 --nproc_per_node=N train_text.py \
  --data-path /gpfs/scrubbed/sriyash/clevr_dit_dataset \
  --model DiT-XL/2 \
  --image-size 256 \
  --vae mse \
  --global-batch-size 256
```

```bash
python sample_text.py \
  --ckpt /tmp/dit_0030000_ema_sample.pt \
  --caption "objects: small gray metal sphere, large yellow metal sphere, large blue rubber cube. horizontal: yellow sphere is right of blue cube, gray sphere is right of yellow sphere. depth: gray sphere is behind yellow sphere, blue cube is behind gray sphere."
```

### PyTorch Training Results

We've trained DiT-XL/2 and DiT-B/4 models from scratch with the PyTorch training script
to verify that it reproduces the original JAX results up to several hundred thousand training iterations. Across our experiments, the PyTorch-trained models give 
similar (and sometimes slightly better) results compared to the JAX-trained models up to reasonable random variation. Some data points:

| DiT Model  | Train Steps | FID-50K<br> (JAX Training) | FID-50K<br> (PyTorch Training) | PyTorch Global Training Seed |
|------------|-------------|----------------------------|--------------------------------|------------------------------|
| XL/2       | 400K        | 19.5                       | **18.1**                       | 42                           |
| B/4        | 400K        | **68.4**                   | 68.9                           | 42                           |
| B/4        | 400K        | 68.4                       | **68.3**                       | 100                          |

These models were trained at 256x256 resolution; we used 8x A100s to train XL/2 and 4x A100s to train B/4. Note that FID 
here is computed with 250 DDPM sampling steps, with the `mse` VAE decoder and without guidance (`cfg-scale=1`). 

**TF32 Note (important for A100 users).** When we ran the above tests, TF32 matmuls were disabled per PyTorch's defaults. 
We've enabled them at the top of `train.py` and `sample.py` because it makes training and sampling way way way faster on 
A100s (and should for other Ampere GPUs too), but note that the use of TF32 may lead to some differences compared to 
the above results.

### Enhancements
Training (and sampling) could likely be sped-up significantly by:
- [ ] using [Flash Attention](https://github.com/HazyResearch/flash-attention) in the DiT model
- [ ] using `torch.compile` in PyTorch 2.0

Basic features that would be nice to add:
- [ ] Monitor FID and other metrics
- [ ] Generate and save samples from the EMA model periodically
- [ ] Resume training from a checkpoint
- [ ] AMP/bfloat16 support

**🔥 Feature Update** Check out this repository at https://github.com/chuanyangjin/fast-DiT to preview a selection of training speed acceleration and memory saving features including gradient checkpointing, mixed precision training and pre-extrated VAE features. With these advancements, we have achieved a training speed of 0.84 steps/sec for DiT-XL/2 using just a single A100 GPU.

## Evaluation (FID, Inception Score, etc.)

We include a [`sample_ddp.py`](sample_ddp.py) script which samples a large number of images from a DiT model in parallel. This script 
generates a folder of samples as well as a `.npz` file which can be directly used with [ADM's TensorFlow
evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to compute FID, Inception Score and
other metrics. For example, to sample 50K images from our pre-trained DiT-XL/2 model over `N` GPUs, run:

```bash
torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py --model DiT-XL/2 --num-fid-samples 50000
```

There are several additional options; see [`sample_ddp.py`](sample_ddp.py) for details. 


## Differences from JAX

Our models were originally trained in JAX on TPUs. The weights in this repo are ported directly from the JAX models. 
There may be minor differences in results stemming from sampling with different floating point precisions. We re-evaluated 
our ported PyTorch weights at FP32, and they actually perform marginally better than sampling in JAX (2.21 FID 
versus 2.27 in the paper).


## BibTeX

```bibtex
@article{Peebles2022DiT,
  title={Scalable Diffusion Models with Transformers},
  author={William Peebles and Saining Xie},
  year={2022},
  journal={arXiv preprint arXiv:2212.09748},
}
```


## Acknowledgments
We thank Kaiming He, Ronghang Hu, Alexander Berg, Shoubhik Debnath, Tim Brooks, Ilija Radosavovic and Tete Xiao for helpful discussions. 
William Peebles is supported by the NSF Graduate Research Fellowship.

This codebase borrows from OpenAI's diffusion repos, most notably [ADM](https://github.com/openai/guided-diffusion).


## License
The code and model weights are licensed under CC-BY-NC. See [`LICENSE.txt`](LICENSE.txt) for details.


results/slurm-dit_clevr_text-134848.out - pretrained finetune / low lr
results/slurm-scratch-unconditional-dit_clevr_uncond-134867.out
134888 -- direct conditioning/text/scratch small model
https://wandb.ai/sriyash-uw-team/DiT-clevr-text-final/runs/ohbhe9n6

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
