#!/bin/bash
#SBATCH --job-name=dit_clevr_text
#SBATCH --qos=normal
#SBATCH --account=socialrl
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/projects/weirdlab/sriyash/DiT/slurm-%x-%j.out

cd /gpfs/projects/weirdlab/sriyash/DiT

/gpfs/scrubbed/sriyash/conda/DiT/bin/torchrun --nnodes=1 --nproc_per_node=2 train_text.py \
  --data-path /gpfs/scrubbed/sriyash/clevr_dit_dataset \
  --model DiT-XL/2 \
  --image-size 256 \
  --vae mse \
  --global-batch-size 8 \
  --num-workers 8
