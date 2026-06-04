#!/bin/bash
#SBATCH --job-name=dit_clevr_text
#SBATCH --qos=normal
#SBATCH --account=socialrl
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=6:00:00
#SBATCH --output=/gpfs/projects/weirdlab/sriyash/DiT/results/slurm-%x-%j.out

cd /gpfs/projects/weirdlab/sriyash/DiT

# module load conda
# conda activate DiT
source .venv/bin/activate
torchrun --nnodes=1 --nproc_per_node=2 train_text.py \
  --data-path /gpfs/scrubbed/sriyash/clevr_dit_dataset \
  --model DiT-XL/2 \
  --image-size 256 \
  --vae mse \
  --global-batch-size 128 \
  --num-workers 8 \
  --results-dir /gpfs/scrubbed/sriyash/DiT-clevr-results \
  --lr 1e-5 \
  --grad-clip 1.0 \
  --ckpt-every 1000
