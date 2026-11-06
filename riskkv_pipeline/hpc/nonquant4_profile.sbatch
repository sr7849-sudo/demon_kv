#!/bin/bash
#SBATCH --job-name=riskkv_nonquant
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=results/e1_runtime/nonquant4_%j.log

set -euo pipefail

cd /scratch/sr7849/KV_Cache/riskkv_pipeline

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate riskkv
source hpc/env.sh

python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"

python -u scripts/06e_profile_hardware.py \
  --config results/e1_runtime/nonquant4.yaml \
  --input data/system_profile25.jsonl \
  --output "results/e1_runtime/nonquant4_${SLURM_JOB_ID}.csv" \
  --warmups 1 \
  --repetitions 3
