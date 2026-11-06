#!/bin/bash
#SBATCH --job-name=riskkv_quant4
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/e1_runtime/quant4_slurm_%j.log

set -euo pipefail

cd /scratch/sr7849/KV_Cache/riskkv_pipeline

# Initialize Conda
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate riskkv

# Initialize compilers and CUDA
source hpc/env.sh

echo "=== GPU ==="
nvidia-smi

echo "=== CUDA ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); assert torch.cuda.is_available()"

echo "=== RUNNING QUANT4 ==="

python -u scripts/06e_profile_hardware.py \
    --config results/e1_runtime/quant4_only.yaml \
    --input data/system_profile1.jsonl \
    --output results/e1_runtime/quant4_gcc12.csv \
    --warmups 1 \
    --repetitions 1

echo "=== CHECKING FOR HIDDEN FAILURES ==="

if grep -E \
    "WARMUP FAILURE|Traceback|Error building extension|FAILED:|Retrying pair" \
    "results/e1_runtime/quant4_slurm_${SLURM_JOB_ID}.log"; then
    echo "ERROR: Quant4 encountered a failure or fallback"
    exit 1
fi

echo "QUANT4 SMOKE COMPLETED"
