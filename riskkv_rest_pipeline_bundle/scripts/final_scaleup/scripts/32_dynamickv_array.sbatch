#!/usr/bin/env bash
#SBATCH --job-name=rkv_dyn
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/final_scaleup/logs/dyn_%A_%a.out
#SBATCH --error=experiments/final_scaleup/logs/dyn_%A_%a.err

set -euo pipefail
cd /scratch/sr7849/KV_Cache/riskkv_pipeline
mkdir -p experiments/final_scaleup/logs
bash scripts/final_scaleup/scripts/31_run_dynamickv_job.sh "$SLURM_ARRAY_TASK_ID"
