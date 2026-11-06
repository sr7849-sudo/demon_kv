#!/bin/bash
#SBATCH --job-name=rkv_kvcf_final
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=experiments/final_scaleup/logs/kvcf_%A_%a.out
#SBATCH --error=experiments/final_scaleup/logs/kvcf_%A_%a.err
set -e
ROOT=/scratch/sr7849/KV_Cache/riskkv_pipeline
cd "${ROOT}"
mkdir -p experiments/final_scaleup/logs
bash scripts/final_scaleup/scripts/14_run_kvcf_final_job.sh "${SLURM_ARRAY_TASK_ID}"
