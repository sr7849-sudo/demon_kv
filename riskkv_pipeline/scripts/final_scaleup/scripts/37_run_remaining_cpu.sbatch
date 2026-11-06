#!/usr/bin/env bash
#SBATCH --job-name=riskkv_cpu
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/final_scaleup/logs/riskkv_cpu_%j.out
#SBATCH --error=experiments/final_scaleup/logs/riskkv_cpu_%j.err

set -euo pipefail

cd /scratch/sr7849/KV_Cache/riskkv_pipeline

export PYTHONPATH=$PWD:${PYTHONPATH:-}

bash scripts/final_scaleup/scripts/37_run_remaining_cpu.sh
