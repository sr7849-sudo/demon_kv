#!/bin/bash

set -euo pipefail

ROOT=/scratch/sr7849/KV_Cache/riskkv_pipeline

cd "$ROOT"

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate riskkv

ORACLE=$ROOT/experiments/full_oracle_qwen3_8b/oracle_longbench_qwen3_8b.csv
ANALYSIS=$ROOT/experiments/full_oracle_qwen3_8b/analysis

mkdir -p "$ANALYSIS"

echo "======================================"
echo "1/4 Oracle landscape"
echo "======================================"

python scripts/02a_oracle_landscape.py \
  --input "$ORACLE" \
  --outdir "$ANALYSIS/01_oracle_landscape"

echo
echo "======================================"
echo "2/4 Request heterogeneity"
echo "======================================"

python scripts/02b_request_heterogeneity.py \
  --input "$ORACLE" \
  --outdir "$ANALYSIS/02_request_heterogeneity"

echo
echo "======================================"
echo "3/4 Signal usefulness"
echo "======================================"

python scripts/02c_signal_usefulness.py \
  --input "$ORACLE" \
  --outdir "$ANALYSIS/03_signal_usefulness"

echo
echo "======================================"
echo "4/4 Oracle opportunity"
echo "======================================"

python scripts/02d_oracle_opportunity.py \
  --input "$ORACLE" \
  --outdir "$ANALYSIS/04_oracle_opportunity" \
  --deltas 0.01 0.05 0.10

echo
echo "======================================"
echo "ALL ORACLE ANALYSES COMPLETE"
echo "======================================"
