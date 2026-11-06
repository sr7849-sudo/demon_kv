#!/bin/bash
#SBATCH --job-name=riskkv_oracle
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/sr7849/KV_Cache/riskkv_pipeline/experiments/full_oracle_qwen3_8b/slurm-%j.out
#SBATCH --error=/scratch/sr7849/KV_Cache/riskkv_pipeline/experiments/full_oracle_qwen3_8b/slurm-%j.err

set -euo pipefail

ROOT=/scratch/sr7849/KV_Cache/riskkv_pipeline
EXP=$ROOT/experiments/full_oracle_qwen3_8b

cd "$ROOT"

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate riskkv

module load cuda/13.0.0

export CUDA_HOME=/share/apps/NYUAD5/cuda/13.0.0
export PATH="$CUDA_HOME/bin:$PATH"

export GCC_INTERNAL_DIR="$CONDA_PREFIX/libexec/gcc/x86_64-conda-linux-gnu/13.4.0"
export SYSROOT="$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot"

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export COMPILER_PATH="$GCC_INTERNAL_DIR"

export GCC_LIB_DIR="$(dirname "$(find "$CONDA_PREFIX" -name crtbeginS.o -print -quit)")"

export LIBRARY_PATH="$GCC_LIB_DIR:$SYSROOT/usr/lib64:$SYSROOT/usr/lib:$CONDA_PREFIX/lib"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CUDA_HOME/lib64:$SYSROOT/usr/lib64"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "========================================"
echo "RiskKV FULL ORACLE"
echo "Job: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Started: $(date -Is)"
echo "========================================"

python - <<'PY'
import torch

print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))

assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (8, 0)

print("GPU PREFLIGHT: PASS")
PY

echo
echo "Verifying frozen experiment..."

cd "$EXP"
sha256sum -c SHA256SUMS
cd "$ROOT"

echo
echo "Starting oracle collection..."

python -u scripts/01_collect_oracle.py \
  --config "$EXP/config.yaml" \
  --input "$EXP/requests.jsonl" \
  --output "$EXP/oracle_longbench_qwen3_8b.csv"

echo
echo "========================================"
echo "Oracle finished: $(date -Is)"
echo "========================================"

python - <<'PY'
import pandas as pd
from pathlib import Path

p = Path(
    "/scratch/sr7849/KV_Cache/riskkv_pipeline/"
    "experiments/full_oracle_qwen3_8b/"
    "oracle_longbench_qwen3_8b.csv"
)

df = pd.read_csv(p)

print("Rows:", len(df))
print("Requests:", df.request_id.nunique())
print("Policies:", df.policy.nunique())

print("\nRows per policy:")
print(df.groupby("policy").size())

print("\nRows per task:")
print(df.groupby("task").size())

required = [
    "quality",
    "full_quality",
    "signed_delta",
    "degradation",
    "actual_seconds",
    "actual_peak_gpu_bytes",
    "peak_incremental_gpu_bytes",
    "kv_bytes_full_theoretical",
    "kv_bytes_policy_estimated",
    "last_logit_entropy",
    "prefill_nll",
]

print("\nMissing values:")
print(df[required].isna().sum())

assert len(df) == 2320
assert df.request_id.nunique() == 290
assert df.policy.nunique() == 8
assert df[required].isna().sum().sum() == 0

print("\nFULL ORACLE VALIDATION: PASS")
PY
