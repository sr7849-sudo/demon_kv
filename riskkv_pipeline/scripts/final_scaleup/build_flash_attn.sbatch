#!/bin/bash
#SBATCH --job-name=build_fa
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=experiments/final_scaleup/flash_attn_build_%j.out
#SBATCH --error=experiments/final_scaleup/flash_attn_build_%j.err

# Do NOT use pipefail here. Some harmless version-printing commands use pipes.
set -e

ROOT=/scratch/sr7849/KV_Cache/riskkv_pipeline
ENV=/scratch/sr7849/conda_envs/riskkv-adakv

# ----------------------------------------------------------------------
# Use the AdaKV environment directly.
# ----------------------------------------------------------------------
export CONDA_PREFIX="${ENV}"
export PATH="${ENV}/bin:${PATH}"

# CUDA 12.8 toolkit installed in this environment.
export CUDA_HOME="${ENV}"

export LD_LIBRARY_PATH="${ENV}/targets/x86_64-linux/lib:${ENV}/lib:${ENV}/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${ENV}/targets/x86_64-linux/lib:${ENV}/lib:${LIBRARY_PATH:-}"
export CPATH="${ENV}/targets/x86_64-linux/include:${ENV}/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="${ENV}/targets/x86_64-linux/include:${ENV}/include:${CPLUS_INCLUDE_PATH:-}"

# Host compiler for CUDA compilation.
export CC="${ENV}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${ENV}/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="${CXX}"

# ----------------------------------------------------------------------
# Force LOCAL source compilation.
#
# This is important because the downloaded upstream binary wheel required
# GLIBC >= 2.32, while Jubail has GLIBC 2.28.
# ----------------------------------------------------------------------
export FLASH_ATTENTION_FORCE_BUILD=TRUE

# A100 GPU architecture = sm_80.
export FLASH_ATTN_CUDA_ARCHS="80"
export TORCH_CUDA_ARCH_LIST="8.0"

# Limit compilation parallelism.
export MAX_JOBS=8
export NVCC_THREADS=2

# Keep all temporary/cache files on scratch.
export TMPDIR=/scratch/sr7849/tmp
export TMP=/scratch/sr7849/tmp
export TEMP=/scratch/sr7849/tmp
export PIP_CACHE_DIR=/scratch/sr7849/pip_cache
export HF_HOME=/scratch/sr7849/hf_cache

mkdir -p "${TMPDIR}"
mkdir -p "${PIP_CACHE_DIR}"

echo "============================================================"
echo "FlashAttention source build"
echo "============================================================"

echo "HOST:"
hostname

echo
echo "PYTHON:"
which python
python --version

echo
echo "NVCC:"
which nvcc
nvcc --version

echo
echo "GCC:"
echo "${CC}"
"${CC}" --version

echo
echo "GLIBC:"
ldd --version

echo
echo "TORCH:"
python - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME

print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA_HOME:", CUDA_HOME)
PY

echo
echo "CUDA HEADERS:"
test -f "${ENV}/targets/x86_64-linux/include/cusparse.h"
echo "cusparse.h: FOUND"

test -f "${ENV}/targets/x86_64-linux/include/cuda_runtime.h"
echo "cuda_runtime.h: FOUND"


echo
echo "============================================================"
echo "Removing previous FlashAttention installation"
echo "============================================================"

python -m pip uninstall -y flash-attn || true


echo
echo "============================================================"
echo "Building FlashAttention 2.8.3 FROM SOURCE"
echo "============================================================"

# --no-cache-dir avoids accidentally reusing the incompatible prebuilt wheel.
# FLASH_ATTENTION_FORCE_BUILD=TRUE tells setup.py not to download that wheel.
python -m pip install \
    --no-cache-dir \
    --no-build-isolation \
    --no-binary=flash-attn \
    flash-attn==2.8.3


echo
echo "============================================================"
echo "FlashAttention import test"
echo "============================================================"

python - <<'PY'
import torch
import flash_attn

print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("FlashAttention:", flash_attn.__version__)

print("FLASH_ATTN IMPORT: PASS")
PY


echo
echo "============================================================"
echo "FLASH ATTENTION SOURCE BUILD: PASS"
echo "============================================================"
