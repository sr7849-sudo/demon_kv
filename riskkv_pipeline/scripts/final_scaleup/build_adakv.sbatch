#!/bin/bash
#SBATCH --job-name=build_adakv
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/final_scaleup/adakv_build_%j.out
#SBATCH --error=experiments/final_scaleup/adakv_build_%j.err

set -eo pipefail

ROOT=/scratch/sr7849/KV_Cache/riskkv_pipeline
ENV=/scratch/sr7849/conda_envs/riskkv-adakv

# ----------------------------------------------------------------------
# Use the scratch environment directly.
# ----------------------------------------------------------------------
export CONDA_PREFIX="${ENV}"
export PATH="${ENV}/bin:${PATH}"

# CUDA toolkit / nvcc location.
export CUDA_HOME="${ENV}"
export LD_LIBRARY_PATH="${ENV}/targets/x86_64-linux/lib:${ENV}/lib:${ENV}/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="${ENV}/targets/x86_64-linux/include:${ENV}/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="${ENV}/targets/x86_64-linux/include:${ENV}/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="${ENV}/targets/x86_64-linux/lib:${ENV}/lib:${LIBRARY_PATH:-}"
export CPATH="${ENV}/targets/x86_64-linux/include:${ENV}/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="${ENV}/targets/x86_64-linux/include:${ENV}/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="${ENV}/targets/x86_64-linux/lib:${ENV}/lib:${LIBRARY_PATH:-}"

# ----------------------------------------------------------------------
# Explicit Conda C/C++ compilers.
#
# CUDA extensions require BOTH:
#   nvcc -> compiles CUDA source
#   gcc/g++ -> compiles host C/C++ source
# ----------------------------------------------------------------------
export CC="${ENV}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${ENV}/bin/x86_64-conda-linux-gnu-g++"

# Tell nvcc which host C++ compiler to use.
export CUDAHOSTCXX="${CXX}"

# A100 = compute capability 8.0.
export TORCH_CUDA_ARCH_LIST="8.0"

# Keep caches off /home.
export PIP_CACHE_DIR=/scratch/sr7849/pip_cache
export HF_HOME=/scratch/sr7849/hf_cache

cd "${ROOT}/external/AdaKV"


echo "============================================================"
echo "AdaKV build environment"
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
echo "CC:"
echo "${CC}"
"${CC}" --version | head -1

echo
echo "CXX:"
echo "${CXX}"
"${CXX}" --version | head -1

echo
echo "TORCH:"
python - <<'PY'
import os
import torch
from torch.utils.cpp_extension import CUDA_HOME

print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("Torch path:", torch.__file__)
print("CUDA_HOME:", CUDA_HOME)
print("CC:", os.environ.get("CC"))
print("CXX:", os.environ.get("CXX"))
print("TORCH_CUDA_ARCH_LIST:", os.environ.get("TORCH_CUDA_ARCH_LIST"))
PY


echo
echo "============================================================"
echo "Building AdaKV"
echo "============================================================"

# AdaKV's top-level makefile performs:
#
#   cd csrc && make
#   pip install -e .
#
make i


echo
echo "============================================================"
echo "Testing AdaKV imports"
echo "============================================================"

PYTHONPATH="${ROOT}/external/AdaKV:${PYTHONPATH:-}" \
python - <<'PY'
import torch
import transformers

print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("Transformers:", transformers.__version__)

from adaptive_snapkv.monkeypatch.monkeypatch import (
    replace_mistral_adaptive,
    replace_llama_adaptive,
)

print("AdaKV monkeypatch imports: PASS")
PY


echo
echo "============================================================"
echo "ADAKV BUILD: PASS"
echo "============================================================"
