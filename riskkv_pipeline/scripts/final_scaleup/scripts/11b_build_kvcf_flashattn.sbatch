#!/bin/bash
#SBATCH --job-name=kvcf_flash
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=experiments/final_scaleup/smoke/kvcf_flash_%j.out
#SBATCH --error=experiments/final_scaleup/smoke/kvcf_flash_%j.err

set -e

ROOT=/scratch/sr7849/KV_Cache/riskkv_pipeline
KVCF=/scratch/sr7849/conda_envs/riskkv-kvcf
CUDAENV=/scratch/sr7849/conda_envs/riskkv-adakv

export PATH="${KVCF}/bin:${CUDAENV}/bin:${PATH}"

export CUDA_HOME="${CUDAENV}"

export LD_LIBRARY_PATH="${CUDAENV}/targets/x86_64-linux/lib:${CUDAENV}/lib:${CUDAENV}/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${CUDAENV}/targets/x86_64-linux/lib:${CUDAENV}/lib:${LIBRARY_PATH:-}"

export CPATH="${CUDAENV}/targets/x86_64-linux/include:${CUDAENV}/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="${CUDAENV}/targets/x86_64-linux/include:${CUDAENV}/include:${CPLUS_INCLUDE_PATH:-}"

export CC="${CUDAENV}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CUDAENV}/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="${CXX}"

export TORCH_CUDA_ARCH_LIST="8.0"
export FLASH_ATTN_CUDA_ARCHS="80"

export MAX_JOBS=8
export NVCC_THREADS=2

export TMPDIR=/scratch/sr7849/tmp/kvcf_flash_${SLURM_JOB_ID}
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"

mkdir -p "${TMPDIR}"

echo "============================================================"
echo "BUILD ENVIRONMENT"
echo "============================================================"

hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo
echo "Python:"
which python
python --version

echo
echo "NVCC:"
which nvcc
nvcc --version

echo
echo "Compiler:"
"${CC}" --version | head -1 || true

echo
echo "Host GLIBC:"
ldd --version | head -1 || true

echo
echo "Torch:"
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print("CUDA:", torch.cuda.is_available())
PY

echo
echo "============================================================"
echo "REMOVE ANY INCOMPATIBLE FLASH-ATTN"
echo "============================================================"

python -m pip uninstall -y flash-attn || true

rm -rf \
    "${KVCF}/lib/python3.11/site-packages/flash_attn" \
    "${KVCF}/lib/python3.11/site-packages/flash_attn-"*.dist-info \
    "${KVCF}/lib/python3.11/site-packages/flash_attn_2_cuda"*.so

echo
echo "============================================================"
echo "GET SOURCE DISTRIBUTION"
echo "============================================================"

SRC="${TMPDIR}/source"
mkdir -p "${SRC}"
cd "${SRC}"

python -m pip download \
    --no-build-isolation \
    --no-binary=:all: \
    --no-deps \
    "flash-attn==2.8.3"

ls -lh

TARBALL=$(find . -maxdepth 1 -name 'flash_attn-2.8.3.tar.gz' -print -quit)

test -n "${TARBALL}"

tar -xzf "${TARBALL}"

cd flash_attn-2.8.3

echo
echo "Source directory:"
pwd

echo
echo "============================================================"
echo "BUILD LOCAL SOURCE"
echo "============================================================"

# Critical:
# We are installing "." -- the local extracted source.
# pip therefore cannot substitute the PyPI package specification itself.
#
# Force build disables setup.py's prebuilt-wheel shortcut.
export FLASH_ATTENTION_FORCE_BUILD=TRUE

python -m pip install \
    --no-build-isolation \
    --no-cache-dir \
    .

echo
echo "============================================================"
echo "VERIFY IMPORT + GPU KERNEL"
echo "============================================================"

python - <<'PY'
import torch
import flash_attn
from flash_attn import flash_attn_func

print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("FlashAttention:", flash_attn.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

q = torch.randn(
    1, 128, 8, 64,
    device="cuda",
    dtype=torch.bfloat16,
)
k = torch.randn_like(q)
v = torch.randn_like(q)

y = flash_attn_func(q, k, v)

print("Output:", y.shape)
print("Finite:", torch.isfinite(y).all().item())

assert y.shape == q.shape
assert torch.isfinite(y).all()

print("KVCF FLASHATTENTION SOURCE BUILD: PASS")
PY
