#!/usr/bin/env bash

# =============================================================================
# 00c_preflight_baselines.sh
# =============================================================================
#
# PURPOSE
# -------
# Structural preflight for the three external KV-cache baselines.
#
# IMPORTANT
# ---------
# We intentionally DO NOT use `conda activate` here.
#
# Jubail's Conda activation/deactivation hooks reference optional shell
# variables, and `set -u` can make harmless hook behavior abort the script.
#
# Instead, we invoke the exact Python binary belonging to each environment.
# This is deterministic and avoids shell-hook problems entirely.
# =============================================================================

set -e

ROOT="/scratch/sr7849/KV_Cache/riskkv_pipeline"
EXT="${ROOT}/external"

KVCF_ENV="/scratch/sr7849/conda_envs/riskkv-kvcf"
ADAKV_ENV="/scratch/sr7849/conda_envs/riskkv-adakv"
DYNKV_ENV="/scratch/sr7849/conda_envs/riskkv-dynamickv"


# =============================================================================
# 1. KVCache-Factory
# =============================================================================

echo
echo "============================================================"
echo "1. KVCache-Factory"
echo "============================================================"

test -d "${EXT}/KVCache-Factory"
test -x "${KVCF_ENV}/bin/python"

"${KVCF_ENV}/bin/python" - <<'PY'
import sys
import torch
import transformers

print("Python:", sys.executable)
print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())

print("KVCache-Factory Python imports: PASS")
PY

if [[ -f "${EXT}/KVCache-Factory/run_longbench.py" ]]; then
    echo "run_longbench.py: FOUND"
else
    echo "ERROR: KVCache-Factory/run_longbench.py not found"
    exit 1
fi


# =============================================================================
# 2. AdaKV
# =============================================================================

echo
echo "============================================================"
echo "2. AdaKV"
echo "============================================================"

test -d "${EXT}/AdaKV"
test -x "${ADAKV_ENV}/bin/python"

PYTHONPATH="${EXT}/AdaKV:${PYTHONPATH:-}" \
"${ADAKV_ENV}/bin/python" - <<'PY'
import sys
import torch
import transformers
import flash_attn

print("Python:", sys.executable)
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("Transformers:", transformers.__version__)
print("FlashAttention:", flash_attn.__version__)
print("CUDA available:", torch.cuda.is_available())

from adaptive_snapkv.monkeypatch.monkeypatch import (
    replace_mistral_adaptive,
    replace_llama_adaptive,
)

print("AdaKV monkeypatch import: PASS")

import tiny_api_cuda
print("tiny_api_cuda import: PASS")
print("tiny_api_cuda path:", tiny_api_cuda.__file__)

print("AdaKV Python/CUDA-extension preflight: PASS")
PY


# =============================================================================
# 3. DynamicKV
# =============================================================================

echo
echo "============================================================"
echo "3. DynamicKV"
echo "============================================================"

test -d "${EXT}/DynamicKV"
test -x "${DYNKV_ENV}/bin/python"

PYTHONPATH="${EXT}/DynamicKV:${PYTHONPATH:-}" \
"${DYNKV_ENV}/bin/python" - <<'PY'
import sys
import torch
import transformers
import datasets
import accelerate

print("Python:", sys.executable)
print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Datasets:", datasets.__version__)
print("Accelerate:", accelerate.__version__)
print("CUDA available:", torch.cuda.is_available())

print("DynamicKV Python imports: PASS")
PY

echo
echo "DynamicKV LongBench-related Python files:"

find "${EXT}/DynamicKV" \
    -type f \
    -name "*.py" \
    | grep -Ei 'longbench|pred' \
    | head -40 || true


echo
echo "============================================================"
echo "BASELINE PREFLIGHT: PASS"
echo "============================================================"
