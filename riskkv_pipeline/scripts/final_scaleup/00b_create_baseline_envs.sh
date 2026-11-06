#!/usr/bin/env bash

# =============================================================================
# 00b_create_baseline_envs.sh
# =============================================================================
#
# PURPOSE
# -------
# Creates separate Conda environments for the external KV-cache baselines.
#
# We DO NOT install them into the main `riskkv` environment because these
# research repositories depend on different Transformers/CUDA versions.
#
# Environments created:
#
#   riskkv-kvcf
#       KVCache-Factory:
#       FullKV, SnapKV, PyramidKV, AdaKV, L2Norm.
#
#   riskkv-adakv
#       Official AdaKV implementation, retained for reproduction checks.
#
#   riskkv-dynamickv
#       Official DynamicKV implementation.
#
# Existing environments are skipped rather than deleted.
# =============================================================================

set -euo pipefail

ROOT="${RISKKV_ROOT:-/scratch/sr7849/KV_Cache/riskkv_pipeline}"
EXT="${ROOT}/external"

# Make Conda activation available inside a non-interactive script.
source "$(conda info --base)/etc/profile.d/conda.sh"

env_exists () {
    conda env list | awk '{print $1}' | grep -qx "$1"
}


# =============================================================================
# 1. KVCache-Factory environment
# =============================================================================

if env_exists riskkv-kvcf; then

    echo "[SKIP] riskkv-kvcf already exists"

else

    echo
    echo "============================================================"
    echo "Creating riskkv-kvcf"
    echo "============================================================"

    conda create -y \
        -n riskkv-kvcf \
        python=3.11

    conda activate riskkv-kvcf

    python -m pip install --upgrade pip

    python -m pip install \
        -r "${EXT}/KVCache-Factory/requirements.txt"

    conda deactivate
fi


# =============================================================================
# 2. Official AdaKV environment
# =============================================================================

if env_exists riskkv-adakv; then

    echo "[SKIP] riskkv-adakv already exists"

else

    echo
    echo "============================================================"
    echo "Creating riskkv-adakv"
    echo "============================================================"

    conda create -y \
        -n riskkv-adakv \
        python=3.11

    conda activate riskkv-adakv

    python -m pip install --upgrade pip

    # AdaKV's code is tied to the older Transformers API.
    python -m pip install \
        "transformers==4.44.2" \
        datasets \
        accelerate \
        tiktoken \
        jieba \
        rouge_score \
        sentencepiece

    cd "${EXT}/AdaKV"

    # Some AdaKV versions provide a Makefile for installing CUDA extensions.
    # Do not fail the entire setup if that extension fails here; we inspect
    # compatibility in the following preflight stage.
    if [[ -f Makefile ]]; then
        echo "AdaKV Makefile found. Attempting installation..."
        make i || echo "[WARN] AdaKV 'make i' failed; will inspect during preflight."
    else
        echo "[INFO] No AdaKV Makefile found; using PYTHONPATH installation."
    fi

    conda deactivate

    cd "${ROOT}"
fi


# =============================================================================
# 3. DynamicKV environment
# =============================================================================

if env_exists riskkv-dynamickv; then

    echo "[SKIP] riskkv-dynamickv already exists"

else

    echo
    echo "============================================================"
    echo "Creating riskkv-dynamickv"
    echo "============================================================"

    conda create -y \
        -n riskkv-dynamickv \
        python=3.11

    conda activate riskkv-dynamickv

    python -m pip install --upgrade pip

    cd "${EXT}/DynamicKV"

    if [[ -f requirements.txt ]]; then

        python -m pip install \
            -r requirements.txt

    else

        python -m pip install \
            "transformers>=4.44.1" \
            datasets \
            accelerate \
            tqdm \
            rouge_score \
            sentencepiece
    fi

    conda deactivate

    cd "${ROOT}"
fi


echo
echo "============================================================"
echo "BASELINE ENVIRONMENT CREATION FINISHED"
echo "============================================================"

conda env list | grep -E \
    'riskkv-kvcf|riskkv-adakv|riskkv-dynamickv' || true
