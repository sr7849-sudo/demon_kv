#!/usr/bin/env bash

# RiskKV HPC environment
# Source this file inside a GPU allocation.

module load gcc/12.2.0
module load cuda/13.0.0
module load binutils/2.46.0

export CC=/share/apps/NYUAD5/gcc/12.2.0/bin/gcc
export CXX=/share/apps/NYUAD5/gcc/12.2.0/bin/g++
export CUDAHOSTCXX="$CXX"

export CUDA_HOME=/share/apps/NYUAD5/cuda/13.0.0
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"

export MAX_JOBS=2
export TORCH_CUDA_ARCH_LIST=8.0

# Prevent CPU oversubscription
export OMP_NUM_THREADS=4

# Verify toolchain
if [[ ! -x "$CXX" ]]; then
    echo "ERROR: GCC 12 C++ compiler unavailable"
    return 1 2>/dev/null || exit 1
fi

echo "RiskKV environment loaded"
echo "CXX=$CXX"
echo "CUDA_HOME=$CUDA_HOME"
