#!/usr/bin/env bash
# Locate and inspect the exact official Mistral DynamicKV launcher in the checked-out repo.
set -e
ROOT=/scratch/sr7849/KV_Cache/riskkv_pipeline
REPO=${ROOT}/external/DynamicKV
cd "${REPO}"
echo "commit=$(git rev-parse HEAD)"
find run -type f -name '*.sh' | grep -Ei 'mistral.*dynamic.*v11|dynamic.*v11.*mistral' | sort
