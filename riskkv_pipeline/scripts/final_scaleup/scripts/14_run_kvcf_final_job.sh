#!/usr/bin/env bash
set -e
JOB_ID="${1:?Usage: $0 JOB_ID}"
ROOT=/scratch/sr7849/KV_Cache/riskkv_pipeline
ENV=/scratch/sr7849/conda_envs/riskkv-kvcf
REPO=${ROOT}/external/KVCache-Factory
TABLE=${ROOT}/experiments/final_scaleup/manifests/kvcf_final_jobs.csv
DATA_ROOT=${ROOT}/data/final_common_longbench_filtered
OUT_ROOT=${ROOT}/experiments/final_scaleup/results/kvcf
CUDAENV=/scratch/sr7849/conda_envs/riskkv-adakv

export PATH="${ENV}/bin:${CUDAENV}/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"
export CUDA_HOME="${CUDAENV}"

# CUDA runtime/toolkit needed by FlashAttention and AdaKV.
# PyTorch's own shared-library directory is needed by tiny_api_cuda.
export LD_LIBRARY_PATH="${ENV}/lib/python3.11/site-packages/torch/lib:${CUDAENV}/targets/x86_64-linux/lib:${CUDAENV}/lib:${CUDAENV}/lib64:${ENV}/lib:${LD_LIBRARY_PATH:-}"

export LIBRARY_PATH="${CUDAENV}/targets/x86_64-linux/lib:${CUDAENV}/lib:${LIBRARY_PATH:-}"
export CPATH="${CUDAENV}/targets/x86_64-linux/include:${CUDAENV}/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="${CUDAENV}/targets/x86_64-linux/include:${CUDAENV}/include:${CPLUS_INCLUDE_PATH:-}"

export HF_HOME=/scratch/sr7849/hf_cache
export TRANSFORMERS_CACHE=/scratch/sr7849/hf_cache
export TMPDIR=/scratch/sr7849/tmp
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "=========="
readarray -t F < <("${ENV}/bin/python" - "${TABLE}" "${JOB_ID}" <<'PY'
import pandas as pd,sys
d=pd.read_csv(sys.argv[1]); r=d[d.job_id==int(sys.argv[2])]
if len(r)!=1: raise SystemExit("job id missing/nonunique")
r=r.iloc[0]
print(r.model); print(r.task); print(r.method); print(int(r.budget_tokens))
PY
)
MODEL="${F[0]}"; TASK="${F[1]}"; METHOD="${F[2]}"; BUDGET="${F[3]}"
DATA="${DATA_ROOT}/${TASK}.jsonl"
OUT="${OUT_ROOT}/${METHOD}/budget_${BUDGET}/${TASK}"
mkdir -p "${OUT}"
if [[ -f "${OUT}/DONE" ]]; then echo "already complete"; exit 0; fi
test -f "${DATA}"
echo "DATA=${DATA}"
echo "DATA_ROWS=$(wc -l < "${DATA}")"
cd "${REPO}"
ARGS=(--model_path "${MODEL}" --datasets "${TASK}" --data_file "${DATA}"
      --save_dir "${OUT}" --eval_batch_size 1 --use_cache True
      --attn_implementation flash_attention_2 --dtype bfloat16
      --method "${METHOD}" --seed 42)
if [[ "${METHOD}" != "FullKV" ]]; then ARGS+=(--max_capacity_prompts "${BUDGET}"); fi
"${ENV}/bin/python" run_longbench.py "${ARGS[@]}"
printf "git_commit=%s\nmodel=%s\ntask=%s\nmethod=%s\nbudget_tokens=%s\ndata=%s\n" \
  "$(git rev-parse HEAD)" "${MODEL}" "${TASK}" "${METHOD}" "${BUDGET}" "${DATA}" > "${OUT}/PROVENANCE.txt"
touch "${OUT}/DONE"
