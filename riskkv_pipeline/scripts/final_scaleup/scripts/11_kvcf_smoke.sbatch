#!/bin/bash
#SBATCH --job-name=rkv_kvcf_smoke
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=experiments/final_scaleup/smoke/kvcf_%j.out
#SBATCH --error=experiments/final_scaleup/smoke/kvcf_%j.err
set -e
ROOT=/scratch/sr7849/KV_Cache/riskkv_pipeline
ENV=/scratch/sr7849/conda_envs/riskkv-kvcf
REPO=${ROOT}/external/KVCache-Factory
DATA=${ROOT}/experiments/final_scaleup/smoke/data/qasper_1.jsonl
MODEL=mistralai/Mistral-7B-Instruct-v0.2
export PATH="${ENV}/bin:${PATH}"

# Required by AdaKV's compiled tiny_api_cuda extension.
# Make PyTorch shared libraries (libc10.so, libtorch*.so) directly visible.
export LD_LIBRARY_PATH="${ENV}/lib/python3.11/site-packages/torch/lib:${ENV}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/scratch/sr7849/hf_cache
export TRANSFORMERS_CACHE=/scratch/sr7849/hf_cache
export TMPDIR=/scratch/sr7849/tmp
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
mkdir -p "${TMPDIR}"
cd "${REPO}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
run_one () {
  METHOD="$1"; BUDGET="$2"
  OUT="${ROOT}/experiments/final_scaleup/smoke/kvcf_${METHOD}_${BUDGET}"
  mkdir -p "${OUT}"
  ARGS=(--model_path "${MODEL}" --datasets qasper --data_file "${DATA}"
        --save_dir "${OUT}" --max_num_examples 1 --eval_batch_size 1
        --use_cache True --attn_implementation flash_attention_2
        --dtype bfloat16 --method "${METHOD}" --seed 42)
  if [[ "${METHOD}" != "FullKV" ]]; then ARGS+=(--max_capacity_prompts "${BUDGET}"); fi
  python run_longbench.py "${ARGS[@]}"
}
run_one FullKV -1
run_one pyramidkv 512
run_one adakv 512
echo "KVCF GPU SMOKE: PASS"
