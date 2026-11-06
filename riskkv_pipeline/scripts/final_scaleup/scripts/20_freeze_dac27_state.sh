#!/usr/bin/env bash
set -euo pipefail

ROOT=/scratch/sr7849/KV_Cache/riskkv_pipeline
OUT="${ROOT}/experiments/final_dac27/audit"
mkdir -p "${OUT}"

STAMP=$(date +%Y%m%d_%H%M%S)
REPORT="${OUT}/state_${STAMP}.txt"

{
  echo "=== RISKKV DAC27 FREEZE ==="
  date -Is
  hostname
  echo

  echo "=== MAIN REPO ==="
  cd "${ROOT}"
  git rev-parse HEAD 2>/dev/null || true
  git status --short 2>/dev/null || true
  echo

  for repo in KVCache-Factory AdaKV DynamicKV LongBench; do
    P="${ROOT}/external/${repo}"
    echo "=== ${repo} ==="
    if [[ -d "${P}/.git" ]]; then
      git -C "${P}" rev-parse HEAD
      git -C "${P}" status --short
    else
      echo "NOT FOUND"
    fi
    echo
  done

  echo "=== KVCF ENV ==="
  /scratch/sr7849/conda_envs/riskkv-kvcf/bin/python - <<'PY'
import sys, torch, transformers
print("python:", sys.version)
print("torch:", torch.__version__)
print("torch_cuda:", torch.version.cuda)
print("transformers:", transformers.__version__)
try:
    import flash_attn
    print("flash_attn:", flash_attn.__version__)
except Exception as e:
    print("flash_attn: ERROR", repr(e))
PY

  echo
  echo "=== DATA SHA256 ==="
  for f in "${ROOT}"/external/KVCache-Factory/data/LongBench/*.jsonl; do
    case "$(basename "$f")" in
      qasper.jsonl|multifieldqa_en.jsonl|hotpotqa.jsonl|triviaqa.jsonl|gov_report.jsonl|multi_news.jsonl|trec.jsonl|passage_retrieval_en.jsonl|passage_count.jsonl)
        sha256sum "$f"
        ;;
    esac
  done
} > "${REPORT}"

echo "WROTE ${REPORT}"
