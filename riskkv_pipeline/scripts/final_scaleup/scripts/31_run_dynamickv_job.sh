#!/usr/bin/env bash
set -euo pipefail
ROOT=/scratch/sr7849/KV_Cache/riskkv_pipeline
DYN="$ROOT/external/DynamicKV"
ENV=/scratch/sr7849/conda_envs/riskkv-dynamickv
MAN="$ROOT/experiments/final_scaleup/manifests/dynamickv_final_jobs.csv"
JOB_ID="${1:?usage: $0 JOB_ID}"

read -r TASK BUDGET < <("$ENV/bin/python" - "$MAN" "$JOB_ID" <<'PY'
import csv,sys
p,j=sys.argv[1],int(sys.argv[2])
with open(p) as f:
    rows=list(csv.DictReader(f))
r=next((x for x in rows if int(x["job_id"])==j),None)
if r is None: raise SystemExit(f"job id {j} missing")
print(r["task"],r["budget"])
PY
)

MODEL_PATH="${RISKKV_DYN_MODEL_PATH:-mistralai/Mistral-7B-Instruct-v0.2}"
DATA="$ROOT/data/final_common_longbench_filtered/${TASK}.jsonl"
OUTROOT="$ROOT/experiments/final_scaleup/results/dynamickv/budget_${BUDGET}/${TASK}"
mkdir -p "$OUTROOT"

if [[ -f "$OUTROOT/DONE" ]]; then
  echo "SKIP DONE $TASK $BUDGET"
  exit 0
fi

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "==========="
echo "TASK=$TASK BUDGET=$BUDGET DATA=$DATA"
echo "DATA_ROWS=$(wc -l < "$DATA")"

TMP="$OUTROOT/raw"
rm -rf "$TMP"
mkdir -p "$TMP"

cd "$DYN"
PYTHONPATH="$DYN:${PYTHONPATH:-}" "$ENV/bin/python" run/longbench/pred.py \
  --save_dir "$TMP" \
  --model_path "$MODEL_PATH" \
  --model_name "mistral-7b-instruct-v0.2" \
  --method "DynamicKV_v11" \
  --dataset "$TASK" \
  --data_file "$DATA" \
  --max_capacity_prompts "$BUDGET" \
  --window_size 8 \
  --kernel_sizes 7 \
  --pooling maxpool \
  --radio_max 10 \
  --eval_batch_size 1 \
  --attn_implementation sdpa

PRED=$(find "$TMP" -type f -name 'dynamickv_v11.json' -size +0c | head -1)
if [[ -z "${PRED:-}" ]]; then
  echo "ERROR: DynamicKV output not found" >&2
  exit 2
fi
N=$(wc -l < "$PRED")
EXPECTED=$(wc -l < "$DATA")
if [[ "$N" -ne "$EXPECTED" ]]; then
  echo "ERROR: output rows $N != expected $EXPECTED" >&2
  exit 3
fi
cp "$PRED" "$OUTROOT/dynamickv.json"
cat > "$OUTROOT/run_meta.json" <<EOF
{"task":"$TASK","budget":$BUDGET,"model_path":"$MODEL_PATH","method":"DynamicKV_v11","window_size":8,"kernel_size":7,"pooling":"maxpool","radio_max":10,"rows":$N}
EOF
touch "$OUTROOT/DONE"
echo "DONE $TASK $BUDGET rows=$N"
