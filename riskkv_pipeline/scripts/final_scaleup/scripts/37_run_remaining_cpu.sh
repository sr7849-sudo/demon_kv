#!/usr/bin/env bash
set -euo pipefail

cd /scratch/sr7849/KV_Cache/riskkv_pipeline

PY=/home/sr7849/.conda/envs/riskkv/bin/python
LOGDIR=experiments/final_scaleup/logs
ANALYSIS=experiments/final_scaleup/analysis

mkdir -p "$LOGDIR" "$ANALYSIS"

LOG="$LOGDIR/remaining_cpu_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "RiskKV remaining CPU pipeline"
echo "START: $(date -Iseconds)"
echo "HOST : $(hostname)"
echo "LOG  : $LOG"
echo "============================================================"

echo
echo "=== [1/6] Conservative risk calibration ==="
$PY scripts/final_scaleup/scripts/23_calibrate.py \
  --delta 0.10 \
  --confidence 0.95 \
  --bins 12

echo
echo "=== [2/6] RiskKV controller selection ==="
$PY scripts/final_scaleup/scripts/24_run_selection.py \
  --delta 0.10 \
  --epsilon 0.05

echo
echo "=== [3/6] Controller + baseline evaluation ==="
$PY scripts/final_scaleup/scripts/25_eval_controller.py \
  --delta 0.10 \
  --epsilon 0.05 \
  --bootstrap 2000

echo
echo "=== [4/6] Matched-candidate MetaKV-style baseline ==="
$PY scripts/final_scaleup/scripts/26_run_metakv_style.py \
  --delta 0.10

echo
echo "=== [5/6] Sparse quality-audit simulation ==="
$PY scripts/final_scaleup/scripts/28_run_quality_audits.py \
  --delta 0.10 \
  --audit-rates 0,0.01,0.05,0.10,0.20 \
  --seeds 20

echo
echo "=== [6/6] Resource-profiling experiment manifest ==="
$PY scripts/final_scaleup/scripts/28b_make_resource_profile_plan.py \
  --n 100 \
  --repeats 3

echo
echo "============================================================"
echo "SUMMARY FILES"
echo "============================================================"

for f in \
  "$ANALYSIS/risk_model/cv_summary.json" \
  "$ANALYSIS/risk_model/calibrator.json" \
  "$ANALYSIS/selection_riskkv.summary.json" \
  "$ANALYSIS/controller_report.json" \
  "$ANALYSIS/selection_metakv_style.summary.json"
do
  echo
  echo "----- $f -----"
  if [[ -f "$f" ]]; then
    cat "$f"
  else
    echo "MISSING"
  fi
done

echo
echo "============================================================"
echo "FINAL SANITY"
echo "============================================================"

$PY - <<'PY'
from pathlib import Path
import pandas as pd
import json

root = Path("experiments/final_scaleup/analysis")

required = [
    root / "oracle_kvcf.csv",
    root / "features_kvcf.csv",
    root / "risk_model/oof_predictions.csv",
    root / "risk_model/calibrator.json",
    root / "selection_riskkv.csv",
    root / "controller_report.json",
    root / "selection_metakv_style.csv",
    root / "quality_audit_sim.csv",
]

missing = [str(p) for p in required if not p.exists()]

if missing:
    print("MISSING FILES:")
    for x in missing:
        print(" ", x)
    raise SystemExit(1)

oracle = pd.read_csv(root / "oracle_kvcf.csv")
rk = pd.read_csv(root / "selection_riskkv.csv")
mk = pd.read_csv(root / "selection_metakv_style.csv")

assert len(oracle) == 27951, len(oracle)
assert oracle.request_id.nunique() == 1331
assert oracle.config.nunique() == 21

assert len(rk) == 1331, len(rk)
assert rk.request_id.nunique() == 1331

assert len(mk) == 1331, len(mk)
assert mk.request_id.nunique() == 1331

print("oracle rows       :", len(oracle))
print("oracle requests   :", oracle.request_id.nunique())
print("oracle configs    :", oracle.config.nunique())
print("RiskKV selections :", len(rk))
print("MetaKV-style sel. :", len(mk))
print("SANITY: PASS")
PY

echo
echo "============================================================"
echo "COMPLETE: $(date -Iseconds)"
echo "============================================================"
