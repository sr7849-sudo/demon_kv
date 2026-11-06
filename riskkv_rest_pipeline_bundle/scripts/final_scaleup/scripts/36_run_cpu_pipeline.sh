#!/usr/bin/env bash
set -euo pipefail
cd /scratch/sr7849/KV_Cache/riskkv_pipeline
PY=/home/sr7849/.conda/envs/riskkv/bin/python

$PY scripts/final_scaleup/scripts/35_smoke_test_rest.py
$PY scripts/final_scaleup/scripts/20_score_kvcf.py
$PY scripts/final_scaleup/scripts/21_build_features.py
$PY scripts/final_scaleup/scripts/22_train_risk.py --delta 0.10 --repeats 5 --folds 5 --ensemble 10
$PY scripts/final_scaleup/scripts/23_calibrate.py --delta 0.10 --confidence 0.95 --bins 12
$PY scripts/final_scaleup/scripts/24_run_selection.py --delta 0.10 --epsilon 0.05
$PY scripts/final_scaleup/scripts/25_eval_controller.py --delta 0.10 --epsilon 0.05 --bootstrap 2000
$PY scripts/final_scaleup/scripts/26_run_metakv_style.py --delta 0.10
$PY scripts/final_scaleup/scripts/28_run_quality_audits.py --delta 0.10 --audit-rates 0,0.01,0.05,0.10,0.20 --seeds 20
$PY scripts/final_scaleup/scripts/28b_make_resource_profile_plan.py --n 100 --repeats 3
echo "CPU PIPELINE COMPLETE"
