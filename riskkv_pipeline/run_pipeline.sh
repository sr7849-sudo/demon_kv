#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-configs/default.yaml}
python scripts/00_check_env.py --config "$CFG"
python scripts/01_collect_oracle.py --config "$CFG"
python scripts/02_characterize.py
python scripts/03_train_risk.py
python scripts/03b_feature_ablation.py
python scripts/03c_unseen_task.py
python scripts/04_evaluate_prediction.py
python scripts/05_provision.py --epsilon 0.02
python scripts/05b_compare_baselines.py --epsilon 0.02
python scripts/06_quality_sweep.py
python scripts/07_system_bench.py --config "$CFG"
python scripts/08_make_figures.py
