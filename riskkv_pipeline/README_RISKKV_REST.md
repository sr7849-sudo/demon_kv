# RiskKV — post-KVCF pipeline

This bundle starts from the frozen 189/189 KVCF matrix and builds the remaining
offline RiskKV analysis pipeline plus the DynamicKV matched-budget launcher.

The CPU/offline track:
- scores all KVCF generations with official LongBench per-request metrics,
- constructs `quality`, FullKV-relative `degradation`, nominal retention/saving,
- extracts runtime-available request/candidate features,
- trains 5×5 repeated request-grouped quality-risk models,
- fits a conservative one-sided Clopper–Pearson risk envelope,
- runs RiskKV selection with FullKV fail-safe,
- evaluates static, task-static, oracle and matched-candidate MetaKV-style baselines,
- bootstraps controller confidence intervals,
- prepares sparse-quality-audit and resource-drift analyses.

The GPU track:
- minimally patches official DynamicKV LongBench code so it respects one frozen
  `--data_file` and one `--dataset`,
- runs the same 9 frozen tasks × budgets {256,512,1024,2048,4096},
- uses the official Mistral DynamicKV settings: DynamicKV_v11, window 8,
  kernel 7, maxpool, radio_max 10.

## Install into existing repo

Copy/rsync this bundle's contents into:

    /scratch/sr7849/KV_Cache/riskkv_pipeline/

Then:

    cd /scratch/sr7849/KV_Cache/riskkv_pipeline
    /home/sr7849/.conda/envs/riskkv/bin/pip install -r requirements_rest.txt
    export PYTHONPATH=$PWD:${PYTHONPATH:-}

Do not modify the frozen KVCF environment.

## CPU pipeline

First run the smoke:

    /home/sr7849/.conda/envs/riskkv/bin/python \
      scripts/final_scaleup/scripts/35_smoke_test_rest.py

Then either run everything:

    bash scripts/final_scaleup/scripts/36_run_cpu_pipeline.sh

or step by step:

    /home/sr7849/.conda/envs/riskkv/bin/python scripts/final_scaleup/scripts/20_score_kvcf.py
    /home/sr7849/.conda/envs/riskkv/bin/python scripts/final_scaleup/scripts/21_build_features.py

    /home/sr7849/.conda/envs/riskkv/bin/python \
      scripts/final_scaleup/scripts/22_train_risk.py \
      --delta 0.10 --repeats 5 --folds 5 --ensemble 10

    /home/sr7849/.conda/envs/riskkv/bin/python \
      scripts/final_scaleup/scripts/23_calibrate.py \
      --delta 0.10 --confidence 0.95 --bins 12

    /home/sr7849/.conda/envs/riskkv/bin/python \
      scripts/final_scaleup/scripts/24_run_selection.py \
      --delta 0.10 --epsilon 0.05

    /home/sr7849/.conda/envs/riskkv/bin/python \
      scripts/final_scaleup/scripts/25_eval_controller.py \
      --delta 0.10 --epsilon 0.05 --bootstrap 2000

    /home/sr7849/.conda/envs/riskkv/bin/python \
      scripts/final_scaleup/scripts/26_run_metakv_style.py \
      --delta 0.10

    /home/sr7849/.conda/envs/riskkv/bin/python \
      scripts/final_scaleup/scripts/28_run_quality_audits.py \
      --delta 0.10 --audit-rates 0,0.01,0.05,0.10,0.20 --seeds 20

    /home/sr7849/.conda/envs/riskkv/bin/python \
      scripts/final_scaleup/scripts/28b_make_resource_profile_plan.py \
      --n 100 --repeats 3

## DynamicKV

Patch official `external/DynamicKV/run/longbench/pred.py`:

    /home/sr7849/.conda/envs/riskkv/bin/python \
      scripts/final_scaleup/scripts/29_patch_dynamickv_frozen_data.py

Build manifest:

    /home/sr7849/.conda/envs/riskkv/bin/python \
      scripts/final_scaleup/scripts/30_make_dynamickv_jobs.py

Default model:

    mistralai/Mistral-7B-Instruct-v0.2

Override only if your frozen KVCF final campaign used a different DynamicKV-supported model:

    export RISKKV_DYN_MODEL_PATH=/path/or/hf/model

Submit:

    sbatch --array=0-44%4 \
      scripts/final_scaleup/scripts/32_dynamickv_array.sbatch

Status:

    /home/sr7849/.conda/envs/riskkv/bin/python \
      scripts/final_scaleup/scripts/33_dynamickv_status.py

Ingest after 45/45:

    /home/sr7849/.conda/envs/riskkv/bin/python \
      scripts/final_scaleup/scripts/34_ingest_dynamickv.py

## Scientific conventions

`quality` is a per-request LongBench metric in [0,1].

    D(r,p) = max(0, Q_full(r) - Q_p(r))
    violation = D > delta

`nominal_saving = 1 - retained_tokens / prompt_tokens` is an algorithmic
retained-token saving. It is **not** measured GPU-memory saving.

For fixed budgets:
    retained_tokens = min(prompt_tokens, budget)

FullKV is always available as a fail-safe.

The MetaKV-style baseline in this bundle is intentionally labeled
"matched-candidate MetaKV-style". It is an open-loop absolute-quality predictor
over the same candidate library. It is not a claim that the official MetaKV
repository has been reproduced. Keep an official MetaKV reproduction as a
separate baseline.

Resource drift simulation requires real measurements with:

    request_id,policy,budget,latency_ms,peak_memory_mb

The simulation harness is not a replacement for the live A100 contention/drift
experiment; use it only to validate analysis code and controller logic.

## One important setting

`20_score_kvcf.py` defaults to Mistral prompt-token counts when both Mistral and
Qwen counts exist because the completed KVCF trace shown during bring-up used
Mistral attention code. If the final 189-cell campaign was actually a Qwen
campaign, run instead:

    python scripts/final_scaleup/scripts/20_score_kvcf.py \
      --model-token-field riskkv_qwen_tokens

Do not leave this ambiguous in the paper.
