# RiskKV research pipeline

This repository is an experiment scaffold for three claims:

1. **Request-level KV sensitivity characterization** — measure within-task variation in quality loss under the same KV policy and control for context length.
2. **Lightweight policy-conditional compression-risk prediction** — predict `quality(full) - quality(policy)` from Full-KV prefill signals plus the candidate policy identifier.
3. **Quality-constrained KV provisioning** — choose the minimum-memory candidate whose predicted quality loss is <= epsilon, then compare with static baselines and the retrospective oracle.

## Important research boundary

The code is designed to test these claims; it does **not** assume they are true. A publishable result must survive held-out requests/tasks/models and show a favorable oracle gap, low false-safe rate, and low controller overhead. No code can guarantee conference acceptance.

## Why KVPress is the default execution substrate

NVIDIA KVPress already exposes multiple prefill compression methods through a common Transformers pipeline. This repo uses its public `kv-press-text-generation` interface rather than reimplementing SnapKV/K-Norm. Quantized KV uses Transformers `QuantizedCache` through the same pipeline where supported.

## Environment

Recommended: Linux + NVIDIA CUDA GPU; Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

For gated models such as Llama, authenticate with Hugging Face before running. The default config uses Qwen/Qwen3-8B to avoid a gated default, but you must ensure the selected checkpoint fits your GPU and license/usage constraints.

## Data format

Create `data/requests.jsonl`, one JSON object per request:

```json
{"request_id":"r1","task":"multi_doc_qa","metric":"f1","context":"...","question":"...","reference":"..."}
```

Supported metrics: `exact_match`, `f1`, `rouge_l`.

For serious experiments, use benchmark-native scorers when the benchmark defines one; the included scorers provide a consistent generic baseline only.

## End-to-end execution

First verify the non-GPU analysis stack without downloading any model:

```bash
python scripts/smoke_test_synthetic.py
pytest -q
```

Then inspect `configs/default.yaml` and run a small real pilot:

```bash
cp data/requests.example.jsonl data/requests.jsonl
python scripts/00_check_env.py --config configs/default.yaml
python scripts/01_collect_oracle.py --config configs/default.yaml --limit 2
```

For the real study:

```bash
./run_pipeline.sh configs/default.yaml
```

## Script map

### `01_collect_oracle.py`
For each request:
- one Full-KV prefill for request features;
- generation under each configured policy;
- task quality score;
- quality degradation from FullKV;
- wall-clock generation time and CUDA peak allocation.

Output: `results/oracle.csv`.

### `02_characterize.py`
Produces:
- task x policy degradation distributions;
- within-length-bin sensitivity controls;
- near-equal-length request pairs with maximal quality-loss differences.

Outputs under `results/characterization/`.

### `03_train_risk.py`
Group-splits by `request_id` so the same request never appears in train and test under different policies. Trains a policy-conditional lightweight regressor.

Outputs:
- `results/risk_model.pkl`
- `results/heldout_predictions.csv`

### `04_evaluate_prediction.py`
Reports MAE/RMSE and, for an unsafe-loss threshold, AUROC/AUPRC plus the false-safe rate among actually unsafe configurations.

### `05_provision.py`
For each held-out request, chooses the minimum-memory feasible policy under epsilon and computes the retrospective oracle and oracle memory gap.

### `06_quality_sweep.py`
Sweeps epsilon to produce the memory-quality frontier.

### `07_system_bench.py`
Reruns selected policies on GPU and records actual quality/time/peak allocation plus analytically estimated raw KV bytes. For DATE-quality reporting, augment this with Nsight Compute/DCGM HBM traffic counters and a serving benchmark for concurrency/TPOT/TTFT.

### `08_make_figures.py`
Creates publication-oriented sensitivity, prediction-calibration, and quality-memory plots.

## Feature design

The default portable feature extractor performs **one uncompressed prefill** and collects:
- context token count;
- prefill NLL / capped perplexity;
- final-token logit entropy, top-1 probability and top-1/top-2 margin;
- per-layer K/V norm, variance and max-absolute summaries.

Attention-entropy features are deliberately not faked. Flash Attention typically does not materialize full attention matrices; if the final paper needs attention entropy, implement a sampled eager-attention measurement on selected layers/query windows and measure its overhead explicitly. The current portable path therefore works without silently switching attention kernels.

## Required paper-grade experimental controls

Do not skip these:
- deterministic decoding for the oracle data unless stochasticity is itself studied;
- group split by request ID;
- held-out task experiment;
- second-model transfer/calibration experiment;
- context-length-matched counterexamples;
- fixed-policy, task-aware, length-aware and oracle baselines;
- epsilon sweep rather than a single operating point;
- repeated timing after warm-up;
- controller/feature extraction overhead;
- actual GPU memory plus, where possible, HBM traffic with Nsight/DCGM;
- confidence intervals / bootstrap intervals across requests;
- report false-safe errors, not only regression MAE.

## Suggested policy set

Start with one lossless baseline, multiple pruning strengths, a second pruning family, and 4-bit KV quantization. Do **not** add dozens of methods until the central request-level sensitivity hypothesis is established.

## Research outputs you want

A strong outcome would show:
- large within-task, length-controlled variance in degradation;
- prefill features predict policy-conditional degradation materially better than task/length-only baselines;
- provisioning substantially reduces average KV memory while keeping quality-constraint violations low;
- RiskKV approaches the retrospective oracle frontier with low runtime overhead;
- gains persist on held-out tasks and at least one second model.

Those are desired empirical outcomes, not hard-coded assumptions.
