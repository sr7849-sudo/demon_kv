# Validation performed in the ChatGPT execution environment

Validated on 2026-08-10:

- `python -m compileall -q riskkv scripts` — PASS
- `pytest -q` — PASS (6 tests)
- synthetic end-to-end pipeline — PASS before cleanup:
  - sensitivity characterization
  - request-grouped predictor training
  - feature ablation
  - leave-one-task-out generalization
  - unsafe-risk evaluation
  - quality-constrained provisioning
  - static/task/length/oracle selector comparison
  - epsilon sweep
  - figure generation

The environment used for these checks did not expose an NVIDIA GPU or cached 8B model weights, so the real CUDA/model/KVPress execution path could not be executed here. The real backend is written against the current public NVIDIA KVPress API documented in its README (`pipeline("kv-press-text-generation", ...); pipe(context, question=..., press=...)`) and Transformers `QuantizedCache` usage. Run `scripts/00_check_env.py` and a 1–2 request pilot on the target NVIDIA machine before a full sweep.

Conference acceptance cannot be guaranteed by code. Novelty and acceptance depend on the empirical findings, related-work position, evaluation quality, and review process. This repository is structured so the central hypotheses can fail visibly rather than being hard-coded to produce a favorable result.
