"""
EXPERIMENT 6
============

RiskKV runtime decision-engine overhead.

Purpose
-------
Measures how expensive the runtime synthesis decision itself is once
prefill risk predictions are available.

This measures:
  - conservative risk scoring;
  - safe-set filtering;
  - candidate ranking;
  - final policy selection.

It does NOT yet include:
  - Qwen prefill itself;
  - extraction of model features;
  - risk-model inference.

Those should later be instrumented directly in the live serving path.

The current result therefore answers:
"Is design-space synthesis itself negligible?"
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd

from riskkv.synthesis_eval import (
    conservative_score,
)


RISK = (
    "experiments/full_oracle_qwen3_8b/"
    "analysis/09_profile_prediction/"
    "policy_risk_oof.csv"
)

HARDWARE = (
    "experiments/full_oracle_qwen3_8b/"
    "hardware/a100_profile/"
    "aggregated_request_policy.csv"
)

OUTDIR = Path(
    "experiments/full_oracle_qwen3_8b/"
    "analysis/20_controller_overhead"
)


REPEATS = 10000


def select_policy(g):
    """
    Representative runtime controller operation.

    The benchmark intentionally contains no file I/O.
    All candidate records are assumed to be resident in memory.
    """
    scores = conservative_score(
        g["prob_safe_mean"].to_numpy(),
        g[
            "risk_uncertainty"
        ].to_numpy(),
        0.5,
    )

    accepted = scores >= 0.90

    if not accepted.any():
        idx = np.where(
            g["policy"].to_numpy()
            == "full"
        )[0]

        if len(idx):
            return int(idx[0])

        return 0

    indices = np.where(
        accepted
    )[0]

    memory = g[
        "memory_ratio"
    ].to_numpy(float)

    best_local = indices[
        np.argmin(
            memory[indices]
        )
    ]

    return int(best_local)


def main():
    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    risk = pd.read_csv(
        RISK
    )

    # Use one delta to avoid benchmarking duplicated
    # candidate tables.
    risk = risk[
        np.isclose(
            risk["delta"],
            0.05,
        )
    ].copy()

    groups = [
        g.reset_index(
            drop=True
        )
        for _, g in risk.groupby(
            "request_id"
        )
    ]

    # Warmup Python/NumPy paths.
    for g in groups[:20]:
        select_policy(g)

    times_ns = []

    for i in range(
        REPEATS
    ):
        g = groups[
            i % len(groups)
        ]

        t0 = (
            time.perf_counter_ns()
        )

        select_policy(g)

        t1 = (
            time.perf_counter_ns()
        )

        times_ns.append(
            t1 - t0
        )

    x = (
        np.asarray(
            times_ns,
            dtype=float,
        )
        / 1e6
    )

    summary = pd.DataFrame(
        [
            {
                "iterations": (
                    REPEATS
                ),
                "mean_ms": x.mean(),
                "median_ms": (
                    np.median(x)
                ),
                "p95_ms": (
                    np.quantile(
                        x,
                        0.95,
                    )
                ),
                "p99_ms": (
                    np.quantile(
                        x,
                        0.99,
                    )
                ),
                "max_ms": x.max(),
            }
        ]
    )

    # Compare with measured A100 request latency.
    raw_hw = pd.read_csv(
        (
            "experiments/"
            "full_oracle_qwen3_8b/"
            "hardware/a100_profile/"
            "raw_runs.csv"
        )
    )

    inference_ms = (
        raw_hw[
            "actual_seconds"
        ].median()
        * 1000.0
    )

    summary[
        "median_inference_ms"
    ] = inference_ms

    summary[
        "overhead_fraction_percent"
    ] = (
        summary["median_ms"]
        / inference_ms
        * 100.0
    )

    summary.to_csv(
        OUTDIR
        / "controller_overhead.csv",
        index=False,
    )

    print(
        "\nCONTROLLER DECISION OVERHEAD"
    )

    print(
        summary.round(6)
        .to_string(index=False)
    )

    print(
        "\nNOTE: excludes feature extraction "
        "and risk-model inference."
    )

    print(
        "\nEXPERIMENT 6: PASS"
    )


if __name__ == "__main__":
    main()
