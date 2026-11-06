"""
Shared utilities for RiskKV runtime-synthesis evaluation.

The intention is to keep ALL downstream experiments consistent:
- same quality-budget definition;
- same Pareto convention;
- same hardware measurements;
- same conservative risk score;
- same policy-selection tie breaking.

This prevents apparently different results from being caused by
slightly different evaluation code in different scripts.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


POLICY_ORDER = [
    "full",
    "snapkv_25",
    "snapkv_50",
    "snapkv_75",
    "knorm_25",
    "knorm_50",
    "knorm_75",
    "quant4",
]


DELTAS = [0.01, 0.05, 0.10]
RISK_TARGETS = [0.01, 0.05]

LAMBDAS = [
    0.0,
    0.25,
    0.50,
    0.75,
    1.00,
    1.50,
    2.00,
]


def pareto_mask(values):
    """
    Return True for non-dominated rows.

    Every supplied objective is minimized.
    """
    x = np.asarray(values, dtype=float)
    n = len(x)

    keep = np.ones(n, dtype=bool)

    for i in range(n):
        dominated = np.any(
            np.all(x <= x[i] + 1e-12, axis=1)
            & np.any(x < x[i] - 1e-12, axis=1)
        )

        if dominated:
            keep[i] = False

    return keep


def conservative_score(mean, uncertainty, lam):
    """
    Conservative predicted safety score.

        score = mean predicted safety - lambda * uncertainty

    lambda is NOT treated as a universal constant. Experiments which
    use it must select it using training/calibration data only.
    """
    return np.clip(
        np.asarray(mean, float)
        - float(lam) * np.asarray(uncertainty, float),
        0.0,
        1.0,
    )


def calibrate_threshold(y_safe, scores, target_violation):
    """
    Find the lowest score threshold giving maximum accepted coverage
    while keeping empirical violation among accepted points <= target.

    This is used on training/calibration data only.
    """
    y = np.asarray(y_safe, int)
    s = np.asarray(scores, float)

    thresholds = np.sort(
        np.unique(
            np.concatenate(
                [
                    s,
                    [0.0, 1.000001],
                ]
            )
        )
    )[::-1]

    best_threshold = 1.000001
    best_coverage = 0.0

    for threshold in thresholds:
        accepted = s >= threshold
        n = int(accepted.sum())

        if n == 0:
            continue

        violation = (
            ((y == 0) & accepted).sum()
            / n
        )

        coverage = n / len(y)

        if (
            violation <= target_violation + 1e-12
            and coverage > best_coverage
        ):
            best_threshold = float(threshold)
            best_coverage = float(coverage)

    return best_threshold


def hardware_columns(df):
    """
    Resolve the repeated-profile column names used by the current
    RiskKV A100 profiler.
    """
    return {
        "memory": "incremental_peak_allocated_gpu_bytes",
        "runtime": "actual_seconds",
        "energy": "gpu_energy_j",
        "power": "gpu_mean_power_w",
    }


def load_hardware_medians(
    raw_path=(
        "experiments/full_oracle_qwen3_8b/"
        "hardware/a100_profile/raw_runs.csv"
    ),
):
    """
    Convert the 5-run hardware profile into one stable median operating
    point per request-policy pair.
    """
    df = pd.read_csv(raw_path)
    cols = hardware_columns(df)

    result = (
        df.groupby(
            [
                "request_id",
                "task",
                "policy",
                "family",
                "memory_ratio",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            memory_bytes=(
                cols["memory"],
                "median",
            ),
            runtime_seconds=(
                cols["runtime"],
                "median",
            ),
            energy_j=(
                cols["energy"],
                "median",
            ),
            power_w=(
                cols["power"],
                "median",
            ),
            context_tokens=(
                "context_tokens",
                "first",
            ),
        )
    )

    return result


def load_quality_hardware(
    oracle_path=(
        "experiments/full_oracle_qwen3_8b/"
        "oracle_longbench_qwen3_8b.csv"
    ),
    hardware_path=(
        "experiments/full_oracle_qwen3_8b/"
        "hardware/a100_profile/raw_runs.csv"
    ),
):
    """
    Combine frozen oracle quality with stable repeated hardware medians.

    Quality and hardware have different provenance:
      degradation -> frozen oracle experiment
      M/T/E/P     -> five-run A100 profile
    """
    oracle = pd.read_csv(
        oracle_path
    )

    hw = load_hardware_medians(
        hardware_path
    )

    quality = (
        oracle[
            [
                "request_id",
                "policy",
                "quality",
                "full_quality",
                "degradation",
            ]
        ]
        .drop_duplicates(
            ["request_id", "policy"]
        )
    )

    x = hw.merge(
        quality,
        on=[
            "request_id",
            "policy",
        ],
        how="left",
        validate="one_to_one",
    )

    if x["degradation"].isna().any():
        bad = x[
            x["degradation"].isna()
        ][
            ["request_id", "policy"]
        ]

        raise RuntimeError(
            "Missing oracle quality for hardware points:\n"
            + bad.to_string(index=False)
        )

    return x


def evaluate_selection(selected, delta):
    """
    Compute the common system metrics for a set containing one selected
    configuration per request.
    """
    x = selected.copy()

    x["violation"] = (
        x["degradation"]
        > delta + 1e-12
    )

    x["catastrophic"] = (
        x["degradation"]
        >= 0.25
    )

    return {
        "n_requests": x[
            "request_id"
        ].nunique(),

        "mean_degradation": x[
            "degradation"
        ].mean(),

        "violation_rate": x[
            "violation"
        ].mean(),

        "catastrophic_rate": x[
            "catastrophic"
        ].mean(),

        "mean_memory_gb": x[
            "memory_bytes"
        ].mean() / 1e9,

        "mean_runtime_seconds": x[
            "runtime_seconds"
        ].mean(),

        "mean_energy_j": x[
            "energy_j"
        ].mean(),

        "mean_power_w": x[
            "power_w"
        ].mean(),

        "fullkv_rate": x[
            "policy"
        ].eq("full").mean(),
    }


def choose_min_objective(
    g,
    objective,
    feasible_mask=None,
):
    """
    Choose one configuration from a request.

    Tie-breaking is deterministic:
      primary deployment objective
      degradation
      memory
      runtime
      policy name
    """
    if feasible_mask is None:
        x = g.copy()
    else:
        x = g.loc[
            feasible_mask
        ].copy()

    if len(x) == 0:
        full = g[
            g["policy"] == "full"
        ]

        if len(full) != 1:
            raise RuntimeError(
                "Could not resolve FullKV fallback."
            )

        return full.iloc[0]

    sort_cols = [
        objective,
        "degradation",
        "memory_bytes",
        "runtime_seconds",
        "policy",
    ]

    sort_cols = list(
        dict.fromkeys(sort_cols)
    )

    return (
        x.sort_values(
            sort_cols
        )
        .iloc[0]
    )
