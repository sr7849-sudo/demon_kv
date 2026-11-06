from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = [
    "actual_seconds",
    "generated_tokens_per_second",

    "incremental_peak_allocated_gpu_bytes",
    "incremental_peak_reserved_gpu_bytes",

    "nvml_peak_memory_bytes",
    "nvml_mean_memory_bytes",

    "gpu_mean_power_w",
    "gpu_peak_power_w",
    "gpu_energy_j",

    "gpu_mean_util_pct",
    "gpu_peak_util_pct",
]


def p05(x):
    return np.quantile(
        x,
        0.05,
    )


def p95(x):
    return np.quantile(
        x,
        0.95,
    )


p05.__name__ = "p05"
p95.__name__ = "p95"


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input",
        default=(
            "experiments/full_oracle_qwen3_8b/"
            "hardware/a100_profile/raw_runs.csv"
        ),
    )

    p.add_argument(
        "--output",
        default=(
            "experiments/full_oracle_qwen3_8b/"
            "hardware/a100_profile/"
            "aggregated_request_policy.csv"
        ),
    )

    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(
        args.input
    )

    required = (
        [
            "request_id",
            "task",
            "policy",
            "context_tokens",
            "repetition",
        ]
        + METRICS
    )

    missing = [
        x
        for x in required
        if x not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    print(
        "RAW ROWS:",
        len(df),
    )

    print(
        "REQUESTS:",
        df.request_id.nunique(),
    )

    print(
        "POLICIES:",
        df.policy.nunique(),
    )

    print(
        "REPETITIONS:",
        sorted(
            df.repetition.unique()
        ),
    )

    grouped = (
        df.groupby(
            [
                "request_id",
                "task",
                "policy",
                "family",
                "memory_ratio",
                "context_tokens",
            ],
            dropna=False,
        )[METRICS]
        .agg(
            [
                "mean",
                "median",
                "std",
                p05,
                p95,
            ]
        )
    )

    grouped.columns = [
        f"{metric}_{stat}"
        for metric, stat
        in grouped.columns
    ]

    grouped = (
        grouped
        .reset_index()
    )

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    grouped.to_csv(
        output,
        index=False,
    )

    # -----------------------------------------
    # Coefficient of variation diagnostic.
    # -----------------------------------------
    grouped[
        "runtime_cv"
    ] = (
        grouped[
            "actual_seconds_std"
        ]
        /
        grouped[
            "actual_seconds_mean"
        ]
    )

    grouped[
        "energy_cv"
    ] = (
        grouped[
            "gpu_energy_j_std"
        ]
        /
        grouped[
            "gpu_energy_j_mean"
        ]
    )

    grouped[
        "memory_cv"
    ] = (
        grouped[
            "incremental_peak_allocated_gpu_bytes_std"
        ]
        /
        grouped[
            "incremental_peak_allocated_gpu_bytes_mean"
        ]
    )

    grouped.to_csv(
        output,
        index=False,
    )

    print(
        "\nRUNTIME CV"
    )

    print(
        grouped[
            "runtime_cv"
        ]
        .describe()
        .round(4)
    )

    print(
        "\nENERGY CV"
    )

    print(
        grouped[
            "energy_cv"
        ]
        .describe()
        .round(4)
    )

    print(
        "\nPOLICY SUMMARY"
    )

    policy = (
        grouped
        .groupby("policy")
        .agg(
            runtime_median=(
                "actual_seconds_median",
                "mean",
            ),

            incremental_memory_gb=(
                "incremental_peak_allocated_gpu_bytes_median",
                lambda x: x.mean()
                / 1e9,
            ),

            nvml_peak_memory_gb=(
                "nvml_peak_memory_bytes_median",
                lambda x: x.mean()
                / 1e9,
            ),

            mean_power_w=(
                "gpu_mean_power_w_median",
                "mean",
            ),

            energy_j=(
                "gpu_energy_j_median",
                "mean",
            ),

            tokens_per_second=(
                "generated_tokens_per_second_median",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        policy
        .round(4)
        .to_string(
            index=False
        )
    )

    policy.to_csv(
        output.parent
        / "aggregated_policy_summary.csv",
        index=False,
    )

    print(
        "\nHARDWARE AGGREGATION: PASS"
    )


if __name__ == "__main__":
    main()
