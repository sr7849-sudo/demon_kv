from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default=(
            "experiments/full_oracle_qwen3_8b/"
            "oracle_longbench_qwen3_8b.csv"
        ),
    )
    p.add_argument(
        "--outdir",
        default=(
            "experiments/full_oracle_qwen3_8b/"
            "analysis/02_request_heterogeneity"
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    compressed = df[
        df["policy"] != "full"
    ].copy()

    if "context_tokens" not in compressed.columns:
        raise RuntimeError(
            "context_tokens missing from oracle dataset"
        )

    request_summary = (
        compressed
        .groupby(
            ["request_id", "task"],
            as_index=False,
        )
        .agg(
            context_tokens=("context_tokens", "first"),
            mean_degradation=("degradation", "mean"),
            median_degradation=("degradation", "median"),
            std_degradation=("degradation", "std"),
            min_degradation=("degradation", "min"),
            max_degradation=("degradation", "max"),
            degradation_range=(
                "degradation",
                lambda x: x.max() - x.min(),
            ),
            mean_signed_delta=("signed_delta", "mean"),
            improved_policy_count=(
                "signed_delta",
                lambda x: int((x < 0).sum()),
            ),
            policies_over_001=(
                "degradation",
                lambda x: int((x > 0.01).sum()),
            ),
            policies_over_005=(
                "degradation",
                lambda x: int((x > 0.05).sum()),
            ),
            policies_over_010=(
                "degradation",
                lambda x: int((x > 0.10).sum()),
            ),
            catastrophic_policy_count=(
                "degradation",
                lambda x: int((x >= 0.25).sum()),
            ),
        )
    )

    request_summary.to_csv(
        out / "request_sensitivity_summary.csv",
        index=False,
    )

    # Determine best/worst policy for each request.
    worst_idx = (
        compressed
        .groupby("request_id")["degradation"]
        .idxmax()
    )

    best_idx = (
        compressed
        .groupby("request_id")["degradation"]
        .idxmin()
    )

    extremes = (
        compressed.loc[
            worst_idx,
            [
                "request_id",
                "task",
                "policy",
                "degradation",
            ],
        ]
        .rename(
            columns={
                "policy": "worst_policy",
                "degradation": "worst_degradation",
            }
        )
        .merge(
            compressed.loc[
                best_idx,
                [
                    "request_id",
                    "policy",
                    "degradation",
                ],
            ].rename(
                columns={
                    "policy": "best_policy",
                    "degradation": "best_degradation",
                }
            ),
            on="request_id",
        )
    )

    extremes.to_csv(
        out / "request_policy_extremes.csv",
        index=False,
    )

    # Same-task + approximately same-length comparison.
    compressed["length_bin"] = -1

    for task, idx in compressed.groupby(
        "task"
    ).groups.items():
        req_lengths = (
            compressed.loc[idx]
            [["request_id", "context_tokens"]]
            .drop_duplicates()
            .set_index("request_id")["context_tokens"]
        )

        try:
            bins = pd.qcut(
                req_lengths,
                q=4,
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            bins = pd.Series(
                0,
                index=req_lengths.index,
            )

        mapping = bins.to_dict()

        mask = compressed["task"] == task
        compressed.loc[
            mask,
            "length_bin",
        ] = (
            compressed.loc[
                mask,
                "request_id",
            ]
            .map(mapping)
            .astype(int)
        )

    heterogeneity = (
        compressed
        .groupby(
            ["task", "policy", "length_bin"],
            as_index=False,
        )
        .agg(
            n=("request_id", "nunique"),
            context_min=("context_tokens", "min"),
            context_median=("context_tokens", "median"),
            context_max=("context_tokens", "max"),
            mean_degradation=("degradation", "mean"),
            std_degradation=("degradation", "std"),
            min_degradation=("degradation", "min"),
            max_degradation=("degradation", "max"),
            iqr_degradation=(
                "degradation",
                lambda x: (
                    x.quantile(0.75)
                    - x.quantile(0.25)
                ),
            ),
            degradation_range=(
                "degradation",
                lambda x: x.max() - x.min(),
            ),
            robust_005_rate=(
                "degradation",
                lambda x: (x <= 0.05).mean(),
            ),
            sensitive_010_rate=(
                "degradation",
                lambda x: (x > 0.10).mean(),
            ),
        )
    )

    heterogeneity.to_csv(
        out / "within_task_length_bin_heterogeneity.csv",
        index=False,
    )

    policy_task = (
        compressed
        .groupby(
            ["task", "policy"],
            as_index=False,
        )
        .agg(
            n=("request_id", "nunique"),
            mean=("degradation", "mean"),
            std=("degradation", "std"),
            median=("degradation", "median"),
            q25=(
                "degradation",
                lambda x: x.quantile(0.25),
            ),
            q75=(
                "degradation",
                lambda x: x.quantile(0.75),
            ),
            min=("degradation", "min"),
            max=("degradation", "max"),
            zero_loss_rate=(
                "degradation",
                lambda x: (x == 0).mean(),
            ),
            sensitive_005_rate=(
                "degradation",
                lambda x: (x > 0.05).mean(),
            ),
            sensitive_010_rate=(
                "degradation",
                lambda x: (x > 0.10).mean(),
            ),
            catastrophic_rate=(
                "degradation",
                lambda x: (x >= 0.25).mean(),
            ),
        )
    )

    policy_task["iqr"] = (
        policy_task["q75"]
        - policy_task["q25"]
    )

    policy_task.to_csv(
        out / "task_policy_heterogeneity.csv",
        index=False,
    )

    print("\nREQUEST SENSITIVITY")
    print(
        request_summary[
            [
                "mean_degradation",
                "max_degradation",
                "degradation_range",
                "policies_over_005",
                "catastrophic_policy_count",
            ]
        ]
        .describe()
        .round(4)
    )

    print(
        "\nWithin-task/length-bin table written to:"
    )
    print(
        out / "within_task_length_bin_heterogeneity.csv"
    )

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))

        for task, g in request_summary.groupby(
            "task"
        ):
            ax.scatter(
                g["context_tokens"],
                g["max_degradation"],
                label=task,
                alpha=0.7,
                s=22,
            )

        ax.set_xlabel("Context tokens")
        ax.set_ylabel(
            "Worst degradation across compressed policies"
        )
        ax.set_title(
            "Request Sensitivity vs. Context Length"
        )
        ax.legend(
            fontsize=8,
            loc="best",
        )
        fig.tight_layout()
        fig.savefig(
            out
            / "sensitivity_vs_context_length.png",
            dpi=250,
        )
        plt.close(fig)

    except ImportError:
        pass

    print("\nREQUEST HETEROGENEITY: PASS")


if __name__ == "__main__":
    main()
