from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--profiles",
        default=(
            "experiments/full_oracle_qwen3_8b/"
            "analysis/08_compression_profiles/"
            "oracle_request_profiles.csv"
        ),
    )

    p.add_argument(
        "--predictions",
        default=(
            "experiments/full_oracle_qwen3_8b/"
            "analysis/09_profile_prediction/"
            "request_profile_oof.csv"
        ),
    )

    p.add_argument(
        "--outdir",
        default=(
            "experiments/full_oracle_qwen3_8b/"
            "analysis/11_profile_analysis"
        ),
    )

    return p.parse_args()


def main():
    args = parse_args()

    out = Path(args.outdir)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    p = pd.read_csv(
        args.profiles
    )

    pred = pd.read_csv(
        args.predictions
    )

    # -----------------------------------------------------
    # Compression headroom by task
    # -----------------------------------------------------
    headroom_cols = [
        "headroom_0p01",
        "headroom_0p05",
        "headroom_0p10",
    ]

    headroom_by_task = (
        p.groupby("task")[
            headroom_cols
        ]
        .agg(
            [
                "mean",
                "median",
                "std",
            ]
        )
    )

    headroom_by_task.to_csv(
        out
        / "headroom_by_task.csv"
    )

    # -----------------------------------------------------
    # Fragility by task
    # -----------------------------------------------------
    fragility = (
        p.groupby(
            "task",
            as_index=False,
        )
        .agg(
            n=("request_id", "nunique"),
            mean_fragility=(
                "fragility_score",
                "mean",
            ),
            median_fragility=(
                "fragility_score",
                "median",
            ),
            mean_max_jump=(
                "max_adjacent_jump",
                "mean",
            ),
            cliff_rate=(
                "cliff",
                "mean",
            ),
            catastrophic_any_rate=(
                "catastrophic_any",
                "mean",
            ),
        )
    )

    fragility.to_csv(
        out
        / "fragility_by_task.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Headroom vs fragility
    # -----------------------------------------------------
    correlations = []

    for headroom in headroom_cols:
        for metric in [
            "fragility_score",
            "max_adjacent_jump",
            "fragility_auc_max",
        ]:
            correlations.append(
                {
                    "headroom": headroom,
                    "fragility_metric": metric,
                    "spearman": (
                        p[
                            headroom
                        ].corr(
                            p[
                                metric
                            ],
                            method="spearman",
                        )
                    ),
                }
            )

    correlations = pd.DataFrame(
        correlations
    )

    correlations.to_csv(
        out
        / "headroom_fragility_correlations.csv",
        index=False,
    )

    print("\nHEADROOM BY TASK")
    print(
        headroom_by_task.round(4)
    )

    print("\nFRAGILITY BY TASK")
    print(
        fragility
        .round(4)
        .to_string(index=False)
    )

    print(
        "\nHEADROOM ↔ FRAGILITY CORRELATION"
    )

    print(
        correlations
        .round(4)
        .to_string(index=False)
    )

    print("\nPROFILE ANALYSIS: PASS")


if __name__ == "__main__":
    main()
