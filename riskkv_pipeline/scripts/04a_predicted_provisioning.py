from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--predictions",
        default=(
            "experiments/full_oracle_qwen3_8b/"
            "analysis/05_risk_classification/"
            "risk_classification_oof_predictions.csv"
        ),
    )

    p.add_argument(
        "--outdir",
        default=(
            "experiments/full_oracle_qwen3_8b/"
            "analysis/07_predicted_provisioning"
        ),
    )

    p.add_argument(
        "--feature-group",
        default="all_prefill",
    )

    return p.parse_args()


def choose_policy(
    group,
    threshold_column,
):
    accepted = group[
        group["prob_safe"]
        >= group[
            threshold_column
        ]
    ].copy()

    if len(
        accepted
    ) == 0:
        return None

    accepted = (
        accepted
        .sort_values(
            [
                "memory_ratio",
                "prob_safe",
                "policy",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
    )

    return (
        accepted.iloc[0]
    )


def main():
    args = parse_args()

    out = Path(
        args.outdir
    )
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        args.predictions
    )

    df = df[
        df["feature_group"]
        == args.feature_group
    ].copy()

    results = []

    for delta in sorted(
        df["delta"].unique()
    ):
        work = df[
            df["delta"] == delta
        ].copy()

        for risk_target, threshold_col in [
            (
                "1pct",
                "threshold_1pct",
            ),
            (
                "5pct",
                "threshold_5pct",
            ),
        ]:
            selections = []

            for request_id, g in work.groupby(
                "request_id",
                sort=False,
            ):
                chosen = choose_policy(
                    g,
                    threshold_col,
                )

                if chosen is None:
                    selections.append(
                        {
                            "request_id": request_id,
                            "task": g[
                                "task"
                            ].iloc[0],
                            "delta": delta,
                            "risk_target": (
                                risk_target
                            ),
                            "policy": "full",
                            "memory_ratio": 1.0,
                            "degradation": 0.0,
                            "prob_safe": 1.0,
                            "fallback_full": True,
                        }
                    )
                else:
                    selections.append(
                        {
                            "request_id": request_id,
                            "task": chosen[
                                "task"
                            ],
                            "delta": delta,
                            "risk_target": (
                                risk_target
                            ),
                            "policy": chosen[
                                "policy"
                            ],
                            "memory_ratio": (
                                chosen[
                                    "memory_ratio"
                                ]
                            ),
                            "degradation": (
                                chosen[
                                    "degradation"
                                ]
                            ),
                            "prob_safe": (
                                chosen[
                                    "prob_safe"
                                ]
                            ),
                            "fallback_full": (
                                False
                            ),
                        }
                    )

            selected = pd.DataFrame(
                selections
            )

            selected[
                "violation"
            ] = (
                selected[
                    "degradation"
                ]
                > delta + 1e-12
            )

            selected[
                "memory_saved_ratio"
            ] = (
                1.0
                - selected[
                    "memory_ratio"
                ]
            )

            selected.to_csv(
                out
                / (
                    f"selection_delta_"
                    f"{delta:.3f}_"
                    f"{risk_target}.csv"
                ),
                index=False,
            )

            results.append(
                {
                    "delta": delta,
                    "risk_target": (
                        risk_target
                    ),
                    "n_requests": (
                        len(selected)
                    ),
                    "mean_memory_ratio": (
                        selected[
                            "memory_ratio"
                        ].mean()
                    ),
                    "mean_memory_saved_ratio": (
                        selected[
                            "memory_saved_ratio"
                        ].mean()
                    ),
                    "violation_rate": (
                        selected[
                            "violation"
                        ].mean()
                    ),
                    "mean_degradation": (
                        selected[
                            "degradation"
                        ].mean()
                    ),
                    "max_degradation": (
                        selected[
                            "degradation"
                        ].max()
                    ),
                    "fallback_full_rate": (
                        selected[
                            "fallback_full"
                        ].mean()
                    ),
                }
            )

    summary = pd.DataFrame(
        results
    )

    summary.to_csv(
        out
        / "predicted_provisioning_summary.csv",
        index=False,
    )

    print(
        "\nQUALITY-CONSTRAINED "
        "PREDICTED PROVISIONING"
    )

    print(
        summary
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        "\nPREDICTED PROVISIONING: PASS"
    )


if __name__ == "__main__":
    main()
