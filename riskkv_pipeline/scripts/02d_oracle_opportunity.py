from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DELTAS = [
    0.01,
    0.05,
    0.10,
]


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
            "analysis/04_oracle_opportunity"
        ),
    )

    p.add_argument(
        "--deltas",
        nargs="+",
        type=float,
        default=DEFAULT_DELTAS,
    )

    return p.parse_args()


def select_oracle(
    group,
    delta,
):
    feasible = group[
        group["degradation"]
        <= delta + 1e-12
    ].copy()

    if len(feasible) == 0:
        raise RuntimeError(
            "No feasible policy. FullKV should "
            "always have zero degradation."
        )

    # Primary objective:
    #   lowest memory footprint.
    #
    # Tie breaking:
    #   lower degradation, then lower runtime.
    feasible = feasible.sort_values(
        [
            "memory_ratio",
            "degradation",
            "actual_seconds",
            "policy",
        ],
        ascending=[
            True,
            True,
            True,
            True,
        ],
    )

    return feasible.iloc[0]


def main():
    args = parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    required = {
        "request_id",
        "task",
        "policy",
        "memory_ratio",
        "quality",
        "full_quality",
        "degradation",
        "signed_delta",
        "actual_seconds",
        "kv_bytes_full_theoretical",
        "kv_bytes_policy_estimated",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing columns: {sorted(missing)}"
        )

    all_selections = []
    summary_rows = []
    task_rows = []

    for delta in args.deltas:
        selections = []

        for request_id, group in df.groupby(
            "request_id",
            sort=False,
        ):
            chosen = select_oracle(
                group,
                delta,
            )

            row = chosen.to_dict()

            row["delta"] = delta
            row["memory_saved_ratio"] = (
                1.0
                - float(
                    chosen["memory_ratio"]
                )
            )
            row["kv_bytes_saved"] = (
                float(
                    chosen[
                        "kv_bytes_full_theoretical"
                    ]
                )
                - float(
                    chosen[
                        "kv_bytes_policy_estimated"
                    ]
                )
            )

            selections.append(row)

        selected = pd.DataFrame(
            selections
        )

        if selected["request_id"].nunique() != 290:
            raise RuntimeError(
                f"Expected 290 selections at delta={delta}"
            )

        if (
            selected["degradation"]
            > delta + 1e-12
        ).any():
            raise RuntimeError(
                f"Oracle constraint violation "
                f"at delta={delta}"
            )

        selected.to_csv(
            out
            / f"oracle_selection_delta_{delta:.3f}.csv",
            index=False,
        )

        all_selections.append(
            selected
        )

        summary_rows.append(
            {
                "delta": delta,
                "n_requests": (
                    selected["request_id"]
                    .nunique()
                ),
                "mean_memory_ratio": (
                    selected["memory_ratio"]
                    .mean()
                ),
                "median_memory_ratio": (
                    selected["memory_ratio"]
                    .median()
                ),
                "mean_memory_saved_ratio": (
                    selected[
                        "memory_saved_ratio"
                    ].mean()
                ),
                "median_memory_saved_ratio": (
                    selected[
                        "memory_saved_ratio"
                    ].median()
                ),
                "mean_quality": (
                    selected["quality"].mean()
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
                "compressed_request_rate": (
                    (
                        selected["policy"]
                        != "full"
                    ).mean()
                ),
                "mean_kv_bytes_saved": (
                    selected[
                        "kv_bytes_saved"
                    ].mean()
                ),
                "total_kv_bytes_saved": (
                    selected[
                        "kv_bytes_saved"
                    ].sum()
                ),
            }
        )

        task_summary = (
            selected
            .groupby(
                "task",
                as_index=False,
            )
            .agg(
                n_requests=(
                    "request_id",
                    "nunique",
                ),
                mean_memory_ratio=(
                    "memory_ratio",
                    "mean",
                ),
                mean_memory_saved_ratio=(
                    "memory_saved_ratio",
                    "mean",
                ),
                mean_degradation=(
                    "degradation",
                    "mean",
                ),
                max_degradation=(
                    "degradation",
                    "max",
                ),
                compressed_request_rate=(
                    "policy",
                    lambda x: (
                        x != "full"
                    ).mean(),
                ),
            )
        )

        task_summary.insert(
            0,
            "delta",
            delta,
        )

        task_rows.append(
            task_summary
        )

        policy_mix = (
            selected["policy"]
            .value_counts(
                normalize=False
            )
            .rename_axis("policy")
            .reset_index(name="count")
        )

        policy_mix["fraction"] = (
            policy_mix["count"]
            / len(selected)
        )

        policy_mix.insert(
            0,
            "delta",
            delta,
        )

        policy_mix.to_csv(
            out
            / f"policy_mix_delta_{delta:.3f}.csv",
            index=False,
        )

    overall = pd.DataFrame(
        summary_rows
    )

    overall.to_csv(
        out
        / "oracle_opportunity_summary.csv",
        index=False,
    )

    tasks = pd.concat(
        task_rows,
        ignore_index=True,
    )

    tasks.to_csv(
        out
        / "oracle_opportunity_by_task.csv",
        index=False,
    )

    combined = pd.concat(
        all_selections,
        ignore_index=True,
    )

    combined.to_csv(
        out
        / "all_oracle_selections.csv",
        index=False,
    )

    print("\nORACLE OPPORTUNITY")
    print(
        overall.round(4).to_string(
            index=False
        )
    )

    for delta in args.deltas:
        x = combined[
            combined["delta"] == delta
        ]

        print(
            f"\nPOLICY MIX @ delta={delta}"
        )

        print(
            x["policy"]
            .value_counts()
            .to_string()
        )

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.plot(
            overall["delta"],
            overall[
                "mean_memory_saved_ratio"
            ],
            marker="o",
        )

        ax.set_xlabel(
            "Allowed quality degradation δ"
        )
        ax.set_ylabel(
            "Mean oracle KV-memory saving"
        )
        ax.set_title(
            "Perfect-Oracle Memory Opportunity"
        )
        ax.set_ylim(
            0,
            min(
                1.0,
                max(
                    0.05,
                    overall[
                        "mean_memory_saved_ratio"
                    ].max()
                    * 1.15,
                ),
            ),
        )
        ax.grid(
            True,
            alpha=0.3,
        )

        fig.tight_layout()

        fig.savefig(
            out
            / "oracle_memory_saving_vs_delta.png",
            dpi=250,
        )

        plt.close(fig)

    except ImportError:
        pass

    print("\nORACLE OPPORTUNITY: PASS")


if __name__ == "__main__":
    main()
