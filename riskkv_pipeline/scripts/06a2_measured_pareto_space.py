from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SPACES = {
    "nominal_2d": [
        "degradation",
        "memory_ratio",
    ],
    "measured_2d": [
        "degradation",
        "peak_incremental_gpu_bytes",
    ],
    "measured_3d": [
        "degradation",
        "peak_incremental_gpu_bytes",
        "actual_seconds",
    ],
}


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--oracle",
        default=(
            "experiments/full_oracle_qwen3_8b/"
            "oracle_longbench_qwen3_8b.csv"
        ),
    )

    p.add_argument(
        "--outdir",
        default=(
            "experiments/full_oracle_qwen3_8b/"
            "analysis/12b_measured_pareto_space"
        ),
    )

    return p.parse_args()


def pareto_mask(values: np.ndarray) -> np.ndarray:
    """
    Minimize every column.

    Point i is dominated if another point j is no worse
    in every objective and strictly better in at least one.
    """

    n = len(values)

    keep = np.ones(
        n,
        dtype=bool,
    )

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            no_worse = np.all(
                values[j]
                <= values[i]
                + 1e-12
            )

            strictly_better = np.any(
                values[j]
                < values[i]
                - 1e-12
            )

            if (
                no_worse
                and strictly_better
            ):
                keep[i] = False
                break

    return keep


def normalized_values(
    g: pd.DataFrame,
    columns: list[str],
) -> np.ndarray:
    """
    Normalize each objective per request.

    Normalization is not required for Pareto dominance,
    but keeping normalized values is useful for later
    distance/frontier analyses.
    """

    x = g[
        columns
    ].to_numpy(float)

    out = np.zeros_like(
        x,
        dtype=float,
    )

    for c in range(
        x.shape[1]
    ):
        col = x[:, c]

        lo = np.min(col)
        hi = np.max(col)

        if (
            hi - lo
            <= 1e-12
        ):
            out[:, c] = 0.0
        else:
            out[:, c] = (
                col - lo
            ) / (
                hi - lo
            )

    return out


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
        args.oracle
    )

    required = {
        "request_id",
        "task",
        "policy",
        "degradation",
        "memory_ratio",
        "peak_incremental_gpu_bytes",
        "actual_seconds",
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    print(
        "ROWS:",
        len(df),
    )

    print(
        "REQUESTS:",
        df[
            "request_id"
        ].nunique(),
    )

    # =====================================================
    # BUILD PARETO FRONTIERS FOR ALL SPACES
    # =====================================================

    all_point_rows = []
    request_rows = []

    for request_id, g in df.groupby(
        "request_id",
        sort=False,
    ):
        g = (
            g.copy()
            .reset_index(
                drop=True
            )
        )

        request_record = {
            "request_id": request_id,
            "task": g[
                "task"
            ].iloc[0],
        }

        for (
            space_name,
            columns,
        ) in SPACES.items():

            values = g[
                columns
            ].to_numpy(float)

            mask = pareto_mask(
                values
            )

            norm = normalized_values(
                g,
                columns,
            )

            sub = g[
                [
                    "request_id",
                    "task",
                    "policy",
                    "degradation",
                    "memory_ratio",
                    "peak_incremental_gpu_bytes",
                    "actual_seconds",
                ]
            ].copy()

            sub[
                "space"
            ] = space_name

            sub[
                "pareto"
            ] = mask

            for k, column in enumerate(
                columns
            ):
                sub[
                    f"norm_{column}"
                ] = norm[:, k]

            all_point_rows.append(
                sub
            )

            frontier = (
                g[
                    mask
                ]
                .sort_values(
                    "policy"
                )
            )

            signature = "|".join(
                sorted(
                    frontier[
                        "policy"
                    ].tolist()
                )
            )

            request_record[
                f"{space_name}_n_pareto"
            ] = len(
                frontier
            )

            request_record[
                f"{space_name}_signature"
            ] = signature

        request_rows.append(
            request_record
        )

    points = pd.concat(
        all_point_rows,
        ignore_index=True,
    )

    requests = pd.DataFrame(
        request_rows
    )

    points.to_csv(
        out
        / "measured_pareto_points.csv",
        index=False,
    )

    requests.to_csv(
        out
        / "measured_pareto_request_summary.csv",
        index=False,
    )

    # =====================================================
    # SPACE-LEVEL SUMMARY
    # =====================================================

    space_rows = []

    for space_name in SPACES:
        ncol = (
            f"{space_name}_n_pareto"
        )

        scol = (
            f"{space_name}_signature"
        )

        space_rows.append(
            {
                "space": space_name,
                "distinct_pareto_sets": (
                    requests[
                        scol
                    ].nunique()
                ),
                "mean_pareto_points": (
                    requests[
                        ncol
                    ].mean()
                ),
                "median_pareto_points": (
                    requests[
                        ncol
                    ].median()
                ),
                "min_pareto_points": (
                    requests[
                        ncol
                    ].min()
                ),
                "max_pareto_points": (
                    requests[
                        ncol
                    ].max()
                ),
            }
        )

    space_summary = pd.DataFrame(
        space_rows
    )

    space_summary.to_csv(
        out
        / "pareto_space_comparison.csv",
        index=False,
    )

    # =====================================================
    # POLICY FRONTIER FREQUENCY PER SPACE
    # =====================================================

    policy_rows = []

    for space_name, g in (
        points[
            points["pareto"]
        ]
        .groupby(
            "space"
        )
    ):
        freq = (
            g.groupby(
                "policy"
            )[
                "request_id"
            ]
            .nunique()
        )

        for (
            policy,
            n_requests,
        ) in freq.items():

            policy_rows.append(
                {
                    "space": (
                        space_name
                    ),
                    "policy": (
                        policy
                    ),
                    "requests_on_frontier": (
                        int(
                            n_requests
                        )
                    ),
                    "frontier_request_rate": (
                        n_requests
                        / df[
                            "request_id"
                        ].nunique()
                    ),
                }
            )

    policy_frequency = (
        pd.DataFrame(
            policy_rows
        )
    )

    policy_frequency.to_csv(
        out
        / "measured_pareto_policy_frequency.csv",
        index=False,
    )

    # =====================================================
    # FRONTIER DIVERSITY BY TASK
    # =====================================================

    task_rows = []

    for task, g in requests.groupby(
        "task"
    ):
        for space_name in SPACES:
            task_rows.append(
                {
                    "task": task,
                    "space": space_name,
                    "requests": len(g),
                    "mean_pareto_points": (
                        g[
                            f"{space_name}_n_pareto"
                        ].mean()
                    ),
                    "median_pareto_points": (
                        g[
                            f"{space_name}_n_pareto"
                        ].median()
                    ),
                    "distinct_frontiers": (
                        g[
                            f"{space_name}_signature"
                        ].nunique()
                    ),
                }
            )

    task_summary = pd.DataFrame(
        task_rows
    )

    task_summary.to_csv(
        out
        / "measured_pareto_by_task.csv",
        index=False,
    )

    # =====================================================
    # HOW OFTEN DOES THE FRONTIER CHANGE?
    # =====================================================

    transition_rows = []

    comparisons = [
        (
            "nominal_2d",
            "measured_2d",
        ),
        (
            "measured_2d",
            "measured_3d",
        ),
        (
            "nominal_2d",
            "measured_3d",
        ),
    ]

    for a, b in comparisons:
        changed = (
            requests[
                f"{a}_signature"
            ]
            != requests[
                f"{b}_signature"
            ]
        )

        transition_rows.append(
            {
                "space_a": a,
                "space_b": b,
                "fraction_frontier_changed": (
                    changed.mean()
                ),
                "requests_changed": int(
                    changed.sum()
                ),
                "requests_total": len(
                    requests
                ),
            }
        )

    transitions = pd.DataFrame(
        transition_rows
    )

    transitions.to_csv(
        out
        / "pareto_space_transitions.csv",
        index=False,
    )

    # =====================================================
    # MOST COMMON SIGNATURES FOR EACH SPACE
    # =====================================================

    signature_rows = []

    for space_name in SPACES:
        col = (
            f"{space_name}_signature"
        )

        counts = (
            requests[
                col
            ]
            .value_counts()
        )

        for (
            signature,
            count,
        ) in counts.items():

            signature_rows.append(
                {
                    "space": space_name,
                    "pareto_signature": (
                        signature
                    ),
                    "requests": int(
                        count
                    ),
                    "rate": (
                        count
                        / len(
                            requests
                        )
                    ),
                }
            )

    signatures = pd.DataFrame(
        signature_rows
    )

    signatures.to_csv(
        out
        / "measured_pareto_signature_frequency.csv",
        index=False,
    )

    # =====================================================
    # PRINT
    # =====================================================

    print(
        "\nPARETO SPACE COMPARISON"
    )

    print(
        space_summary
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        "\nFRONTIER CHANGES"
    )

    print(
        transitions
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        "\nPOLICY FREQUENCY BY SPACE"
    )

    print(
        policy_frequency
        .sort_values(
            [
                "space",
                "frontier_request_rate",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .round(4)
        .to_string(
            index=False
        )
    )

    for space_name in SPACES:
        print(
            f"\nTOP FRONTIERS: "
            f"{space_name}"
        )

        x = (
            signatures[
                signatures[
                    "space"
                ]
                == space_name
            ]
            .sort_values(
                "requests",
                ascending=False,
            )
            .head(
                15
            )
        )

        print(
            x.round(4)
            .to_string(
                index=False
            )
        )

    print(
        "\nPER-TASK FRONTIER DIVERSITY"
    )

    print(
        task_summary
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        "\nMEASURED PARETO ANALYSIS: PASS"
    )


if __name__ == "__main__":
    main()
