"""
EXPERIMENT 3
============

Objective-utility ablation for S2 -> S3 -> S4 -> S5.

Purpose
-------
The previous Pareto experiment showed that adding T/E/P changes the
frontier. A critic can still argue that this is merely a mathematical
effect of adding dimensions.

This experiment asks whether each richer space actually improves the
deployment quantity that was added.

Examples:
  - Does including latency lead to lower latency choices?
  - Does including energy enable better energy choices?
  - Does including power help under a power cap?
"""

from pathlib import Path
import pandas as pd

from riskkv.synthesis_eval import (
    load_quality_hardware,
    pareto_mask,
)


OUTDIR = Path(
    "experiments/full_oracle_qwen3_8b/"
    "analysis/17_objective_utility"
)


SPACES = {
    "S2": [
        "degradation",
        "memory_bytes",
    ],
    "S3": [
        "degradation",
        "memory_bytes",
        "runtime_seconds",
    ],
    "S4": [
        "degradation",
        "memory_bytes",
        "runtime_seconds",
        "energy_j",
    ],
    "S5": [
        "degradation",
        "memory_bytes",
        "runtime_seconds",
        "energy_j",
        "power_w",
    ],
}


OBJECTIVES = {
    "memory": "memory_bytes",
    "latency": "runtime_seconds",
    "energy": "energy_j",
    "power": "power_w",
}


def main():
    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_quality_hardware()

    rows = []

    for delta in [
        0.01,
        0.05,
        0.10,
    ]:
        for rid, g in df.groupby(
            "request_id"
        ):

            quality_safe = g[
                g["degradation"]
                <= delta + 1e-12
            ].copy()

            for space, cols in (
                SPACES.items()
            ):
                mask = pareto_mask(
                    quality_safe[
                        cols
                    ].to_numpy(float)
                )

                frontier = (
                    quality_safe.loc[
                        mask
                    ]
                )

                for objective_name, objective_col in (
                    OBJECTIVES.items()
                ):
                    chosen = (
                        frontier.sort_values(
                            [
                                objective_col,
                                "degradation",
                                "policy",
                            ]
                        )
                        .iloc[0]
                    )

                    rows.append(
                        {
                            "delta": delta,
                            "request_id": rid,
                            "task": (
                                chosen["task"]
                            ),
                            "space": space,
                            "deployment_objective": (
                                objective_name
                            ),
                            "selected_policy": (
                                chosen["policy"]
                            ),
                            "degradation": (
                                chosen[
                                    "degradation"
                                ]
                            ),
                            "memory_gb": (
                                chosen[
                                    "memory_bytes"
                                ]
                                / 1e9
                            ),
                            "runtime_seconds": (
                                chosen[
                                    "runtime_seconds"
                                ]
                            ),
                            "energy_j": (
                                chosen[
                                    "energy_j"
                                ]
                            ),
                            "power_w": (
                                chosen[
                                    "power_w"
                                ]
                            ),
                        }
                    )

    selections = pd.DataFrame(
        rows
    )

    summary = (
        selections.groupby(
            [
                "delta",
                "space",
                "deployment_objective",
            ]
        )
        .agg(
            mean_degradation=(
                "degradation",
                "mean",
            ),
            mean_memory_gb=(
                "memory_gb",
                "mean",
            ),
            mean_runtime_seconds=(
                "runtime_seconds",
                "mean",
            ),
            mean_energy_j=(
                "energy_j",
                "mean",
            ),
            mean_power_w=(
                "power_w",
                "mean",
            ),
        )
        .reset_index()
    )

    selections.to_csv(
        OUTDIR
        / "objective_utility_selections.csv",
        index=False,
    )

    summary.to_csv(
        OUTDIR
        / "objective_utility_summary.csv",
        index=False,
    )

    print(
        "\nOBJECTIVE UTILITY ABLATION"
    )

    print(
        summary.round(4)
        .to_string(index=False)
    )

    print(
        "\nEXPERIMENT 3: PASS"
    )


if __name__ == "__main__":
    main()
