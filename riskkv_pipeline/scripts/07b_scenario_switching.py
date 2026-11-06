"""
EXPERIMENT 2
============

Scenario-conditioned runtime synthesis.

Purpose
-------
Demonstrates that the SAME request legitimately receives different KV
configurations when deployment conditions change.

No weighted objective is used.

Each scenario specifies:
  - one primary objective;
  - explicit constraints.

Thresholds are derived from empirical A100 operating distributions so
that we do not invent arbitrary absolute numbers.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from riskkv.synthesis_eval import (
    choose_min_objective,
    load_quality_hardware,
)


OUTDIR = Path(
    "experiments/full_oracle_qwen3_8b/"
    "analysis/16_scenario_switching"
)


def main():
    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_quality_hardware()

    # ---------------------------------------------------------
    # Empirical system thresholds.
    #
    # These are not claimed to be universal SLA values.
    # They define reproducible stress scenarios from the
    # measured A100 design space itself.
    # ---------------------------------------------------------
    memory_cap = df.memory_bytes.quantile(
        0.50
    )

    latency_cap = df.runtime_seconds.quantile(
        0.50
    )

    energy_cap = df.energy_j.quantile(
        0.50
    )

    power_cap = df.power_w.quantile(
        0.50
    )

    scenarios = {
        "quality_critical": {
            "delta": 0.01,
            "objective": "runtime_seconds",
            "constraints": {},
        },

        "memory_critical": {
            "delta": 0.10,
            "objective": "memory_bytes",
            "constraints": {
                "runtime_seconds": latency_cap,
            },
        },

        "latency_critical": {
            "delta": 0.05,
            "objective": "runtime_seconds",
            "constraints": {
                "memory_bytes": memory_cap,
            },
        },

        "energy_critical": {
            "delta": 0.05,
            "objective": "energy_j",
            "constraints": {
                "runtime_seconds": latency_cap,
            },
        },

        "power_capped": {
            "delta": 0.05,
            "objective": "runtime_seconds",
            "constraints": {
                "power_w": power_cap,
            },
        },
    }

    selections = []

    for scenario, cfg in scenarios.items():

        for rid, g in df.groupby(
            "request_id"
        ):
            feasible = (
                g["degradation"]
                <= cfg["delta"] + 1e-12
            )

            for column, cap in (
                cfg["constraints"].items()
            ):
                feasible &= (
                    g[column]
                    <= cap + 1e-12
                )

            chosen = choose_min_objective(
                g,
                cfg["objective"],
                feasible,
            )

            selections.append(
                {
                    "scenario": scenario,
                    "request_id": rid,
                    "task": chosen["task"],
                    "policy": chosen["policy"],
                    "delta": cfg["delta"],
                    "objective": cfg[
                        "objective"
                    ],
                    "degradation": chosen[
                        "degradation"
                    ],
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
                    "energy_j": chosen[
                        "energy_j"
                    ],
                    "power_w": chosen[
                        "power_w"
                    ],
                    "fullkv_fallback": (
                        chosen["policy"]
                        == "full"
                    ),
                }
            )

    selections = pd.DataFrame(
        selections
    )

    pivot = selections.pivot(
        index="request_id",
        columns="scenario",
        values="policy",
    )

    scenario_names = list(
        scenarios
    )

    transitions = []

    for i, a in enumerate(
        scenario_names
    ):
        for b in scenario_names[
            i + 1:
        ]:
            changed = (
                pivot[a]
                != pivot[b]
            )

            transitions.append(
                {
                    "scenario_a": a,
                    "scenario_b": b,
                    "requests": len(changed),
                    "requests_changed": (
                        int(changed.sum())
                    ),
                    "fraction_changed": (
                        changed.mean()
                    ),
                }
            )

    # How many unique policies does each request use
    # across all deployment scenarios?
    per_request = (
        selections.groupby(
            "request_id"
        )
        .agg(
            unique_policies=(
                "policy",
                "nunique",
            ),
        )
        .reset_index()
    )

    summary = (
        selections.groupby(
            "scenario"
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
            fullkv_rate=(
                "fullkv_fallback",
                "mean",
            ),
        )
        .reset_index()
    )

    selections.to_csv(
        OUTDIR / "scenario_selections.csv",
        index=False,
    )

    pivot.to_csv(
        OUTDIR
        / "same_request_policy_matrix.csv"
    )

    pd.DataFrame(
        transitions
    ).to_csv(
        OUTDIR
        / "scenario_switching_rates.csv",
        index=False,
    )

    per_request.to_csv(
        OUTDIR
        / "policies_per_request.csv",
        index=False,
    )

    summary.to_csv(
        OUTDIR / "scenario_summary.csv",
        index=False,
    )

    print(
        "\nSCENARIO SUMMARY"
    )
    print(
        summary.round(4)
        .to_string(index=False)
    )

    print(
        "\nSAME-REQUEST POLICY SWITCHING"
    )

    print(
        pd.DataFrame(
            transitions
        )
        .round(4)
        .to_string(index=False)
    )

    print(
        "\nUNIQUE POLICIES PER REQUEST"
    )

    print(
        per_request[
            "unique_policies"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nEXPERIMENT 2: PASS"
    )


if __name__ == "__main__":
    main()
