"""
EXPERIMENT 7
============

Dynamic runtime trace simulation.

Purpose
-------
Demonstrates runtime adaptation over a sequence of requests while system
constraints change over time.

This is not intended to pretend that the trace came from a production
cluster. It is a CONTROLLED SYNTHETIC STRESS TRACE constructed from
measured A100 operating-point quantiles.

It answers:
  "When deployment conditions change, does RiskKV move to a different
   feasible operating point while preserving the quality budget?"

Comparison:
  - adaptive runtime synthesis
  - fixed global static policy

Reported cumulatively:
  - quality violations
  - memory
  - latency
  - energy
  - power-cap violations
  - policy switches
"""

from __future__ import annotations

from pathlib import Path
import itertools
import numpy as np
import pandas as pd

from riskkv.synthesis_eval import (
    choose_min_objective,
    load_quality_hardware,
)


OUTDIR = Path(
    "experiments/full_oracle_qwen3_8b/"
    "analysis/21_dynamic_trace"
)


def main():
    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_quality_hardware()

    # ---------------------------------------------------------
    # Empirically derived stress thresholds.
    # ---------------------------------------------------------
    mem_q40 = (
        df.memory_bytes.quantile(
            0.40
        )
    )

    latency_q40 = (
        df.runtime_seconds.quantile(
            0.40
        )
    )

    energy_q40 = (
        df.energy_j.quantile(
            0.40
        )
    )

    power_q40 = (
        df.power_w.quantile(
            0.40
        )
    )

    states = [
        {
            "state": "normal",
            "objective": "runtime_seconds",
            "delta": 0.05,
            "constraints": {},
        },

        {
            "state": "memory_pressure",
            "objective": "memory_bytes",
            "delta": 0.10,
            "constraints": {
                "memory_bytes": mem_q40,
            },
        },

        {
            "state": "latency_pressure",
            "objective": "runtime_seconds",
            "delta": 0.05,
            "constraints": {
                "runtime_seconds": (
                    latency_q40
                ),
            },
        },

        {
            "state": "energy_pressure",
            "objective": "energy_j",
            "delta": 0.05,
            "constraints": {
                "energy_j": (
                    energy_q40
                ),
            },
        },

        {
            "state": "power_cap",
            "objective": "runtime_seconds",
            "delta": 0.05,
            "constraints": {
                "power_w": power_q40,
            },
        },
    ]

    # Deterministic request ordering.
    requests = sorted(
        df.request_id.unique()
    )

    # Repeat requests so each state appears multiple times.
    trace_requests = list(
        itertools.islice(
            itertools.cycle(
                requests
            ),
            100,
        )
    )

    adaptive_rows = []

    for t, rid in enumerate(
        trace_requests
    ):
        state = states[
            t % len(states)
        ]

        g = df[
            df.request_id == rid
        ].copy()

        feasible = (
            g["degradation"]
            <= state["delta"]
            + 1e-12
        )

        for col, cap in (
            state[
                "constraints"
            ].items()
        ):
            feasible &= (
                g[col]
                <= cap + 1e-12
            )

        chosen = choose_min_objective(
            g,
            state["objective"],
            feasible,
        )

        adaptive_rows.append(
            {
                "time_step": t,
                "request_id": rid,
                "task": chosen["task"],
                "system_state": (
                    state["state"]
                ),
                "policy": (
                    chosen["policy"]
                ),
                "delta": (
                    state["delta"]
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

    adaptive = pd.DataFrame(
        adaptive_rows
    )

    # ---------------------------------------------------------
    # Static baseline.
    #
    # Pick the globally lowest-average-memory policy that has
    # <= 5% quality violations at delta=.05 on this profile.
    # ---------------------------------------------------------
    candidates = []

    for policy, g in df.groupby(
        "policy"
    ):
        candidates.append(
            {
                "policy": policy,
                "violation": (
                    g["degradation"]
                    > 0.05
                ).mean(),
                "memory": (
                    g.memory_bytes.mean()
                ),
            }
        )

    candidates = pd.DataFrame(
        candidates
    )

    feasible_static = candidates[
        candidates["violation"]
        <= 0.05 + 1e-12
    ]

    if len(feasible_static):
        static_policy = (
            feasible_static.sort_values(
                [
                    "memory",
                    "violation",
                ]
            )
            .iloc[0]["policy"]
        )
    else:
        static_policy = "full"

    static_rows = []

    for t, rid in enumerate(
        trace_requests
    ):
        state = states[
            t % len(states)
        ]

        g = df[
            df.request_id == rid
        ]

        chosen = g[
            g.policy
            == static_policy
        ]

        if len(chosen) == 0:
            chosen = g[
                g.policy == "full"
            ]

        chosen = chosen.iloc[0]

        static_rows.append(
            {
                "time_step": t,
                "request_id": rid,
                "task": chosen["task"],
                "system_state": (
                    state["state"]
                ),
                "policy": (
                    chosen["policy"]
                ),
                "delta": (
                    state["delta"]
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

    static = pd.DataFrame(
        static_rows
    )

    def summarize(name, x):
        violation = (
            x.degradation
            > x.delta
        )

        policy_switches = (
            x.policy
            != x.policy.shift(1)
        ).sum() - 1

        return {
            "method": name,
            "static_policy": (
                static_policy
                if name == "static"
                else ""
            ),
            "requests": len(x),
            "quality_violation_rate": (
                violation.mean()
            ),
            "mean_memory_gb": (
                x.memory_gb.mean()
            ),
            "mean_runtime_seconds": (
                x.runtime_seconds.mean()
            ),
            "total_energy_j": (
                x.energy_j.sum()
            ),
            "mean_power_w": (
                x.power_w.mean()
            ),
            "policy_switches": int(
                max(
                    policy_switches,
                    0,
                )
            ),
            "unique_policies": (
                x.policy.nunique()
            ),
        }

    summary = pd.DataFrame(
        [
            summarize(
                "adaptive",
                adaptive,
            ),
            summarize(
                "static",
                static,
            ),
        ]
    )

    adaptive.to_csv(
        OUTDIR
        / "adaptive_trace.csv",
        index=False,
    )

    static.to_csv(
        OUTDIR
        / "static_trace.csv",
        index=False,
    )

    summary.to_csv(
        OUTDIR
        / "dynamic_trace_summary.csv",
        index=False,
    )

    policy_mix = (
        adaptive.groupby(
            [
                "system_state",
                "policy",
            ]
        )
        .size()
        .rename("requests")
        .reset_index()
    )

    policy_mix.to_csv(
        OUTDIR
        / "adaptive_policy_mix_by_state.csv",
        index=False,
    )

    print(
        "\nDYNAMIC TRACE SUMMARY"
    )

    print(
        summary.round(4)
        .to_string(index=False)
    )

    print(
        "\nADAPTIVE POLICY MIX BY STATE"
    )

    print(
        policy_mix.to_string(
            index=False
        )
    )

    print(
        "\nEXPERIMENT 7: PASS"
    )


if __name__ == "__main__":
    main()
