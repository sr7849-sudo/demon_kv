"""
EXPERIMENT 8B
=============

Energy- and power-profile drift.

Purpose
-------
Tests whether offline A100 energy/power profiles can be corrected when
actual runtime behavior deviates from the original characterization.

Two independent feedback channels are evaluated:

    ENERGY:
        actual GPU joules vs offline predicted joules

    POWER:
        actual mean GPU watts vs offline predicted watts

This is important because power and energy are not interchangeable:

    E = integral P(t) dt

A configuration can draw lower average power but consume more total
energy because it executes longer.
"""

from pathlib import Path

import pandas as pd

from riskkv.feedback_control import (
    PIDController,
    choose_best_gain,
    controller_grid,
    load_raw_hardware,
    make_drift_cycle,
    run_resource_cycle,
    summarize_resource_trace,
)

from riskkv.synthesis_eval import (
    load_quality_hardware,
)


OUTDIR = Path(
    "experiments/full_oracle_qwen3_8b/"
    "analysis/23_energy_power_feedback"
)


DELTA = 0.05

RESOURCES = {
    "energy": {
        "column": "energy_j",

        # Cap relative to FullKV energy.
        "cap_multipliers": [
            1.00,
            1.05,
            1.10,
        ],
    },

    "power": {
        "column": "power_w",

        # FullKV remains feasible at >=1.0.
        "cap_multipliers": [
            1.00,
            1.05,
            1.10,
        ],
    },
}


CONTROLLERS = [
    "fixed",
    "p",
    "pi",
    "pid",
]


def main():

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile = (
        load_quality_hardware()
    )

    raw = load_raw_hardware()

    request_ids = sorted(
        profile[
            "request_id"
        ].unique()
    )

    calibration_requests = (
        request_ids * 4
    )[:100]

    test_requests = (
        list(
            reversed(
                request_ids
            )
        )
        * 4
    )[:100]

    drift = make_drift_cycle(
        medium=1.10,
        high=1.20,
    )

    summaries = []
    gain_rows = []
    traces = []

    for resource_name, cfg in (
        RESOURCES.items()
    ):

        resource_col = cfg[
            "column"
        ]

        for cap_multiplier in (
            cfg[
                "cap_multipliers"
            ]
        ):

            for kind in (
                CONTROLLERS
            ):

                candidates = []

                # -------------------------------------------------
                # CALIBRATION-ONLY GAIN SELECTION
                # -------------------------------------------------

                for gains in (
                    controller_grid(
                        kind
                    )
                ):

                    ctrl = (
                        None
                        if kind
                        == "fixed"
                        else PIDController(
                            **gains
                        )
                    )

                    x = run_resource_cycle(
                        profile=profile,
                        raw=raw,
                        request_sequence=(
                            calibration_requests
                        ),
                        drift_factors=drift,
                        resource_col=(
                            resource_col
                        ),
                        cap_multiplier=(
                            cap_multiplier
                        ),
                        delta=DELTA,
                        controller=ctrl,
                    )

                    m = (
                        summarize_resource_trace(
                            x
                        )
                    )

                    m.update(
                        {
                            "resource": (
                                resource_name
                            ),
                            "controller": (
                                kind
                            ),
                            "cap_multiplier": (
                                cap_multiplier
                            ),
                            **gains,
                        }
                    )

                    candidates.append(
                        m
                    )

                best = choose_best_gain(
                    candidates
                )

                gain_rows.append(
                    best.to_dict()
                )

                selected_gains = {
                    "kp": float(
                        best["kp"]
                    ),
                    "ki": float(
                        best["ki"]
                    ),
                    "kd": float(
                        best["kd"]
                    ),
                }

                ctrl = (
                    None
                    if kind
                    == "fixed"
                    else PIDController(
                        **selected_gains
                    )
                )

                # -------------------------------------------------
                # HELD-OUT REPLAY
                # -------------------------------------------------

                x = run_resource_cycle(
                    profile=profile,
                    raw=raw,
                    request_sequence=(
                        test_requests
                    ),
                    drift_factors=drift,
                    resource_col=(
                        resource_col
                    ),
                    cap_multiplier=(
                        cap_multiplier
                    ),
                    delta=DELTA,
                    controller=ctrl,
                )

                x["controller"] = (
                    kind
                )

                x["resource_name"] = (
                    resource_name
                )

                x["cap_multiplier"] = (
                    cap_multiplier
                )

                traces.append(x)

                m = (
                    summarize_resource_trace(
                        x
                    )
                )

                m.update(
                    {
                        "resource": (
                            resource_name
                        ),
                        "controller": (
                            kind
                        ),
                        "cap_multiplier": (
                            cap_multiplier
                        ),
                        **selected_gains,
                    }
                )

                summaries.append(
                    m
                )

    summary = pd.DataFrame(
        summaries
    )

    gains = pd.DataFrame(
        gain_rows
    )

    trace = pd.concat(
        traces,
        ignore_index=True,
    )

    summary.to_csv(
        OUTDIR
        / "energy_power_feedback_summary.csv",
        index=False,
    )

    gains.to_csv(
        OUTDIR
        / "selected_controller_gains.csv",
        index=False,
    )

    trace.to_csv(
        OUTDIR
        / "energy_power_feedback_trace.csv",
        index=False,
    )

    print(
        "\nENERGY / POWER FEEDBACK"
    )

    print(
        summary.round(5)
        .to_string(
            index=False
        )
    )

    print(
        "\nEXPERIMENT 8B: PASS"
    )


if __name__ == "__main__":
    main()
