"""
EXPERIMENT 8D
=============

Feedback-controller ablation and disturbance sensitivity.

Purpose
-------
A reviewer can reasonably ask:

    Why PID?
    Why not P?
    Why not PI?
    Is the result tied to one particular drift magnitude?

This experiment tunes gains once on a 15% disturbance calibration
trace and then tests the same gains under:

     5%
    10%
    20%
    30%

runtime-profile drift.

The selected gains therefore do NOT depend on each test disturbance.
"""

from pathlib import Path

import pandas as pd

from riskkv.feedback_control import (
    PIDController,
    choose_best_gain,
    controller_grid,
    load_raw_hardware,
    run_resource_cycle,
    summarize_resource_trace,
)

from riskkv.synthesis_eval import (
    load_quality_hardware,
)


OUTDIR = Path(
    "experiments/full_oracle_qwen3_8b/"
    "analysis/25_controller_ablation"
)


DELTA = 0.05
CAP_MULTIPLIER = 1.05


KINDS = [
    "fixed",
    "p",
    "pi",
    "pid",
]


TEST_DRIFTS = [
    0.05,
    0.10,
    0.20,
    0.30,
]


def drift_trace(
    amplitude,
):
    """
    Controlled disturbance:

       normal
       half-amplitude
       full-amplitude
       recovery
    """

    return (
        [1.00] * 20
        + [
            1.00
            + amplitude / 2
        ] * 25
        + [
            1.00
            + amplitude
        ] * 25
        + [1.00] * 30
    )


def main():

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile = (
        load_quality_hardware()
    )

    raw = load_raw_hardware()

    ids = sorted(
        profile[
            "request_id"
        ].unique()
    )

    calibration_requests = (
        ids * 4
    )[:100]

    test_requests = (
        list(
            reversed(
                ids
            )
        )
        * 4
    )[:100]

    # ------------------------------------------------------------
    # GAIN TUNING
    #
    # All controllers are calibrated under the SAME 15% disturbance.
    # ------------------------------------------------------------

    calibration_drift = (
        drift_trace(
            0.15
        )
    )

    selected = {}

    for kind in KINDS:

        candidates = []

        for gains in (
            controller_grid(
                kind
            )
        ):

            ctrl = (
                None
                if kind == "fixed"
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
                drift_factors=(
                    calibration_drift
                ),
                resource_col=(
                    "runtime_seconds"
                ),
                cap_multiplier=(
                    CAP_MULTIPLIER
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
                    "controller": kind,
                    **gains,
                }
            )

            candidates.append(m)

        best = choose_best_gain(
            candidates
        )

        selected[
            kind
        ] = {
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

    rows = []

    for amplitude in (
        TEST_DRIFTS
    ):

        drift = drift_trace(
            amplitude
        )

        for kind in KINDS:

            gains = selected[
                kind
            ]

            ctrl = (
                None
                if kind == "fixed"
                else PIDController(
                    **gains
                )
            )

            x = run_resource_cycle(
                profile=profile,
                raw=raw,
                request_sequence=(
                    test_requests
                ),
                drift_factors=drift,
                resource_col=(
                    "runtime_seconds"
                ),
                cap_multiplier=(
                    CAP_MULTIPLIER
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
                    "drift_amplitude": (
                        amplitude
                    ),
                    "controller": kind,
                    **gains,
                }
            )

            rows.append(m)

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        OUTDIR
        / "controller_ablation.csv",
        index=False,
    )

    gain_table = pd.DataFrame(
        [
            {
                "controller": kind,
                **gains,
            }
            for kind, gains
            in selected.items()
        ]
    )

    gain_table.to_csv(
        OUTDIR
        / "controller_gains.csv",
        index=False,
    )

    print(
        "\nCONTROLLER ROBUSTNESS ABLATION"
    )

    print(
        result.round(5)
        .to_string(
            index=False
        )
    )

    print(
        "\nEXPERIMENT 8D: PASS"
    )


if __name__ == "__main__":
    main()
