"""
EXPERIMENT 8A
=============

Closed-loop latency-drift control.

Research question
-----------------
If offline latency profiles become stale at runtime, can feedback
reduce SLA violations without giving up all compression opportunity?

Architecture
------------
Calibration trace:
    used ONLY to select P/PI/PID gains.

Test trace:
    fresh request ordering;
    same pre-specified disturbance structure;
    no gain tuning.

Controllers:
    fixed offline profile
    P
    PI
    PID

Quality feasibility is oracle-known in THIS experiment so that we
isolate the hardware-feedback loop. Learned quality feedback is tested
separately in 08c.
"""

from pathlib import Path

import numpy as np
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
    "analysis/22_latency_feedback"
)


DELTA = 0.05

# SLA sensitivity:
# 1.00 = no slowdown allowed relative to FullKV
# 1.05 = up to 5% slower
# 1.10 = up to 10% slower
CAP_MULTIPLIERS = [
    1.00,
    1.05,
    1.10,
]

CONTROLLERS = [
    "fixed",
    "p",
    "pi",
    "pid",
]


def make_sequences(
    requests,
):
    """
    Use two deterministic but different request orders.

    This ensures gain selection and final evaluation are not performed
    on the exact same request-to-drift mapping.
    """

    calibration = (
        requests * 4
    )[:100]

    shifted = (
        list(
            reversed(
                requests
            )
        )
        * 4
    )[:100]

    return (
        calibration,
        shifted,
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

    requests = sorted(
        profile[
            "request_id"
        ].unique()
    )

    calibration_requests, test_requests = (
        make_sequences(
            requests
        )
    )

    drift = make_drift_cycle(
        medium=1.10,
        high=1.20,
    )

    gain_rows = []
    test_rows = []
    traces = []

    for cap_multiplier in (
        CAP_MULTIPLIERS
    ):

        for kind in CONTROLLERS:

            calibration_results = []

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

                trace = run_resource_cycle(
                    profile=profile,
                    raw=raw,
                    request_sequence=(
                        calibration_requests
                    ),
                    drift_factors=drift,
                    resource_col=(
                        "runtime_seconds"
                    ),
                    cap_multiplier=(
                        cap_multiplier
                    ),
                    delta=DELTA,
                    controller=ctrl,
                )

                m = (
                    summarize_resource_trace(
                        trace
                    )
                )

                m.update(
                    {
                        "controller": kind,
                        "cap_multiplier": (
                            cap_multiplier
                        ),
                        **gains,
                    }
                )

                calibration_results.append(
                    m
                )

            best = choose_best_gain(
                calibration_results
            )

            gain_rows.append(
                best.to_dict()
            )

            # -----------------------------------------------------
            # FINAL TEST
            #
            # Reset the controller. Calibration state is NOT carried
            # into the evaluation trace.
            # -----------------------------------------------------

            gains = {
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
                if kind == "fixed"
                else PIDController(
                    **gains
                )
            )

            test = run_resource_cycle(
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
                    cap_multiplier
                ),
                delta=DELTA,
                controller=ctrl,
            )

            test[
                "controller"
            ] = kind

            test[
                "cap_multiplier"
            ] = cap_multiplier

            traces.append(
                test
            )

            m = (
                summarize_resource_trace(
                    test
                )
            )

            m.update(
                {
                    "controller": kind,
                    "cap_multiplier": (
                        cap_multiplier
                    ),
                    **gains,
                }
            )

            test_rows.append(
                m
            )

    gains = pd.DataFrame(
        gain_rows
    )

    summary = pd.DataFrame(
        test_rows
    )

    traces = pd.concat(
        traces,
        ignore_index=True,
    )

    phase_summary = (
        traces.groupby(
            [
                "cap_multiplier",
                "controller",
                "phase",
            ],
            as_index=False,
        )
        .agg(
            violation_rate=(
                "resource_violation",
                "mean",
            ),
            mean_memory_saving=(
                "memory_saving",
                "mean",
            ),
            mean_actual_latency=(
                "actual_resource",
                "mean",
            ),
            fullkv_rate=(
                "fullkv",
                "mean",
            ),
        )
    )

    gains.to_csv(
        OUTDIR
        / "selected_controller_gains.csv",
        index=False,
    )

    summary.to_csv(
        OUTDIR
        / "latency_feedback_summary.csv",
        index=False,
    )

    traces.to_csv(
        OUTDIR
        / "latency_feedback_trace.csv",
        index=False,
    )

    phase_summary.to_csv(
        OUTDIR
        / "latency_feedback_by_phase.csv",
        index=False,
    )

    print(
        "\nSELECTED GAINS"
    )

    print(
        gains.round(5)
        .to_string(
            index=False
        )
    )

    print(
        "\nLATENCY FEEDBACK TEST"
    )

    print(
        summary.round(5)
        .to_string(
            index=False
        )
    )

    print(
        "\nPHASE BREAKDOWN"
    )

    print(
        phase_summary.round(5)
        .to_string(
            index=False
        )
    )

    print(
        "\nEXPERIMENT 8A: PASS"
    )


if __name__ == "__main__":
    main()
