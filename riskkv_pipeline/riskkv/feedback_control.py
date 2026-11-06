"""
Shared closed-loop feedback utilities for RiskKV.

============================================================
ARCHITECTURE
============================================================

The existing RiskKV controller is essentially open-loop:

    offline profile
          +
    request risk prediction
          |
          v
    configuration selection
          |
          v
       execution

The feedback extension changes this to:

    offline profile
          |
          v
    predicted runtime cost
          |
          v
    configuration selection
          |
          v
       execution
          |
          v
    actual telemetry
          |
          v
      error signal
          |
          +-----------> correction for NEXT request


IMPORTANT:
Feedback is applied at REQUEST BOUNDARIES.

We do NOT switch KV mechanisms halfway through one generation.
This avoids introducing cache-conversion latency and correctness
questions into the first feedback-controller evaluation.

The hardware experiments use the five measured A100 repetitions as
the real underlying execution samples. Controlled multiplicative drift
is then introduced to test whether a controller can compensate when
offline profiles become stale.

The drift experiment is therefore a controlled replay experiment,
NOT a claim that the synthetic drift itself came from production.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from riskkv.synthesis_eval import (
    load_quality_hardware,
)


RAW_HARDWARE = (
    "experiments/full_oracle_qwen3_8b/"
    "hardware/a100_profile/raw_runs.csv"
)


# ============================================================
# PID CONTROLLER
# ============================================================

@dataclass
class PIDController:
    """
    Lightweight dimensionless feedback controller.

    We feed it normalized prediction error:

        e_t = actual / predicted - 1

    Example:
        predicted latency = 4.0 s
        actual latency    = 4.4 s

        e_t = 4.4 / 4.0 - 1 = +0.10

    Positive error means the offline estimate was optimistic.

    The controller returns a correction u_t, and RiskKV subsequently
    uses:

        corrected_prediction =
            offline_prediction * (1 + u_t)

    Output clipping prevents unstable corrections.

    For the quality-feedback experiment the same controller is reused,
    but its output is added to the probability threshold instead.
    """

    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0

    output_min: float = -0.25
    output_max: float = 1.00

    integral_limit: float = 5.0

    integral: float = 0.0
    previous_error: float | None = None
    value: float = 0.0

    def reset(self):
        self.integral = 0.0
        self.previous_error = None
        self.value = 0.0

    def update(self, error: float) -> float:
        """
        Apply one P/PI/PID update.

        P  = immediate response to current error
        I  = persistent accumulated bias
        D  = response to rate of change

        Derivative feedback can be noisy in real systems, which is why
        P, PI, and PID are explicitly compared rather than assuming PID
        must be best.
        """

        error = float(error)

        self.integral += error

        self.integral = float(
            np.clip(
                self.integral,
                -self.integral_limit,
                self.integral_limit,
            )
        )

        if self.previous_error is None:
            derivative = 0.0
        else:
            derivative = (
                error
                - self.previous_error
            )

        output = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        self.value = float(
            np.clip(
                output,
                self.output_min,
                self.output_max,
            )
        )

        self.previous_error = error

        return self.value


# ============================================================
# CONTROLLER SEARCH SPACE
# ============================================================

def controller_grid(kind: str):
    """
    Pre-specified coarse gain grids.

    We do NOT tune gains on the final test trace.

    Experiment scripts tune these gains on a calibration trace and
    evaluate the selected gains on a separate replay trace.
    """

    kind = kind.lower()

    if kind == "fixed":
        return [
            {
                "kp": 0.0,
                "ki": 0.0,
                "kd": 0.0,
            }
        ]

    if kind == "p":
        return [
            {
                "kp": kp,
                "ki": 0.0,
                "kd": 0.0,
            }
            for kp in [
                0.25,
                0.50,
                0.75,
                1.00,
            ]
        ]

    if kind == "pi":
        return [
            {
                "kp": kp,
                "ki": ki,
                "kd": 0.0,
            }
            for kp, ki in product(
                [
                    0.25,
                    0.50,
                    0.75,
                ],
                [
                    0.01,
                    0.03,
                    0.05,
                    0.10,
                ],
            )
        ]

    if kind == "pid":
        return [
            {
                "kp": kp,
                "ki": ki,
                "kd": kd,
            }
            for kp, ki, kd in product(
                [
                    0.25,
                    0.50,
                    0.75,
                ],
                [
                    0.01,
                    0.03,
                    0.05,
                ],
                [
                    0.05,
                    0.10,
                    0.20,
                ],
            )
        ]

    raise ValueError(
        f"Unknown controller kind: {kind}"
    )


# ============================================================
# HARDWARE DATA
# ============================================================

def load_raw_hardware(
    path=RAW_HARDWARE,
):
    """
    Load the 1,000 repeated hardware executions.

    Every request-policy pair has five measured repetitions.

    We keep each repetition separately because the feedback experiments
    replay actual measured runtime/power/energy samples instead of
    creating Gaussian pseudo-measurements.
    """

    raw = pd.read_csv(path)

    raw = raw.copy()

    raw["memory_bytes"] = (
        raw[
            "incremental_peak_allocated_gpu_bytes"
        ]
    )

    raw["runtime_seconds"] = (
        raw["actual_seconds"]
    )

    raw["energy_j"] = (
        raw["gpu_energy_j"]
    )

    raw["power_w"] = (
        raw["gpu_mean_power_w"]
    )

    # If no explicit repetition number exists, preserve original
    # execution ordering and create one.
    raw["_rep_idx"] = (
        raw.groupby(
            [
                "request_id",
                "policy",
            ],
            sort=False,
        )
        .cumcount()
    )

    return raw


def measured_execution(
    raw: pd.DataFrame,
    request_id,
    policy,
    step: int,
):
    """
    Return one REAL measured execution for a request-policy pair.

    The five repetitions are replayed cyclically:

        step 0 -> repetition 0
        step 1 -> repetition 1
        ...
        step 5 -> repetition 0 again

    This retains the empirically observed measurement variability.
    """

    g = raw[
        (
            raw["request_id"]
            == request_id
        )
        & (
            raw["policy"]
            == policy
        )
    ]

    if len(g) == 0:
        raise RuntimeError(
            f"No hardware measurement for "
            f"{request_id=} {policy=}"
        )

    idx = int(
        step % len(g)
    )

    return g.iloc[idx]


# ============================================================
# CONTROLLED DRIFT TRACE
# ============================================================

def make_drift_cycle(
    medium=1.10,
    high=1.20,
):
    """
    100-request controlled drift cycle.

      requests  0-19 : normal
      requests 20-44 : moderate drift
      requests 45-69 : high drift
      requests 70-99 : recovery

    This gives the controller both:
      - an upward disturbance;
      - a recovery period.

    It therefore tests both reaction and de-escalation.
    """

    return np.asarray(
        (
            [1.00] * 20
            + [medium] * 25
            + [high] * 25
            + [1.00] * 30
        ),
        dtype=float,
    )


def phase_from_factor(
    factor: float,
):
    if np.isclose(
        factor,
        1.0,
    ):
        return "normal"

    if factor < 1.15:
        return "moderate_drift"

    return "high_drift"


# ============================================================
# GENERIC RESOURCE-FEEDBACK REPLAY
# ============================================================

def run_resource_cycle(
    profile: pd.DataFrame,
    raw: pd.DataFrame,
    request_sequence,
    drift_factors,
    resource_col: str,
    cap_multiplier: float,
    delta: float,
    controller: PIDController | None,
    objective_col: str = "memory_bytes",
    drift_compressed_only: bool = True,
):
    """
    Generic request-level resource feedback experiment.

    ------------------------------------------------------------
    HOW THE SECTION WORKS
    ------------------------------------------------------------

    1. For the current request, obtain all eight KV configurations.

    2. Use TRUE oracle degradation only to isolate the resource-loop
       experiment:

           D(r,p) <= delta

       This means latency/power feedback is tested independently of
       risk-predictor error.

    3. Start from the offline median hardware profile.

    4. Apply the current controller correction to predicted resource
       cost.

    5. Select the lowest-memory policy that is predicted to satisfy the
       resource cap.

    6. Replay an ACTUAL measured A100 repetition and apply controlled
       drift to the realized resource cost.

    7. Observe prediction error and update P/PI/PID.

    8. The correction influences the NEXT request only.

    FullKV remains available as fallback.
    """

    if len(request_sequence) != len(
        drift_factors
    ):
        raise ValueError(
            "request_sequence and drift_factors "
            "must have equal length"
        )

    rows = []

    for step, (
        request_id,
        drift,
    ) in enumerate(
        zip(
            request_sequence,
            drift_factors,
        )
    ):

        g = profile[
            profile[
                "request_id"
            ]
            == request_id
        ].copy()

        if len(g) == 0:
            raise RuntimeError(
                f"Unknown request: {request_id}"
            )

        full = g[
            g["policy"]
            == "full"
        ].iloc[0]

        # --------------------------------------------------------
        # Runtime/SLA cap is expressed relative to that request's
        # measured FullKV baseline.
        #
        # Sweeping cap_multiplier avoids relying on one arbitrary SLA.
        # --------------------------------------------------------

        cap = (
            float(
                full[
                    resource_col
                ]
            )
            * float(
                cap_multiplier
            )
        )

        safe = g[
            g[
                "degradation"
            ]
            <= delta + 1e-12
        ].copy()

        correction = (
            0.0
            if controller is None
            else controller.value
        )

        # --------------------------------------------------------
        # FullKV is not subject to the synthetic compression-specific
        # drift in these first experiments.
        #
        # The disturbance represents stale estimates of compression
        # kernel/cache-management overhead.
        # --------------------------------------------------------

        safe[
            "feedback_multiplier"
        ] = np.where(
            safe[
                "policy"
            ].eq(
                "full"
            ),
            1.0,
            1.0 + correction,
        )

        safe[
            "predicted_resource"
        ] = (
            safe[
                resource_col
            ]
            * safe[
                "feedback_multiplier"
            ]
        )

        feasible = safe[
            safe[
                "predicted_resource"
            ]
            <= cap + 1e-12
        ]

        if len(feasible):
            chosen = (
                feasible.sort_values(
                    [
                        objective_col,
                        "degradation",
                        resource_col,
                        "policy",
                    ]
                )
                .iloc[0]
            )
        else:
            # A systems SLA may occasionally become infeasible.
            # FullKV is retained as the semantic-quality fallback.
            chosen = full.copy()

        actual_sample = measured_execution(
            raw,
            request_id,
            chosen["policy"],
            step,
        )

        base_actual = float(
            actual_sample[
                resource_col
            ]
        )

        if (
            drift_compressed_only
            and chosen[
                "policy"
            ]
            == "full"
        ):
            actual_resource = (
                base_actual
            )
        else:
            actual_resource = (
                base_actual
                * float(
                    drift
                )
            )

        predicted_before_execution = float(
            chosen[
                resource_col
            ]
        )

        if (
            chosen[
                "policy"
            ]
            != "full"
        ):
            predicted_before_execution *= (
                1.0
                + correction
            )

        violation = (
            actual_resource
            > cap + 1e-12
        )

        overrun = max(
            0.0,
            actual_resource
            - cap,
        )

        memory_saving = (
            1.0
            - float(
                chosen[
                    "memory_bytes"
                ]
            )
            / float(
                full[
                    "memory_bytes"
                ]
            )
        )

        # --------------------------------------------------------
        # CLOSED-LOOP ERROR UPDATE
        # --------------------------------------------------------

        error = np.nan

        if (
            controller is not None
            and chosen[
                "policy"
            ]
            != "full"
        ):

            baseline = float(
                chosen[
                    resource_col
                ]
            )

            if baseline > 0:
                error = (
                    actual_resource
                    / baseline
                    - 1.0
                )

                controller.update(
                    error
                )

        rows.append(
            {
                "step": step,
                "request_id": request_id,
                "task": chosen["task"],
                "phase": phase_from_factor(
                    drift
                ),
                "drift_factor": drift,
                "policy": chosen["policy"],
                "delta": delta,
                "resource": resource_col,
                "cap": cap,
                "predicted_resource": (
                    predicted_before_execution
                ),
                "actual_resource": (
                    actual_resource
                ),
                "resource_violation": (
                    violation
                ),
                "overrun": overrun,
                "memory_saving": (
                    memory_saving
                ),
                "degradation": chosen[
                    "degradation"
                ],
                "controller_error": error,
                "controller_correction": (
                    0.0
                    if controller is None
                    else controller.value
                ),
                "fullkv": (
                    chosen["policy"]
                    == "full"
                ),
            }
        )

    return pd.DataFrame(rows)


def summarize_resource_trace(
    x: pd.DataFrame,
):
    """
    Common resource-control outcome metrics.

    Lower is better:
        violation_rate
        mean_overrun
        fullkv_rate

    Higher is usually better:
        mean_memory_saving

    controller correction itself is diagnostic, not an objective.
    """

    return {
        "requests": len(x),

        "violation_rate": (
            x[
                "resource_violation"
            ]
            .mean()
        ),

        "mean_overrun": (
            x["overrun"].mean()
        ),

        "p95_overrun": (
            x[
                "overrun"
            ]
            .quantile(
                0.95
            )
        ),

        "mean_memory_saving": (
            x[
                "memory_saving"
            ]
            .mean()
        ),

        "fullkv_rate": (
            x[
                "fullkv"
            ]
            .mean()
        ),

        "mean_abs_correction": (
            x[
                "controller_correction"
            ]
            .abs()
            .mean()
        ),
    }


def choose_best_gain(
    result_rows,
):
    """
    Gain selection is lexicographic.

    Primary goal:
        minimize resource/SLA violation rate.

    Secondary goal:
        maximize memory saving.

    Tertiary:
        minimize overrun.

    No final-test outcomes are used here.
    """

    result = pd.DataFrame(
        result_rows
    )

    result = result.sort_values(
        [
            "violation_rate",
            "mean_memory_saving",
            "mean_overrun",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )

    return result.iloc[0]
