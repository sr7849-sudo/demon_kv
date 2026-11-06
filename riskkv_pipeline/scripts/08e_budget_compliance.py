"""
EXPERIMENT 8E
=============

Closed-loop workload-level energy budgeting.

============================================================
QUESTION
============================================================

Can RiskKV manage a FINITE TOTAL ENERGY BUDGET across a sequence of
requests while maintaining the requested quality-risk constraint?

This is closer to a real budgeting controller than treating every
request independently.

============================================================
CONTROLLER
============================================================

At time t:

    remaining budget
        ----------------
       remaining requests

gives the current per-request energy quota.

RiskKV then:

    1. predicts the quality-safe policy set;
    2. predicts runtime energy;
    3. selects the lowest-memory configuration expected to fit the
       current energy quota;
    4. executes;
    5. subtracts ACTUAL measured energy from remaining budget;
    6. updates the PI correction from prediction error.

The offline hardware profile is therefore an INITIAL PRIOR rather than
an immutable truth.

============================================================
BUDGET DEFINITION
============================================================

We avoid inventing arbitrary absolute joule budgets.

For the controlled replay we compute:

    B_oracle_min

= minimum total energy achievable by a post-hoc quality-safe oracle.

and

    B_full

= total energy of FullKV.

Then budget positions are placed BETWEEN those two empirical bounds:

    25% of gap
    50% of gap
    75% of gap

The oracle is used only to DEFINE experimental stress levels.
It is not available to RiskKV during selection.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from riskkv.feedback_control import (
    PIDController,
    load_raw_hardware,
    measured_execution,
)

from riskkv.synthesis_eval import (
    calibrate_threshold,
    load_quality_hardware,
)


RISK = (
    "experiments/full_oracle_qwen3_8b/"
    "analysis/09_profile_prediction/"
    "policy_risk_oof.csv"
)


OUTDIR = Path(
    "experiments/full_oracle_qwen3_8b/"
    "analysis/26_budget_compliance"
)


DELTA = 0.10
RISK_TARGET = 0.05


BUDGET_POSITIONS = [
    0.25,
    0.50,
    0.75,
]


def drift_trace():
    """
    Runtime energy behavior changes over the 100-request workload.

    Compression-specific energy cost increases temporarily and later
    recovers.
    """

    return np.asarray(
        (
            [1.00] * 25
            + [1.10] * 25
            + [1.20] * 25
            + [1.00] * 25
        ),
        dtype=float,
    )


def actual_energy(
    raw,
    request_id,
    policy,
    step,
    drift,
):
    sample = measured_execution(
        raw,
        request_id,
        policy,
        step,
    )

    e = float(
        sample[
            "energy_j"
        ]
    )

    # Controlled drift is applied to compressed configurations.
    if policy != "full":
        e *= float(drift)

    return e


def build_runtime_table(
    profile,
    risk,
):
    """
    Attach OOF safety probabilities to the 25 hardware-profile requests.

    FullKV has no risk-model row because it is the quality reference.
    It is therefore explicitly added as probability 1.
    """

    compressed = profile[
        profile[
            "policy"
        ]
        != "full"
    ].merge(
        risk[
            [
                "request_id",
                "policy",
                "prob_safe_mean",
                "risk_uncertainty",
            ]
        ],
        on=[
            "request_id",
            "policy",
        ],
        how="left",
        validate="one_to_one",
    )

    if (
        compressed[
            "prob_safe_mean"
        ]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Missing risk predictions for "
            "hardware-profile requests."
        )

    full = profile[
        profile[
            "policy"
        ]
        == "full"
    ].copy()

    full[
        "prob_safe_mean"
    ] = 1.0

    full[
        "risk_uncertainty"
    ] = 0.0

    return pd.concat(
        [
            compressed,
            full,
        ],
        ignore_index=True,
    )


def oracle_budget_bounds(
    runtime,
    raw,
    request_sequence,
    drift,
):
    """
    Compute empirical lower/upper energy bounds for stress-level design.

    LOWER:
        post-hoc minimum actual-energy configuration satisfying
        true quality budget.

    UPPER:
        actual FullKV energy.

    Neither future quantity is visible to the runtime controller.
    """

    oracle_total = 0.0
    full_total = 0.0

    for step, (
        rid,
        factor,
    ) in enumerate(
        zip(
            request_sequence,
            drift,
        )
    ):

        g = runtime[
            runtime[
                "request_id"
            ]
            == rid
        ]

        safe = g[
            g[
                "degradation"
            ]
            <= DELTA + 1e-12
        ]

        energies = []

        for _, row in (
            safe.iterrows()
        ):

            energies.append(
                actual_energy(
                    raw,
                    rid,
                    row[
                        "policy"
                    ],
                    step,
                    factor,
                )
            )

        oracle_total += min(
            energies
        )

        full_total += (
            actual_energy(
                raw,
                rid,
                "full",
                step,
                factor,
            )
        )

    return (
        oracle_total,
        full_total,
    )


def select_static_policy(
    calibration,
):
    """
    Global static comparison policy.

    Choose the lowest-memory configuration whose empirical calibration
    violation rate satisfies the requested quality-risk target.
    """

    table = (
        calibration.groupby(
            "policy"
        )
        .agg(
            violation_rate=(
                "safe",
                lambda x:
                    1.0
                    - x.mean(),
            ),
            memory_ratio=(
                "memory_ratio",
                "mean",
            ),
        )
        .reset_index()
    )

    feasible = table[
        table[
            "violation_rate"
        ]
        <= RISK_TARGET
        + 1e-12
    ]

    if len(feasible) == 0:
        return "full"

    return (
        feasible.sort_values(
            [
                "memory_ratio",
                "violation_rate",
                "policy",
            ]
        )
        .iloc[0][
            "policy"
        ]
    )


def run_budget_controller(
    runtime,
    raw,
    request_sequence,
    drift,
    total_budget,
    threshold,
    method,
    static_policy=None,
):
    """
    Compare three controllers:

    static:
        one global compression policy

    open_loop:
        request-aware safety prediction + offline energy profile;
        no correction of stale energy estimates

    closed_loop:
        same RiskKV policy space + PI correction from actual energy.

    Both dynamic controllers know how much energy remains because
    budget accounting necessarily uses actual consumed energy.

    The difference is whether measured execution error corrects the
    future COST MODEL.
    """

    remaining = float(
        total_budget
    )

    ctrl = PIDController(
        kp=0.50,
        ki=0.05,
        kd=0.0,
        output_min=-0.25,
        output_max=1.00,
    )

    rows = []

    n_steps = len(
        request_sequence
    )

    for step, (
        rid,
        factor,
    ) in enumerate(
        zip(
            request_sequence,
            drift,
        )
    ):

        g = runtime[
            runtime[
                "request_id"
            ]
            == rid
        ].copy()

        remaining_steps = (
            n_steps
            - step
        )

        quota = (
            remaining
            / remaining_steps
        )

        if method == "static":

            chosen = g[
                g[
                    "policy"
                ]
                == static_policy
            ]

            if len(chosen) == 0:
                chosen = g[
                    g[
                        "policy"
                    ]
                    == "full"
                ]

            chosen = (
                chosen.iloc[0]
            )

            correction = 0.0

        else:

            accepted = g[
                g[
                    "prob_safe_mean"
                ]
                >= threshold
            ].copy()

            if len(accepted) == 0:
                accepted = g[
                    g[
                        "policy"
                    ]
                    == "full"
                ].copy()

            correction = (
                ctrl.value
                if method
                == "closed_loop"
                else 0.0
            )

            accepted[
                "predicted_energy"
            ] = accepted[
                "energy_j"
            ]

            compressed = (
                accepted[
                    "policy"
                ]
                != "full"
            )

            accepted.loc[
                compressed,
                "predicted_energy",
            ] *= (
                1.0
                + correction
            )

            feasible = accepted[
                accepted[
                    "predicted_energy"
                ]
                <= quota + 1e-12
            ]

            if len(feasible):

                # Maximize memory reduction while staying within the
                # current predicted energy quota.
                chosen = (
                    feasible.sort_values(
                        [
                            "memory_bytes",
                            "predicted_energy",
                            "policy",
                        ]
                    )
                    .iloc[0]
                )

            else:

                # Budget recovery:
                # select the lowest predicted-energy safe option.
                chosen = (
                    accepted.sort_values(
                        [
                            "predicted_energy",
                            "memory_bytes",
                            "policy",
                        ]
                    )
                    .iloc[0]
                )

        actual = actual_energy(
            raw,
            rid,
            chosen[
                "policy"
            ],
            step,
            factor,
        )

        remaining -= actual

        full = g[
            g[
                "policy"
            ]
            == "full"
        ].iloc[0]

        memory_saving = (
            1.0
            - chosen[
                "memory_bytes"
            ]
            / full[
                "memory_bytes"
            ]
        )

        quality_violation = (
            chosen[
                "degradation"
            ]
            > DELTA + 1e-12
        )

        # --------------------------------------------------------
        # CLOSED-LOOP ENERGY UPDATE
        # --------------------------------------------------------

        if (
            method
            == "closed_loop"
            and chosen[
                "policy"
            ]
            != "full"
        ):

            offline_prediction = float(
                chosen[
                    "energy_j"
                ]
            )

            error = (
                actual
                / offline_prediction
                - 1.0
            )

            ctrl.update(
                error
            )

        rows.append(
            {
                "step": step,
                "request_id": rid,
                "method": method,
                "policy": (
                    chosen[
                        "policy"
                    ]
                ),
                "quota_j": quota,
                "actual_energy_j": (
                    actual
                ),
                "remaining_budget_j": (
                    remaining
                ),
                "memory_saving": (
                    memory_saving
                ),
                "quality_violation": (
                    quality_violation
                ),
                "degradation": (
                    chosen[
                        "degradation"
                    ]
                ),
                "controller_correction": (
                    ctrl.value
                    if method
                    == "closed_loop"
                    else 0.0
                ),
                "fullkv": (
                    chosen[
                        "policy"
                    ]
                    == "full"
                ),
            }
        )

    return pd.DataFrame(
        rows
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

    risk = pd.read_csv(
        RISK
    )

    risk = risk[
        np.isclose(
            risk[
                "delta"
            ],
            DELTA,
        )
    ].copy()

    hardware_ids = set(
        profile[
            "request_id"
        ]
    )

    # ------------------------------------------------------------
    # Threshold is calibrated on NON-hardware-profile requests.
    #
    # Therefore the 25 hardware requests do not contribute their
    # quality outcomes to threshold selection.
    # ------------------------------------------------------------

    calibration = risk[
        ~risk[
            "request_id"
        ].isin(
            hardware_ids
        )
    ].copy()

    threshold = calibrate_threshold(
        calibration[
            "safe"
        ],
        calibration[
            "prob_safe_mean"
        ],
        RISK_TARGET,
    )

    static_policy = (
        select_static_policy(
            calibration
        )
    )

    hw_risk = risk[
        risk[
            "request_id"
        ].isin(
            hardware_ids
        )
    ].copy()

    runtime = (
        build_runtime_table(
            profile,
            hw_risk,
        )
    )

    ids = sorted(
        profile[
            "request_id"
        ].unique()
    )

    request_sequence = (
        ids * 4
    )[:100]

    drift = drift_trace()

    oracle_min, full_total = (
        oracle_budget_bounds(
            runtime,
            raw,
            request_sequence,
            drift,
        )
    )

    print(
        "Oracle minimum actual energy:",
        round(
            oracle_min,
            3,
        ),
        "J"
    )

    print(
        "FullKV actual energy:",
        round(
            full_total,
            3,
        ),
        "J"
    )

    print(
        "Risk threshold:",
        threshold
    )

    print(
        "Static policy:",
        static_policy
    )

    traces = []
    summaries = []

    for position in (
        BUDGET_POSITIONS
    ):

        total_budget = (
            oracle_min
            + position
            * (
                full_total
                - oracle_min
            )
        )

        for method in [
            "static",
            "open_loop",
            "closed_loop",
        ]:

            x = run_budget_controller(
                runtime=runtime,
                raw=raw,
                request_sequence=(
                    request_sequence
                ),
                drift=drift,
                total_budget=(
                    total_budget
                ),
                threshold=(
                    threshold
                ),
                method=method,
                static_policy=(
                    static_policy
                ),
            )

            x[
                "budget_position"
            ] = position

            x[
                "total_budget_j"
            ] = total_budget

            traces.append(x)

            consumed = (
                x[
                    "actual_energy_j"
                ].sum()
            )

            overshoot = max(
                0.0,
                consumed
                - total_budget,
            )

            summaries.append(
                {
                    "budget_position": (
                        position
                    ),
                    "method": method,
                    "total_budget_j": (
                        total_budget
                    ),
                    "actual_energy_j": (
                        consumed
                    ),
                    "budget_overshoot_j": (
                        overshoot
                    ),
                    "budget_overshoot_pct": (
                        100.0
                        * overshoot
                        / total_budget
                    ),
                    "quality_violation_rate": (
                        x[
                            "quality_violation"
                        ].mean()
                    ),
                    "mean_memory_saving": (
                        x[
                            "memory_saving"
                        ].mean()
                    ),
                    "fullkv_rate": (
                        x[
                            "fullkv"
                        ].mean()
                    ),
                    "final_remaining_budget_j": (
                        x[
                            "remaining_budget_j"
                        ].iloc[-1]
                    ),
                }
            )

    trace = pd.concat(
        traces,
        ignore_index=True,
    )

    summary = pd.DataFrame(
        summaries
    )

    trace.to_csv(
        OUTDIR
        / "budget_compliance_trace.csv",
        index=False,
    )

    summary.to_csv(
        OUTDIR
        / "budget_compliance_summary.csv",
        index=False,
    )

    print(
        "\nBUDGET COMPLIANCE"
    )

    print(
        summary.round(5)
        .to_string(
            index=False
        )
    )

    print(
        "\nEXPERIMENT 8E: PASS"
    )


if __name__ == "__main__":
    main()
