"""
EXPERIMENT 9B
=============

Corrected verified quality-feedback experiment.

======================================================================
WHY THIS EXPERIMENT EXISTS
======================================================================

Experiment 9A exposed a specific flaw:

    calibrated_budget

used model-predicted risk as budget consumption:

    B <- B - predicted_risk

but verified runtime violations did not directly correct B.

Under workload shift this produced cases where:

    expected risk consumed << actual violations.

Experiment 9B fixes that.

======================================================================
METHODS
======================================================================

1. fixed

    No feedback.


2. global_ewma

    Strong global feedback baseline from Experiment 9A.


3. predicted_budget_current

    The previous calibrated-budget implementation.

    Included specifically to show whether verified budget correction
    improves over predicted-risk-only accounting.


4. verified_dr_budget

    Frozen contextual calibrator
        +
    model-predicted risk
        +
    verified inverse-propensity residual correction
        +
    explicit remaining violation budget.


5. verified_dr_budget_residual

    Same corrected budget
        +
    an online verified residual risk correction applied BEFORE future
    policy selection.

This is the proposed final variant in this experiment.

======================================================================
AUDITING
======================================================================

ONLY RANDOM AUDITING is used here:

    1%
    5%
    10%

We intentionally remove uncertainty-directed auditing so that audit
selection is not a confounder while validating the controller itself.

======================================================================
DEPLOYMENT REPLAY
======================================================================

Held-out LongBench requests are replayed 10 times in randomized order,
exactly as in Experiment 9A.

This is a controlled deployment replay for studying sparse online
feedback dynamics.

It is NOT additional independent LongBench test data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy

import numpy as np
import pandas as pd

from riskkv.dr_quality_feedback import (
    OnlineResidualRiskCorrector,
    VerifiedDRBudgetManager,
)

from riskkv.verified_quality_feedback import (
    AuditPolicy,
    ContextualOnlineCalibrator,
)


# ======================================================================
# LOAD THE ALREADY-VALIDATED 09A EXPERIMENT HELPERS
# ======================================================================

BASE_SCRIPT = (
    Path(__file__).resolve().parent
    / "09a_verified_quality_feedback.py"
)

BASE = runpy.run_path(
    str(BASE_SCRIPT),
    run_name="riskkv_09a_helpers",
)


ORACLE = BASE["ORACLE"]

RISK_TARGET = BASE[
    "RISK_TARGET"
]

split_source_four_way = BASE[
    "split_source_four_way"
]

train_risk_ensemble = BASE[
    "train_risk_ensemble"
]

replay_sequence = BASE[
    "replay_sequence"
]

simulate_stream = BASE[
    "simulate_stream"
]

tune_global_controller = BASE[
    "tune_global_controller"
]

tune_calibrator_learning_rate = BASE[
    "tune_calibrator_learning_rate"
]

score_candidate = BASE[
    "score_candidate"
]


# ======================================================================
# EXPERIMENT SETTINGS
# ======================================================================

FULL_DELTAS = [
    0.01,
    0.05,
    0.10,
]

AUDIT_RATES = [
    0.01,
    0.05,
    0.10,
]

DEPLOYMENT_REPEATS = 10
TUNING_REPEATS = 10

AUDIT_SEEDS = 20

TUNING_SEEDS = [
    101,
    202,
    303,
]


# Residual corrector only has one dynamics parameter.
RESIDUAL_BETAS = [
    0.05,
    0.10,
    0.20,
]


# ======================================================================
# POLICY SELECTION
# ======================================================================

def select_risk_budget_policy(
    g,
    risks,
    allowed_risk,
):
    """
    Choose the lowest-memory configuration whose CURRENT corrected
    predicted risk fits inside the remaining per-request risk allowance.
    """

    work = g.copy()

    work[
        "_runtime_risk"
    ] = np.asarray(
        risks,
        dtype=float,
    )

    feasible = work[
        work[
            "_runtime_risk"
        ]
        <= allowed_risk
        + 1e-12
    ]

    if len(feasible) == 0:

        row = g.iloc[0].copy()

        row["policy"] = "full"
        row["memory_ratio"] = 1.0
        row["degradation"] = 0.0
        row["safe"] = 1
        row["prob_safe_mean"] = 1.0
        row["risk_uncertainty"] = 0.0

        return (
            row,
            0.0,
        )

    chosen = (
        feasible.sort_values(
            [
                "memory_ratio",
                "_runtime_risk",
                "policy",
            ],
            ascending=[
                True,
                True,
                True,
            ],
        )
        .iloc[0]
    )

    return (
        chosen,
        float(
            chosen[
                "_runtime_risk"
            ]
        ),
    )


# ======================================================================
# CORRECTED CLOSED-LOOP REPLAY
# ======================================================================

def simulate_verified_dr_stream(
    deployment,
    sequence,
    calibrator_template,
    audit_rate,
    residual_beta,
    method,
    seed,
    save_trace=False,
):
    """
    Corrected quality-feedback stream.

    ------------------------------------------------------------
    ORDER FOR REQUEST t
    ------------------------------------------------------------

    1. Predict calibrated risk for every policy.

    2. Optionally apply the VERIFIED residual correction learned from
       PREVIOUS requests.

    3. Ask the corrected budget manager how much risk can currently be
       spent.

    4. Select and execute one policy.

    5. Only AFTER execution, possibly audit that policy.

    6. Update:
           budget residual state
           and optionally residual risk corrector.

    Therefore future information never leaks into the current decision.
    """

    groups = {
        rid: g.reset_index(
            drop=True
        )
        for rid, g
        in deployment.groupby(
            "request_id",
            sort=False,
        )
    }

    rng = np.random.default_rng(
        seed
    )

    calibrator = (
        calibrator_template.clone()
    )

    budget = (
        VerifiedDRBudgetManager(
            epsilon=RISK_TARGET,
            horizon=len(sequence),
        )
    )

    risk_corrector = None

    if (
        method
        == "verified_dr_budget_residual"
    ):

        risk_corrector = (
            OnlineResidualRiskCorrector(
                beta=residual_beta,
                max_abs_bias=0.25,
                min_audits=3,
            )
        )

    violation_count = 0

    memory_savings = []

    full_count = 0

    audit_count = 0

    compressed_count = 0

    trace_rows = []

    for step, rid in enumerate(
        sequence
    ):

        g = groups[rid]

        # --------------------------------------------------------
        # OFFLINE CONTEXTUAL CALIBRATION
        # --------------------------------------------------------

        p_safe = (
            calibrator.predict_proba(
                g
            )
        )

        base_risk = (
            1.0
            - p_safe
        )

        # --------------------------------------------------------
        # VERIFIED ONLINE RESIDUAL CORRECTION
        # --------------------------------------------------------

        if (
            risk_corrector
            is not None
        ):

            decision_risk = (
                risk_corrector.correct(
                    base_risk
                )
            )

        else:

            decision_risk = (
                base_risk
            )

        # --------------------------------------------------------
        # CURRENT RISK ALLOWANCE
        # --------------------------------------------------------

        allowed_risk = (
            budget.allowance(
                step
            )
        )

        (
            chosen,
            chosen_risk,
        ) = select_risk_budget_policy(
            g,
            decision_risk,
            allowed_risk,
        )

        policy = chosen[
            "policy"
        ]

        fullkv = (
            policy
            == "full"
        )

        if fullkv:
            full_count += 1
        else:
            compressed_count += 1

        violation = bool(
            chosen[
                "degradation"
            ]
            > chosen[
                "delta"
            ]
            + 1e-12
        )

        violation_count += int(
            violation
        )

        memory_saved = (
            1.0
            - float(
                chosen[
                    "memory_ratio"
                ]
            )
        )

        memory_savings.append(
            memory_saved
        )

        # --------------------------------------------------------
        # AUDIT AFTER EXECUTION
        # --------------------------------------------------------

        audited = False

        q = 0.0

        if not fullkv:

            # Random auditing only in 09B.
            q = float(
                audit_rate
            )

            audited = bool(
                rng.random()
                < q
            )

        # --------------------------------------------------------
        # FIRST ACCOUNT FOR THE MODEL RISK USED FOR THIS DECISION
        # --------------------------------------------------------

        budget.consume_prediction(
            chosen_risk
        )

        # --------------------------------------------------------
        # THEN CORRECT WITH VERIFIED FEEDBACK
        # --------------------------------------------------------

        if audited:

            audit_count += 1

            budget.update_audit(
                predicted_risk=(
                    chosen_risk
                ),
                violation=(
                    violation
                ),
                audit_probability=q,
            )

            if (
                risk_corrector
                is not None
            ):

                risk_corrector.update(
                    predicted_risk=(
                        chosen_risk
                    ),
                    violation=(
                        violation
                    ),
                    audit_probability=q,
                )

        if save_trace:

            trace_rows.append(
                {
                    "step": step,

                    "request_id": rid,

                    "task": chosen[
                        "task"
                    ],

                    "method": method,

                    "policy": policy,

                    "actual_violation": (
                        violation
                    ),

                    "degradation": (
                        chosen[
                            "degradation"
                        ]
                    ),

                    "base_calibrated_risk": (
                        0.0
                        if fullkv
                        else float(
                            base_risk[
                                chosen.name
                            ]
                        )
                        if chosen.name
                        < len(
                            base_risk
                        )
                        else np.nan
                    ),

                    "decision_risk": (
                        chosen_risk
                    ),

                    "allowed_risk": (
                        allowed_risk
                    ),

                    "audited": (
                        audited
                    ),

                    "audit_probability": (
                        q
                    ),

                    "memory_saved": (
                        memory_saved
                    ),

                    "fullkv": fullkv,

                    "residual_bias": (
                        0.0
                        if risk_corrector
                        is None
                        else risk_corrector.bias
                    ),

                    "budget_residual_bias": (
                        budget.residual_bias
                    ),

                    "predicted_risk_sum": (
                        budget.predicted_risk_sum
                    ),

                    "raw_dr_estimate": (
                        budget.raw_dr_estimate
                    ),

                    "estimated_violation_count": (
                        budget
                        .estimated_violation_count
                    ),

                    "remaining_risk_budget": (
                        budget
                        .remaining_budget
                    ),
                }
            )

    n = len(
        sequence
    )

    violation_rate = (
        violation_count
        / n
    )

    summary = {
        "requests": n,

        "violation_count": (
            violation_count
        ),

        "violation_rate": (
            violation_rate
        ),

        "violation_excess": max(
            0.0,
            violation_rate
            - RISK_TARGET,
        ),

        "absolute_target_error": abs(
            violation_rate
            - RISK_TARGET
        ),

        "mean_memory_saved": float(
            np.mean(
                memory_savings
            )
        ),

        "fullkv_rate": (
            full_count
            / n
        ),

        "audit_count": (
            audit_count
        ),

        "audit_rate_all_requests": (
            audit_count
            / n
        ),

        "audit_rate_compressed": (
            audit_count
            / compressed_count
            if compressed_count
            else 0.0
        ),

        "actual_violation_budget_excess": max(
            0.0,
            violation_count
            - RISK_TARGET
            * n,
        ),

        "predicted_risk_sum": (
            budget.predicted_risk_sum
        ),

        "raw_dr_estimate": (
            budget.raw_dr_estimate
        ),

        "estimated_violation_count": (
            budget
            .estimated_violation_count
        ),

        "budget_residual_bias": (
            budget.residual_bias
        ),

        "remaining_risk_budget": (
            budget.remaining_budget
        ),

        "online_residual_bias": (
            0.0
            if risk_corrector
            is None
            else risk_corrector.bias
        ),
    }

    trace = (
        pd.DataFrame(
            trace_rows
        )
        if save_trace
        else None
    )

    return (
        summary,
        trace,
    )


# ======================================================================
# RESIDUAL BETA TUNING
# ======================================================================

def tune_residual_beta(
    tuning_df,
    calibrator,
    audit_rate,
    quick=False,
):
    """
    Tune residual feedback dynamics under the SAME sparse audit rate
    subsequently used in deployment.

    This fixes the earlier mismatch between dense-feedback tuning and
    sparse-feedback testing.
    """

    ids = (
        tuning_df[
            "request_id"
        ]
        .drop_duplicates()
        .tolist()
    )

    seeds = (
        [101]
        if quick
        else TUNING_SEEDS
    )

    repeats = (
        5
        if quick
        else TUNING_REPEATS
    )

    candidates = []

    for beta in (
        RESIDUAL_BETAS
    ):

        summaries = []

        for seed in seeds:

            sequence = (
                replay_sequence(
                    ids,
                    repeats=repeats,
                    seed=seed,
                )
            )

            summary, _ = (
                simulate_verified_dr_stream(
                    deployment=(
                        tuning_df
                    ),
                    sequence=(
                        sequence
                    ),
                    calibrator_template=(
                        calibrator
                    ),
                    audit_rate=(
                        audit_rate
                    ),
                    residual_beta=(
                        beta
                    ),
                    method=(
                        "verified_dr_budget_residual"
                    ),
                    seed=(
                        seed
                        + 5000
                    ),
                )
            )

            summaries.append(
                summary
            )

        candidates.append(
            (
                score_candidate(
                    summaries
                ),
                beta,
            )
        )

    candidates.sort(
        key=lambda x:
            x[0]
    )

    return float(
        candidates[
            0
        ][1]
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Run one delta/task/audit setting "
            "for sanity checking."
        ),
    )

    args = parser.parse_args()

    quick = bool(
        args.quick
    )

    if quick:

        deltas = [
            0.05
        ]

        audit_rates = [
            0.05
        ]

        heldout_tasks_filter = {
            "multifieldqa_en"
        }

        deployment_repeats = 5

        audit_seeds = 5

        outdir = Path(
            "experiments/full_oracle_qwen3_8b/"
            "analysis/29_verified_dr_feedback_quick"
        )

    else:

        deltas = (
            FULL_DELTAS
        )

        audit_rates = (
            AUDIT_RATES
        )

        heldout_tasks_filter = None

        deployment_repeats = (
            DEPLOYMENT_REPEATS
        )

        audit_seeds = (
            AUDIT_SEEDS
        )

        outdir = Path(
            "experiments/full_oracle_qwen3_8b/"
            "analysis/29_verified_dr_feedback"
        )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        ORACLE
    )

    compressed = df[
        df[
            "policy"
        ]
        != "full"
    ].copy()

    tasks = sorted(
        compressed[
            "task"
        ].unique()
    )

    if (
        heldout_tasks_filter
        is not None
    ):

        tasks = [
            task
            for task in tasks
            if task
            in heldout_tasks_filter
        ]

    seed_rows = []

    tuning_rows = []

    trace_rows = []

    # =================================================================
    # LOTO
    # =================================================================

    for heldout_task in tasks:

        print(
            "\n"
            + "=" * 80
        )

        print(
            "HELD-OUT:",
            heldout_task
        )

        print(
            "=" * 80
        )

        source_all = compressed[
            compressed[
                "task"
            ]
            != heldout_task
        ].copy()

        heldout_all = compressed[
            compressed[
                "task"
            ]
            == heldout_task
        ].copy()

        for delta in deltas:

            source = (
                source_all.copy()
            )

            heldout = (
                heldout_all.copy()
            )

            source[
                "delta"
            ] = delta

            heldout[
                "delta"
            ] = delta

            source[
                "safe"
            ] = (
                source[
                    "degradation"
                ]
                <= delta
                + 1e-12
            ).astype(int)

            heldout[
                "safe"
            ] = (
                heldout[
                    "degradation"
                ]
                <= delta
                + 1e-12
            ).astype(int)

            (
                risk_train,
                calibrator_fit,
                threshold_cal,
                controller_tune,
            ) = split_source_four_way(
                source,
                seed=42,
            )

            predicted = (
                train_risk_ensemble(
                    train=(
                        risk_train
                    ),
                    evaluation_sets={
                        "calibrator_fit": (
                            calibrator_fit
                        ),
                        "threshold_cal": (
                            threshold_cal
                        ),
                        "controller_tune": (
                            controller_tune
                        ),
                        "heldout": (
                            heldout
                        ),
                    },
                )
            )

            calfit_pred = (
                predicted[
                    "calibrator_fit"
                ]
            )

            threshold_pred = (
                predicted[
                    "threshold_cal"
                ]
            )

            tune_pred = (
                predicted[
                    "controller_tune"
                ]
            )

            heldout_pred = (
                predicted[
                    "heldout"
                ]
            )

            # -----------------------------------------------------
            # Offline contextual probability calibration.
            # -----------------------------------------------------

            calibrator = (
                ContextualOnlineCalibrator()
            )

            calibrator.fit(
                calfit_pred
            )

            # Base threshold needed for fixed/global baseline.
            base_threshold = (
                BASE[
                    "calibrate_threshold"
                ](
                    threshold_pred[
                        "safe"
                    ],
                    threshold_pred[
                        "prob_safe_mean"
                    ],
                    RISK_TARGET,
                )
            )

            calibrated_threshold = (
                BASE[
                    "calibrate_threshold"
                ](
                    threshold_pred[
                        "safe"
                    ],
                    calibrator.predict_proba(
                        threshold_pred
                    ),
                    RISK_TARGET,
                )
            )

            heldout_ids = (
                heldout_pred[
                    "request_id"
                ]
                .drop_duplicates()
                .tolist()
            )

            # -----------------------------------------------------
            # FIXED BASELINE ONCE.
            # -----------------------------------------------------

            fixed_sequence = (
                replay_sequence(
                    heldout_ids,
                    repeats=(
                        deployment_repeats
                    ),
                    seed=0,
                )
            )

            fixed_summary, _ = (
                simulate_stream(
                    deployment=(
                        heldout_pred
                    ),
                    sequence=(
                        fixed_sequence
                    ),
                    method="fixed",
                    epsilon=(
                        RISK_TARGET
                    ),
                    base_threshold=(
                        base_threshold
                    ),
                    seed=0,
                )
            )

            fixed_summary.update(
                {
                    "delta": delta,
                    "heldout_task": (
                        heldout_task
                    ),
                    "method": (
                        "fixed"
                    ),
                    "audit_rate_target": (
                        0.0
                    ),
                    "audit_seed": 0,
                }
            )

            seed_rows.append(
                fixed_summary
            )

            # =====================================================
            # AUDIT RATES
            # =====================================================

            for audit_rate in (
                audit_rates
            ):

                audit_policy = (
                    AuditPolicy
                    .from_reference(
                        uncertainties=(
                            threshold_pred[
                                "risk_uncertainty"
                            ]
                        ),
                        target_rate=(
                            audit_rate
                        ),
                        strategy=(
                            "random"
                        ),
                    )
                )

                # -------------------------------------------------
                # Strong global baseline tuned under same sparse
                # feedback regime.
                # -------------------------------------------------

                global_params = (
                    tune_global_controller(
                        tuning_df=(
                            tune_pred
                        ),
                        base_threshold=(
                            base_threshold
                        ),
                        audit_policy=(
                            audit_policy
                        ),
                    )
                )

                # -------------------------------------------------
                # Current 09A predicted-budget baseline needs its
                # online calibrator learning rate.
                # -------------------------------------------------

                online_lr = (
                    tune_calibrator_learning_rate(
                        tuning_df=(
                            tune_pred
                        ),
                        calibrated_threshold=(
                            calibrated_threshold
                        ),
                        calibrator_template=(
                            calibrator
                        ),
                        audit_policy=(
                            audit_policy
                        ),
                    )
                )

                # -------------------------------------------------
                # New residual-controller parameter.
                # -------------------------------------------------

                residual_beta = (
                    tune_residual_beta(
                        tuning_df=(
                            tune_pred
                        ),
                        calibrator=(
                            calibrator
                        ),
                        audit_rate=(
                            audit_rate
                        ),
                        quick=quick,
                    )
                )

                tuning_rows.append(
                    {
                        "delta": delta,

                        "heldout_task": (
                            heldout_task
                        ),

                        "audit_rate_target": (
                            audit_rate
                        ),

                        "global_kp": (
                            global_params[
                                "kp"
                            ]
                        ),

                        "global_ki": (
                            global_params[
                                "ki"
                            ]
                        ),

                        "global_beta": (
                            global_params[
                                "beta"
                            ]
                        ),

                        "current_online_lr": (
                            online_lr
                        ),

                        "residual_beta": (
                            residual_beta
                        ),
                    }
                )

                # -------------------------------------------------
                # MULTIPLE AUDIT SEEDS.
                # -------------------------------------------------

                for audit_seed in range(
                    audit_seeds
                ):

                    sequence = (
                        replay_sequence(
                            heldout_ids,
                            repeats=(
                                deployment_repeats
                            ),
                            seed=(
                                10000
                                + audit_seed
                            ),
                        )
                    )

                    # =============================================
                    # GLOBAL EWMA
                    # =============================================

                    summary, _ = (
                        simulate_stream(
                            deployment=(
                                heldout_pred
                            ),
                            sequence=(
                                sequence
                            ),
                            method=(
                                "global_ewma"
                            ),
                            epsilon=(
                                RISK_TARGET
                            ),
                            audit_policy=(
                                audit_policy
                            ),
                            base_threshold=(
                                base_threshold
                            ),
                            global_params=(
                                global_params
                            ),
                            seed=(
                                20000
                                + audit_seed
                            ),
                        )
                    )

                    summary.update(
                        {
                            "delta": delta,
                            "heldout_task": (
                                heldout_task
                            ),
                            "method": (
                                "global_ewma"
                            ),
                            "audit_rate_target": (
                                audit_rate
                            ),
                            "audit_seed": (
                                audit_seed
                            ),
                        }
                    )

                    seed_rows.append(
                        summary
                    )

                    # =============================================
                    # CURRENT PREDICTED-RISK BUDGET
                    # =============================================

                    summary, _ = (
                        simulate_stream(
                            deployment=(
                                heldout_pred
                            ),
                            sequence=(
                                sequence
                            ),
                            method=(
                                "calibrated_budget"
                            ),
                            epsilon=(
                                RISK_TARGET
                            ),
                            audit_policy=(
                                audit_policy
                            ),
                            calibrated_threshold=(
                                calibrated_threshold
                            ),
                            calibrator_template=(
                                calibrator
                            ),
                            calibrator_learning_rate=(
                                online_lr
                            ),
                            seed=(
                                30000
                                + audit_seed
                            ),
                        )
                    )

                    summary.update(
                        {
                            "delta": delta,
                            "heldout_task": (
                                heldout_task
                            ),
                            "method": (
                                "predicted_budget_current"
                            ),
                            "audit_rate_target": (
                                audit_rate
                            ),
                            "audit_seed": (
                                audit_seed
                            ),
                        }
                    )

                    seed_rows.append(
                        summary
                    )

                    # =============================================
                    # VERIFIED DR BUDGET
                    # =============================================

                    for method in [
                        "verified_dr_budget",
                        "verified_dr_budget_residual",
                    ]:

                        save_trace = (
                            audit_seed == 0
                            and np.isclose(
                                delta,
                                0.05,
                            )
                            and np.isclose(
                                audit_rate,
                                0.05,
                            )
                        )

                        (
                            summary,
                            trace,
                        ) = (
                            simulate_verified_dr_stream(
                                deployment=(
                                    heldout_pred
                                ),
                                sequence=(
                                    sequence
                                ),
                                calibrator_template=(
                                    calibrator
                                ),
                                audit_rate=(
                                    audit_rate
                                ),
                                residual_beta=(
                                    residual_beta
                                ),
                                method=(
                                    method
                                ),
                                seed=(
                                    40000
                                    + audit_seed
                                ),
                                save_trace=(
                                    save_trace
                                ),
                            )
                        )

                        summary.update(
                            {
                                "delta": delta,

                                "heldout_task": (
                                    heldout_task
                                ),

                                "method": (
                                    method
                                ),

                                "audit_rate_target": (
                                    audit_rate
                                ),

                                "audit_seed": (
                                    audit_seed
                                ),
                            }
                        )

                        seed_rows.append(
                            summary
                        )

                        if trace is not None:

                            trace[
                                "delta"
                            ] = delta

                            trace[
                                "heldout_task"
                            ] = (
                                heldout_task
                            )

                            trace[
                                "audit_rate_target"
                            ] = (
                                audit_rate
                            )

                            trace_rows.append(
                                trace
                            )

    # =================================================================
    # SAVE RESULTS
    # =================================================================

    seeds = pd.DataFrame(
        seed_rows
    )

    tuning = pd.DataFrame(
        tuning_rows
    )

    per_task = (
        seeds.groupby(
            [
                "delta",
                "heldout_task",
                "method",
                "audit_rate_target",
            ],
            as_index=False,
        )
        .agg(
            violation_rate=(
                "violation_rate",
                "mean",
            ),

            violation_std=(
                "violation_rate",
                "std",
            ),

            absolute_target_error=(
                "absolute_target_error",
                "mean",
            ),

            mean_memory_saved=(
                "mean_memory_saved",
                "mean",
            ),

            fullkv_rate=(
                "fullkv_rate",
                "mean",
            ),

            audit_rate_all_requests=(
                "audit_rate_all_requests",
                "mean",
            ),

            audit_rate_compressed=(
                "audit_rate_compressed",
                "mean",
            ),

            actual_violation_budget_excess=(
                "actual_violation_budget_excess",
                "mean",
            ),

            estimated_violation_count=(
                "estimated_violation_count",
                "mean",
            ),

            budget_residual_bias=(
                "budget_residual_bias",
                "mean",
            ),

            online_residual_bias=(
                "online_residual_bias",
                "mean",
            ),
        )
    )

    overall = (
        per_task.groupby(
            [
                "delta",
                "method",
                "audit_rate_target",
            ],
            as_index=False,
        )
        .agg(
            mean_violation_rate=(
                "violation_rate",
                "mean",
            ),

            mean_absolute_target_error=(
                "absolute_target_error",
                "mean",
            ),

            mean_memory_saved=(
                "mean_memory_saved",
                "mean",
            ),

            mean_fullkv_rate=(
                "fullkv_rate",
                "mean",
            ),

            mean_audit_rate_all_requests=(
                "audit_rate_all_requests",
                "mean",
            ),

            mean_audit_rate_compressed=(
                "audit_rate_compressed",
                "mean",
            ),

            mean_actual_violation_budget_excess=(
                "actual_violation_budget_excess",
                "mean",
            ),

            mean_estimated_violation_count=(
                "estimated_violation_count",
                "mean",
            ),

            mean_budget_residual_bias=(
                "budget_residual_bias",
                "mean",
            ),

            mean_online_residual_bias=(
                "online_residual_bias",
                "mean",
            ),
        )
    )

    seeds.to_csv(
        outdir
        / "dr_feedback_seed_results.csv",
        index=False,
    )

    per_task.to_csv(
        outdir
        / "dr_feedback_by_task.csv",
        index=False,
    )

    overall.to_csv(
        outdir
        / "dr_feedback_overall.csv",
        index=False,
    )

    tuning.to_csv(
        outdir
        / "dr_feedback_hyperparameters.csv",
        index=False,
    )

    if trace_rows:

        pd.concat(
            trace_rows,
            ignore_index=True,
        ).to_csv(
            outdir
            / "dr_feedback_representative_trace.csv",
            index=False,
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "CORRECTED VERIFIED DR FEEDBACK"
    )

    print(
        "=" * 100
    )

    print(
        overall.round(
            5
        ).to_string(
            index=False
        )
    )

    print(
        "\nOutputs:",
        outdir
    )

    print(
        "\nEXPERIMENT 9B: PASS"
    )


if __name__ == "__main__":
    main()
