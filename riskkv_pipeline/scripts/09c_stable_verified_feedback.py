"""
EXPERIMENT 09C

Stable verified quality feedback.

Comparison:
    fixed
    global_ewma
    predicted_budget_current
    verified_replacement_budget
    verified_replacement_budget_residual

Only RANDOM audits:
    1%, 5%, 10%.

Key correction relative to 09B:
    - no inverse-propensity explosion in the runtime controller;
    - no retroactive residual bias applied to all past requests;
    - audited request replaces predicted debit with exact verified z;
    - residual feedback affects future risk only.

Held-out-task replay remains a controlled online-feedback experiment,
not additional independent LongBench data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy

import numpy as np
import pandas as pd

from riskkv.stable_quality_feedback import (
    VerifiedReplacementBudget,
    OnlineResidualBias,
)

from riskkv.verified_quality_feedback import (
    AuditPolicy,
    ContextualOnlineCalibrator,
)


# =====================================================================
# REUSE 09A HELPERS
# =====================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

A = runpy.run_path(
    str(
        SCRIPT_DIR
        / "09a_verified_quality_feedback.py"
    ),
    run_name="riskkv_09a_helpers",
)

ORACLE = A["ORACLE"]

RISK_TARGET = A[
    "RISK_TARGET"
]

split_source_four_way = A[
    "split_source_four_way"
]

train_risk_ensemble = A[
    "train_risk_ensemble"
]

replay_sequence = A[
    "replay_sequence"
]

simulate_stream = A[
    "simulate_stream"
]

tune_global_controller = A[
    "tune_global_controller"
]

tune_calibrator_learning_rate = A[
    "tune_calibrator_learning_rate"
]

score_candidate = A[
    "score_candidate"
]

calibrate_threshold = A[
    "calibrate_threshold"
]


# =====================================================================
# SETTINGS
# =====================================================================

DELTAS = [
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

AUDIT_SEEDS = 20

TUNING_SEEDS = [
    101,
    202,
    303,
]

TUNING_REPEATS = 10

RESIDUAL_BETAS = [
    0.05,
    0.10,
    0.20,
]


# =====================================================================
# SELECTION
# =====================================================================

def select_budget_policy(
    g,
    risks,
    allowed_risk,
):
    work = g.copy()

    work[
        "_risk"
    ] = np.asarray(
        risks,
        dtype=float,
    )

    feasible = work[
        work["_risk"]
        <= allowed_risk
        + 1e-12
    ]

    if len(feasible) == 0:

        row = g.iloc[0].copy()

        row["policy"] = "full"
        row["memory_ratio"] = 1.0
        row["degradation"] = 0.0
        row["safe"] = 1
        row["risk_uncertainty"] = 0.0

        return (
            row,
            0.0,
        )

    chosen = (
        feasible.sort_values(
            [
                "memory_ratio",
                "_risk",
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
            chosen["_risk"]
        ),
    )


# =====================================================================
# CLOSED LOOP
# =====================================================================

def simulate_stable_stream(
    deployment,
    sequence,
    calibrator_template,
    audit_rate,
    residual_beta,
    method,
    seed,
    save_trace=False,
):
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
        VerifiedReplacementBudget(
            epsilon=RISK_TARGET,
            horizon=len(sequence),
        )
    )

    corrector = None

    if (
        method
        == "verified_replacement_budget_residual"
    ):
        corrector = (
            OnlineResidualBias(
                beta=residual_beta,
                min_audits=3,
                max_abs_bias=0.20,
            )
        )

    violations = 0
    full_count = 0
    compressed_count = 0
    audit_count = 0

    memory_saved_values = []

    traces = []

    for step, rid in enumerate(
        sequence
    ):

        g = groups[rid]

        # ---------------------------------------------------------
        # OFFLINE CONTEXTUAL CALIBRATION
        # ---------------------------------------------------------

        p_safe = (
            calibrator.predict_proba(
                g
            )
        )

        base_risk = (
            1.0
            - p_safe
        )

        # ---------------------------------------------------------
        # ONLINE VERIFIED RESIDUAL CORRECTION
        # Affects FUTURE requests only.
        # ---------------------------------------------------------

        if corrector is None:

            runtime_risk = (
                base_risk
            )

        else:

            runtime_risk = (
                corrector.correct(
                    base_risk
                )
            )

        # ---------------------------------------------------------
        # CURRENT REMAINING BUDGET
        # ---------------------------------------------------------

        allowance = (
            budget.allowance(
                step
            )
        )

        (
            chosen,
            selected_risk,
        ) = select_budget_policy(
            g,
            runtime_risk,
            allowance,
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

        violations += int(
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

        memory_saved_values.append(
            memory_saved
        )

        # ---------------------------------------------------------
        # INITIAL BUDGET DEBIT
        #
        # FullKV risk is zero.
        # ---------------------------------------------------------

        budget.consume_prediction(
            selected_risk
        )

        # ---------------------------------------------------------
        # RANDOM VERIFIED AUDIT
        # ---------------------------------------------------------

        audited = False

        if not fullkv:

            audited = bool(
                rng.random()
                < audit_rate
            )

        if audited:

            audit_count += 1

            # Replace this request's predicted debit with actual z.
            budget.replace_with_verified(
                predicted_risk=(
                    selected_risk
                ),
                violation=(
                    violation
                ),
            )

            # Use residual ONLY for future requests.
            if corrector is not None:

                corrector.update(
                    predicted_risk=(
                        selected_risk
                    ),
                    violation=(
                        violation
                    ),
                )

        if save_trace:

            traces.append(
                {
                    "step": step,
                    "request_id": rid,
                    "task": (
                        chosen["task"]
                    ),
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

                    "selected_risk": (
                        selected_risk
                    ),

                    "allowed_risk": (
                        allowance
                    ),

                    "audited": (
                        audited
                    ),

                    "memory_saved": (
                        memory_saved
                    ),

                    "fullkv": (
                        fullkv
                    ),

                    "online_residual_bias": (
                        0.0
                        if corrector
                        is None
                        else corrector.bias
                    ),

                    "audit_residual_mean": (
                        budget
                        .mean_audit_residual
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
        violations
        / n
    )

    return (
        {
            "requests": n,

            "violation_count": (
                violations
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
                    memory_saved_values
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
                violations
                - RISK_TARGET
                * n,
            ),

            "estimated_violation_count": (
                budget
                .estimated_violation_count
            ),

            "remaining_risk_budget": (
                budget
                .remaining_budget
            ),

            "audit_residual_mean": (
                budget
                .mean_audit_residual
            ),

            "online_residual_bias": (
                0.0
                if corrector
                is None
                else corrector.bias
            ),
        },

        (
            pd.DataFrame(
                traces
            )
            if save_trace
            else None
        ),
    )


# =====================================================================
# TUNE BETA UNDER SAME SPARSE-AUDIT REGIME
# =====================================================================

def tune_beta(
    tuning_df,
    calibrator,
    audit_rate,
    quick=False,
):
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

            seq = replay_sequence(
                ids,
                repeats=repeats,
                seed=seed,
            )

            s, _ = (
                simulate_stable_stream(
                    deployment=(
                        tuning_df
                    ),
                    sequence=seq,
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
                        "verified_replacement_budget_residual"
                    ),
                    seed=(
                        5000
                        + seed
                    ),
                )
            )

            summaries.append(s)

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
        candidates[0][1]
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--quick",
        action="store_true",
    )

    args = parser.parse_args()

    if args.quick:

        deltas = [0.05]
        audit_rates = [0.05]
        tasks_filter = {
            "multifieldqa_en"
        }

        deployment_repeats = 5
        audit_seeds = 5

        outdir = Path(
            "experiments/full_oracle_qwen3_8b/"
            "analysis/30_stable_verified_feedback_quick"
        )

    else:

        deltas = DELTAS
        audit_rates = AUDIT_RATES
        tasks_filter = None

        deployment_repeats = (
            DEPLOYMENT_REPEATS
        )

        audit_seeds = (
            AUDIT_SEEDS
        )

        outdir = Path(
            "experiments/full_oracle_qwen3_8b/"
            "analysis/30_stable_verified_feedback"
        )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        ORACLE
    )

    df = df[
        df["policy"]
        != "full"
    ].copy()

    tasks = sorted(
        df["task"].unique()
    )

    if tasks_filter is not None:

        tasks = [
            t
            for t in tasks
            if t in tasks_filter
        ]

    rows = []
    tuning_rows = []
    traces = []

    for heldout_task in tasks:

        print(
            "\nHELD-OUT:",
            heldout_task
        )

        source_all = df[
            df["task"]
            != heldout_task
        ].copy()

        heldout_all = df[
            df["task"]
            == heldout_task
        ].copy()

        for delta in deltas:

            source = (
                source_all.copy()
            )

            heldout = (
                heldout_all.copy()
            )

            source["delta"] = delta
            heldout["delta"] = delta

            source["safe"] = (
                source["degradation"]
                <= delta
                + 1e-12
            ).astype(int)

            heldout["safe"] = (
                heldout["degradation"]
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

            pred = train_risk_ensemble(
                train=risk_train,
                evaluation_sets={
                    "calfit": (
                        calibrator_fit
                    ),
                    "threshold": (
                        threshold_cal
                    ),
                    "tune": (
                        controller_tune
                    ),
                    "heldout": (
                        heldout
                    ),
                },
            )

            calfit = pred["calfit"]
            threshold = pred["threshold"]
            tune = pred["tune"]
            test = pred["heldout"]

            calibrator = (
                ContextualOnlineCalibrator()
            )

            calibrator.fit(
                calfit
            )

            base_threshold = (
                calibrate_threshold(
                    threshold["safe"],
                    threshold[
                        "prob_safe_mean"
                    ],
                    RISK_TARGET,
                )
            )

            calibrated_threshold = (
                calibrate_threshold(
                    threshold["safe"],
                    calibrator.predict_proba(
                        threshold
                    ),
                    RISK_TARGET,
                )
            )

            test_ids = (
                test["request_id"]
                .drop_duplicates()
                .tolist()
            )

            # =====================================================
            # FIXED
            # =====================================================

            seq = replay_sequence(
                test_ids,
                repeats=(
                    deployment_repeats
                ),
                seed=0,
            )

            s, _ = simulate_stream(
                deployment=test,
                sequence=seq,
                method="fixed",
                epsilon=RISK_TARGET,
                base_threshold=(
                    base_threshold
                ),
                seed=0,
            )

            s.update(
                {
                    "delta": delta,
                    "heldout_task": (
                        heldout_task
                    ),
                    "method": "fixed",
                    "audit_rate_target": 0.0,
                    "audit_seed": 0,
                }
            )

            rows.append(s)

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
                            threshold[
                                "risk_uncertainty"
                            ]
                        ),
                        target_rate=(
                            audit_rate
                        ),
                        strategy="random",
                    )
                )

                global_params = (
                    tune_global_controller(
                        tuning_df=tune,
                        base_threshold=(
                            base_threshold
                        ),
                        audit_policy=(
                            audit_policy
                        ),
                    )
                )

                online_lr = (
                    tune_calibrator_learning_rate(
                        tuning_df=tune,
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

                beta = tune_beta(
                    tuning_df=tune,
                    calibrator=calibrator,
                    audit_rate=audit_rate,
                    quick=args.quick,
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
                        "residual_beta": (
                            beta
                        ),
                    }
                )

                for audit_seed in range(
                    audit_seeds
                ):

                    seq = replay_sequence(
                        test_ids,
                        repeats=(
                            deployment_repeats
                        ),
                        seed=(
                            10000
                            + audit_seed
                        ),
                    )

                    # ---------------------------------------------
                    # GLOBAL EWMA
                    # ---------------------------------------------

                    s, _ = simulate_stream(
                        deployment=test,
                        sequence=seq,
                        method=(
                            "global_ewma"
                        ),
                        epsilon=RISK_TARGET,
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

                    s.update(
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

                    rows.append(s)

                    # ---------------------------------------------
                    # OLD PREDICTED BUDGET
                    # ---------------------------------------------

                    s, _ = simulate_stream(
                        deployment=test,
                        sequence=seq,
                        method=(
                            "calibrated_budget"
                        ),
                        epsilon=RISK_TARGET,
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

                    s.update(
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

                    rows.append(s)

                    # ---------------------------------------------
                    # NEW STABLE METHODS
                    # ---------------------------------------------

                    for method in [
                        "verified_replacement_budget",
                        "verified_replacement_budget_residual",
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

                        s, trace = (
                            simulate_stable_stream(
                                deployment=test,
                                sequence=seq,
                                calibrator_template=(
                                    calibrator
                                ),
                                audit_rate=(
                                    audit_rate
                                ),
                                residual_beta=(
                                    beta
                                ),
                                method=method,
                                seed=(
                                    40000
                                    + audit_seed
                                ),
                                save_trace=(
                                    save_trace
                                ),
                            )
                        )

                        s.update(
                            {
                                "delta": delta,
                                "heldout_task": (
                                    heldout_task
                                ),
                                "method": method,
                                "audit_rate_target": (
                                    audit_rate
                                ),
                                "audit_seed": (
                                    audit_seed
                                ),
                            }
                        )

                        rows.append(s)

                        if trace is not None:

                            trace["delta"] = (
                                delta
                            )

                            trace[
                                "heldout_task"
                            ] = heldout_task

                            trace[
                                "audit_rate_target"
                            ] = audit_rate

                            traces.append(
                                trace
                            )

    # =================================================================
    # OUTPUT
    # =================================================================

    seeds = pd.DataFrame(rows)

    common = [
        "delta",
        "heldout_task",
        "method",
        "audit_rate_target",
    ]

    per_task = (
        seeds.groupby(
            common,
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
            audit_residual_mean=(
                "audit_residual_mean",
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
            mean_audit_residual=(
                "audit_residual_mean",
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
        / "stable_feedback_seed_results.csv",
        index=False,
    )

    per_task.to_csv(
        outdir
        / "stable_feedback_by_task.csv",
        index=False,
    )

    overall.to_csv(
        outdir
        / "stable_feedback_overall.csv",
        index=False,
    )

    pd.DataFrame(
        tuning_rows
    ).to_csv(
        outdir
        / "stable_feedback_hyperparameters.csv",
        index=False,
    )

    if traces:

        pd.concat(
            traces,
            ignore_index=True,
        ).to_csv(
            outdir
            / "stable_feedback_trace.csv",
            index=False,
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "STABLE VERIFIED QUALITY FEEDBACK"
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
        "\nEXPERIMENT 09C: PASS"
    )


if __name__ == "__main__":
    main()
