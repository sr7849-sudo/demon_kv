"""
EXPERIMENT 8C
=============

Quality-risk feedback under unseen-task distribution shift.

============================================================
MAIN QUESTION
============================================================

Suppose RiskKV is calibrated on several workloads but deployment moves
to a previously unseen LongBench task.

Does sparse runtime feedback help drive observed violation rate back
toward the requested risk target?

============================================================
EXPERIMENT DESIGN
============================================================

For each held-out LongBench task:

    1. Train the risk predictor only on OTHER tasks.

    2. Reserve part of the source tasks for threshold calibration.

    3. Deploy on the completely held-out task.

    4. Compare:

         fixed threshold

         PI feedback with:
             1% audit
             5% audit
            10% audit

An "audit" means that, in the controlled benchmark experiment, we use
the known FullKV/reference outcome to reveal whether the selected
compressed policy actually violated:

    D(r,p) > delta

At real deployment this could correspond to selective FullKV replay,
a downstream success signal, or another trustworthy quality label.

The benchmark oracle labels are used here ONLY AFTER the decision,
which simulates delayed runtime feedback.

============================================================
WHY REQUEST-LEVEL FEEDBACK
============================================================

The controller updates the threshold for FUTURE requests.

It does not undo or rewrite the request that already executed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
)

from riskkv.feedback_control import (
    PIDController,
)

from riskkv.synthesis_eval import (
    calibrate_threshold,
)


ORACLE = (
    "experiments/full_oracle_qwen3_8b/"
    "oracle_longbench_qwen3_8b.csv"
)


OUTDIR = Path(
    "experiments/full_oracle_qwen3_8b/"
    "analysis/24_quality_feedback_shift"
)


DELTAS = [
    0.01,
    0.05,
    0.10,
]

RISK_TARGET = 0.05

AUDIT_RATES = [
    0.00,
    0.01,
    0.05,
    0.10,
]

AUDIT_SEEDS = 30

ENSEMBLE_SIZE = 5


NUMERIC_PREFILL = [
    "context_tokens",
    "prefill_seconds",
    "last_logit_entropy",
    "last_logit_top1",
    "last_logit_margin",
    "k_l2_mean",
    "k_l2_std_layers",
    "k_l2_max",
    "v_l2_mean",
    "v_l2_std_layers",
    "v_l2_max",
    "k_std_mean",
    "k_std_std_layers",
    "k_std_max",
    "v_std_mean",
    "v_std_std_layers",
    "v_std_max",
    "k_maxabs_mean",
    "k_maxabs_std_layers",
    "k_maxabs_max",
    "v_maxabs_mean",
    "v_maxabs_std_layers",
    "v_maxabs_max",
    "prefill_nll",
    "prefill_perplexity_capped",
]


def build_features(
    df,
    use_prefill=True,
    columns=None,
):
    """
    Policy-conditioned predictor input.

    Policy identity is necessary because RiskKV is predicting:

        safety(request, policy)

    rather than just:

        safety(request)

    The prefill feature set mirrors the earlier RiskKV prediction
    experiments so that the feedback study does not quietly introduce
    a completely different predictor.
    """

    x = pd.get_dummies(
        df[
            [
                "policy"
            ]
        ],
        columns=[
            "policy"
        ],
        dtype=float,
    )

    if (
        "memory_ratio"
        in df.columns
    ):
        x[
            "memory_ratio"
        ] = pd.to_numeric(
            df[
                "memory_ratio"
            ],
            errors="coerce",
        )

    if use_prefill:
        for col in (
            NUMERIC_PREFILL
        ):
            if col in df.columns:
                x[col] = (
                    pd.to_numeric(
                        df[col],
                        errors="coerce",
                    )
                )

    x = x.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    x = x.fillna(
        x.median(
            numeric_only=True
        )
    )

    if columns is not None:
        x = x.reindex(
            columns=columns,
            fill_value=0.0,
        )

    return x


def split_source_requests(
    source,
    seed=42,
    calibration_fraction=0.20,
):
    """
    Request-level split.

    All seven policy rows for one request remain in the same split,
    avoiding request leakage between predictor training and threshold
    calibration.
    """

    ids = (
        source[
            "request_id"
        ]
        .drop_duplicates()
        .to_numpy()
    )

    rng = np.random.default_rng(
        seed
    )

    rng.shuffle(ids)

    n_cal = max(
        1,
        int(
            round(
                calibration_fraction
                * len(ids)
            )
        ),
    )

    cal_ids = set(
        ids[:n_cal]
    )

    train = source[
        ~source[
            "request_id"
        ].isin(
            cal_ids
        )
    ].copy()

    cal = source[
        source[
            "request_id"
        ].isin(
            cal_ids
        )
    ].copy()

    return train, cal


def train_ensemble(
    train,
    evaluation_sets,
    use_prefill,
):
    """
    Bootstrap request-level ensembles.

    Ensemble disagreement becomes predictive uncertainty.

    Each member receives a different bootstrap sample of TRAINING
    requests only.
    """

    train_ids = (
        train[
            "request_id"
        ]
        .drop_duplicates()
        .to_numpy()
    )

    predictions = {
        name: []
        for name in (
            evaluation_sets
        )
    }

    feature_columns = None

    for member in range(
        ENSEMBLE_SIZE
    ):

        rng = (
            np.random.default_rng(
                1000
                + member
            )
        )

        bootstrap_ids = (
            rng.choice(
                train_ids,
                size=len(
                    train_ids
                ),
                replace=True,
            )
        )

        boot = pd.concat(
            [
                train[
                    train[
                        "request_id"
                    ]
                    == rid
                ]
                for rid
                in bootstrap_ids
            ],
            ignore_index=True,
        )

        X_train = build_features(
            boot,
            use_prefill=(
                use_prefill
            ),
        )

        feature_columns = (
            X_train.columns
        )

        model = (
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=(
                    1000
                    + member
                ),
            )
        )

        model.fit(
            X_train,
            boot[
                "safe"
            ],
        )

        for name, df in (
            evaluation_sets.items()
        ):

            X = build_features(
                df,
                use_prefill=(
                    use_prefill
                ),
                columns=(
                    feature_columns
                ),
            )

            p = (
                model.predict_proba(
                    X
                )[:, 1]
            )

            predictions[
                name
            ].append(p)

    outputs = {}

    for name, df in (
        evaluation_sets.items()
    ):

        stack = np.vstack(
            predictions[name]
        )

        out = df.copy()

        out[
            "prob_safe_mean"
        ] = stack.mean(
            axis=0
        )

        out[
            "risk_uncertainty"
        ] = stack.std(
            axis=0,
            ddof=0,
        )

        outputs[name] = out

    return outputs


def fullkv_row(
    g,
):
    """
    Explicit FullKV fallback.

    By definition:
        degradation(FullKV) = 0
        memory ratio = 1
    """

    r = g.iloc[0].copy()

    r[
        "policy"
    ] = "full"

    r[
        "memory_ratio"
    ] = 1.0

    r[
        "degradation"
    ] = 0.0

    r[
        "safe"
    ] = 1

    r[
        "prob_safe_mean"
    ] = 1.0

    r[
        "risk_uncertainty"
    ] = 0.0

    return r


def simulate_stream(
    deployment,
    base_threshold,
    audit_rate,
    kp,
    ki,
    seed,
    risk_target,
):
    """
    Request-level quality-feedback loop.

    Threshold at request t:

        tau_t =
            base_threshold
            + PI_correction_t

    If audited outcomes show violations above the desired risk target,
    the PI correction becomes positive and tau increases.

    Higher tau:
        fewer compressed policies accepted
        -> more conservative controller.

    Lower tau:
        more compression opportunity.
    """

    rng = np.random.default_rng(
        seed
    )

    request_ids = (
        deployment[
            "request_id"
        ]
        .drop_duplicates()
        .to_numpy()
    )

    rng.shuffle(
        request_ids
    )

    ctrl = PIDController(
        kp=kp,
        ki=ki,
        kd=0.0,
        output_min=-0.40,
        output_max=0.40,
        integral_limit=5.0,
    )

    audit_violations = []
    rows = []

    for step, rid in enumerate(
        request_ids
    ):

        g = deployment[
            deployment[
                "request_id"
            ]
            == rid
        ].copy()

        tau = float(
            np.clip(
                base_threshold
                + ctrl.value,
                0.0,
                1.000001,
            )
        )

        accepted = g[
            g[
                "prob_safe_mean"
            ]
            >= tau
        ]

        if len(accepted):
            chosen = (
                accepted.sort_values(
                    [
                        "memory_ratio",
                        "prob_safe_mean",
                        "policy",
                    ],
                    ascending=[
                        True,
                        False,
                        True,
                    ],
                )
                .iloc[0]
            )
        else:
            chosen = (
                fullkv_row(
                    g
                )
            )

        violation = (
            chosen[
                "degradation"
            ]
            > chosen[
                "delta"
            ]
            + 1e-12
        )

        # --------------------------------------------------------
        # SPARSE AUDIT
        #
        # Only compressed decisions are useful quality audits.
        #
        # Random auditing gives an approximately unbiased estimate of
        # actual selected-policy violation frequency.
        # --------------------------------------------------------

        audited = False

        if (
            chosen[
                "policy"
            ]
            != "full"
            and rng.random()
            < audit_rate
        ):

            audited = True

            audit_violations.append(
                float(
                    violation
                )
            )

            observed_rate = float(
                np.mean(
                    audit_violations
                )
            )

            error = (
                observed_rate
                - risk_target
            )

            ctrl.update(
                error
            )

        rows.append(
            {
                "step": step,
                "request_id": rid,
                "policy": (
                    chosen[
                        "policy"
                    ]
                ),
                "threshold": tau,
                "controller_correction": (
                    ctrl.value
                ),
                "audited": audited,
                "violation": (
                    violation
                ),
                "degradation": (
                    chosen[
                        "degradation"
                    ]
                ),
                "memory_saved": (
                    1.0
                    - chosen[
                        "memory_ratio"
                    ]
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


def tune_pi(
    calibration,
    base_threshold,
    risk_target,
):
    """
    Tune the PI gains ONLY on source-task calibration requests.

    During gain tuning we allow 100% calibration feedback because the
    goal is to choose controller dynamics offline.

    Final held-out-task evaluation still uses sparse 1/5/10% auditing.
    """

    gain_grid = [
        (kp, ki)
        for kp in [
            0.05,
            0.10,
            0.20,
            0.30,
            0.50,
        ]
        for ki in [
            0.005,
            0.01,
            0.02,
            0.05,
        ]
    ]

    rows = []

    for kp, ki in gain_grid:

        x = simulate_stream(
            calibration,
            base_threshold=(
                base_threshold
            ),
            audit_rate=1.0,
            kp=kp,
            ki=ki,
            seed=123,
            risk_target=(
                risk_target
            ),
        )

        v = (
            x[
                "violation"
            ].mean()
        )

        saving = (
            x[
                "memory_saved"
            ].mean()
        )

        rows.append(
            {
                "kp": kp,
                "ki": ki,
                "violation_rate": v,
                "violation_excess": max(
                    0.0,
                    v
                    - risk_target,
                ),
                "memory_saved": (
                    saving
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    # First satisfy the risk target if possible,
    # then maximize memory savings.
    result = result.sort_values(
        [
            "violation_excess",
            "memory_saved",
            "violation_rate",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )

    return result.iloc[0]


def main():

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        ORACLE
    )

    # Risk predictor only needs compressed policies.
    df = df[
        df[
            "policy"
        ]
        != "full"
    ].copy()

    tasks = sorted(
        df[
            "task"
        ].unique()
    )

    summaries = []
    traces = []
    tuning_rows = []

    for delta in DELTAS:

        work = df.copy()

        work[
            "delta"
        ] = delta

        work[
            "safe"
        ] = (
            work[
                "degradation"
            ]
            <= delta + 1e-12
        ).astype(int)

        for heldout_task in tasks:

            source = work[
                work[
                    "task"
                ]
                != heldout_task
            ].copy()

            heldout = work[
                work[
                    "task"
                ]
                == heldout_task
            ].copy()

            train, calibration = (
                split_source_requests(
                    source,
                    seed=42,
                )
            )

            outputs = train_ensemble(
                train=train,
                evaluation_sets={
                    "calibration": (
                        calibration
                    ),
                    "heldout": (
                        heldout
                    ),
                },
                use_prefill=True,
            )

            cal_pred = outputs[
                "calibration"
            ]

            test_pred = outputs[
                "heldout"
            ]

            initial_threshold = (
                calibrate_threshold(
                    cal_pred[
                        "safe"
                    ],
                    cal_pred[
                        "prob_safe_mean"
                    ],
                    RISK_TARGET,
                )
            )

            best = tune_pi(
                cal_pred,
                base_threshold=(
                    initial_threshold
                ),
                risk_target=(
                    RISK_TARGET
                ),
            )

            kp = float(
                best["kp"]
            )

            ki = float(
                best["ki"]
            )

            tuning_rows.append(
                {
                    "delta": delta,
                    "heldout_task": (
                        heldout_task
                    ),
                    "initial_threshold": (
                        initial_threshold
                    ),
                    "kp": kp,
                    "ki": ki,
                }
            )

            for audit_rate in (
                AUDIT_RATES
            ):

                seed_summaries = []

                n_seeds = (
                    1
                    if audit_rate == 0
                    else AUDIT_SEEDS
                )

                for audit_seed in range(
                    n_seeds
                ):

                    x = simulate_stream(
                        test_pred,
                        base_threshold=(
                            initial_threshold
                        ),
                        audit_rate=(
                            audit_rate
                        ),
                        kp=(
                            0.0
                            if audit_rate
                            == 0
                            else kp
                        ),
                        ki=(
                            0.0
                            if audit_rate
                            == 0
                            else ki
                        ),
                        seed=(
                            10000
                            + audit_seed
                        ),
                        risk_target=(
                            RISK_TARGET
                        ),
                    )

                    x[
                        "delta"
                    ] = delta

                    x[
                        "heldout_task"
                    ] = heldout_task

                    x[
                        "audit_rate"
                    ] = audit_rate

                    x[
                        "audit_seed"
                    ] = audit_seed

                    traces.append(x)

                    seed_summaries.append(
                        {
                            "violation_rate": (
                                x[
                                    "violation"
                                ].mean()
                            ),
                            "memory_saved": (
                                x[
                                    "memory_saved"
                                ].mean()
                            ),
                            "fullkv_rate": (
                                x[
                                    "fullkv"
                                ].mean()
                            ),
                            "actual_audit_rate": (
                                x[
                                    "audited"
                                ].mean()
                            ),
                        }
                    )

                seed_df = pd.DataFrame(
                    seed_summaries
                )

                summaries.append(
                    {
                        "delta": delta,
                        "heldout_task": (
                            heldout_task
                        ),
                        "audit_rate_target": (
                            audit_rate
                        ),
                        "mean_violation_rate": (
                            seed_df[
                                "violation_rate"
                            ].mean()
                        ),
                        "std_violation_rate": (
                            seed_df[
                                "violation_rate"
                            ].std(
                                ddof=1
                            )
                            if len(
                                seed_df
                            ) > 1
                            else 0.0
                        ),
                        "mean_memory_saved": (
                            seed_df[
                                "memory_saved"
                            ].mean()
                        ),
                        "mean_fullkv_rate": (
                            seed_df[
                                "fullkv_rate"
                            ].mean()
                        ),
                        "mean_actual_audit_rate": (
                            seed_df[
                                "actual_audit_rate"
                            ].mean()
                        ),
                    }
                )

    summary = pd.DataFrame(
        summaries
    )

    tuning = pd.DataFrame(
        tuning_rows
    )

    trace = pd.concat(
        traces,
        ignore_index=True,
    )

    overall = (
        summary.groupby(
            [
                "delta",
                "audit_rate_target",
            ],
            as_index=False,
        )
        .agg(
            mean_violation_rate=(
                "mean_violation_rate",
                "mean",
            ),
            mean_memory_saved=(
                "mean_memory_saved",
                "mean",
            ),
            mean_fullkv_rate=(
                "mean_fullkv_rate",
                "mean",
            ),
            mean_actual_audit_rate=(
                "mean_actual_audit_rate",
                "mean",
            ),
        )
    )

    summary.to_csv(
        OUTDIR
        / "quality_feedback_by_task.csv",
        index=False,
    )

    overall.to_csv(
        OUTDIR
        / "quality_feedback_overall.csv",
        index=False,
    )

    tuning.to_csv(
        OUTDIR
        / "quality_feedback_pi_gains.csv",
        index=False,
    )

    trace.to_csv(
        OUTDIR
        / "quality_feedback_trace.csv",
        index=False,
    )

    print(
        "\nQUALITY FEEDBACK UNDER TASK SHIFT"
    )

    print(
        overall.round(5)
        .to_string(
            index=False
        )
    )

    print(
        "\nPER-TASK RESULTS"
    )

    print(
        summary.round(5)
        .to_string(
            index=False
        )
    )

    print(
        "\nEXPERIMENT 8C: PASS"
    )


if __name__ == "__main__":
    main()
