"""
EXPERIMENT 9A
=============

Verified online quality feedback for RiskKV.

======================================================================
WHAT THIS EXPERIMENT REPLACES
======================================================================

Previous exploratory methods:

    - global cumulative-mean PI
    - KMeans context-conditioned PI

were useful to establish that feedback can matter, but they had
methodological limitations.

This experiment evaluates a more principled architecture.

======================================================================
METHODS
======================================================================

A) fixed

    Base RiskKV predictor + frozen source-calibrated threshold.

    No runtime quality feedback.


B) global_ewma

    Strong global-feedback baseline.

    Uses:
        - one shared threshold correction;
        - exponentially weighted feedback;
        - inverse-propensity normalization.

    This is intentionally stronger than the old cumulative-mean PI.


C) online_calibrator

    Shared contextual probability recalibrator.

    Runtime verified outcomes update:

        P_safe_hat
            ->
        P_safe_calibrated

    The decision threshold remains fixed.


D) calibrated_budget

    Proposed full quality-feedback controller.

    Uses:
        online contextual calibration
            +
        explicit violation-budget accounting.

    No PI/PID is used for quality-risk control.

======================================================================
AUDITING
======================================================================

For feedback methods:

    target audit rates:
        1%
        5%
        10%

    strategies:
        random
        uncertainty

The uncertainty strategy always retains a random audit floor.

======================================================================
DISTRIBUTION SHIFT
======================================================================

Leave one LongBench task completely unseen during:

    - base-risk training;
    - contextual-calibrator fitting;
    - threshold calibration;
    - controller hyperparameter tuning.

The held-out task is then replayed as a longer deployment stream.

Why replay?
-----------
Each held-out task contains only ~51-60 unique requests.

At a 1% audit rate this would produce fewer than one verified feedback
event on average, making online feedback statistically meaningless.

Therefore each held-out workload is replayed 10 times in shuffled order.

IMPORTANT:
This is a controlled bootstrap-style deployment replay, NOT 600 unique
test examples.

Every result must be described accordingly.

======================================================================
SOURCE DATA PARTITION
======================================================================

Source tasks are split at REQUEST level into four disjoint subsets:

    65%  base risk-model training

    15%  contextual-calibrator fitting

    10%  threshold calibration

    10%  controller/learning-rate tuning

This directly prevents the old issue where controller parameters were
tuned under conditions different from deployment.

Hyperparameters are tuned under the SAME sparse audit rate and SAME
audit strategy subsequently used on the held-out deployment replay.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
)

from sklearn.metrics import (
    brier_score_loss,
)

from riskkv.synthesis_eval import (
    calibrate_threshold,
)

from riskkv.verified_quality_feedback import (
    AuditPolicy,
    ContextualOnlineCalibrator,
    GlobalEWMAThresholdController,
    ViolationBudgetManager,
    expected_calibration_error,
)


# ======================================================================
# PATHS
# ======================================================================

ORACLE = (
    "experiments/full_oracle_qwen3_8b/"
    "oracle_longbench_qwen3_8b.csv"
)

OUTDIR = Path(
    "experiments/full_oracle_qwen3_8b/"
    "analysis/28_verified_quality_feedback"
)


# ======================================================================
# MAIN SETTINGS
# ======================================================================

DELTAS = [
    0.01,
    0.05,
    0.10,
]

RISK_TARGET = 0.05

AUDIT_RATES = [
    0.01,
    0.05,
    0.10,
]

AUDIT_STRATEGIES = [
    "random",
    "uncertainty",
]


# Repeat each held-out request 10 times to create a controlled
# long-running deployment trace for feedback evaluation.
DEPLOYMENT_REPEATS = 10

# Controller tuning also uses replay because sparse 1% auditing on
# ~20 source tuning requests would otherwise provide almost no labels.
TUNING_REPEATS = 10


AUDIT_SEEDS = 20

TUNING_SEEDS = [
    101,
    202,
    303,
]


ENSEMBLE_SIZE = 5


# ======================================================================
# HYPERPARAMETER GRIDS
# ======================================================================

# Strong global baseline.
GLOBAL_KP = [
    0.25,
    0.50,
    1.00,
]

GLOBAL_KI = [
    0.00,
    0.01,
    0.05,
]

GLOBAL_BETA = [
    0.05,
    0.10,
]


# Online contextual-calibration learning rates.
CALIBRATOR_LR = [
    0.001,
    0.003,
    0.010,
    0.030,
]


# ======================================================================
# BASE RISK-PREDICTOR FEATURES
# ======================================================================

PREDICTOR_FEATURES = [
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


# ======================================================================
# REQUEST-LEVEL FOUR-WAY SOURCE SPLIT
# ======================================================================

def split_source_four_way(
    source,
    seed=42,
):
    """
    Strict request-level split.

    Policy rows belonging to the same request never cross subsets.
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

    n = len(ids)

    n_train = int(
        round(
            0.65 * n
        )
    )

    n_calfit = int(
        round(
            0.15 * n
        )
    )

    n_threshold = int(
        round(
            0.10 * n
        )
    )

    train_ids = set(
        ids[
            :n_train
        ]
    )

    calfit_ids = set(
        ids[
            n_train
            :n_train
            + n_calfit
        ]
    )

    threshold_ids = set(
        ids[
            n_train
            + n_calfit
            :n_train
            + n_calfit
            + n_threshold
        ]
    )

    tuning_ids = set(
        ids[
            n_train
            + n_calfit
            + n_threshold
            :
        ]
    )

    def subset(
        selected,
    ):
        return source[
            source[
                "request_id"
            ].isin(
                selected
            )
        ].copy()

    return (
        subset(
            train_ids
        ),
        subset(
            calfit_ids
        ),
        subset(
            threshold_ids
        ),
        subset(
            tuning_ids
        ),
    )


# ======================================================================
# BASE RISK MODEL
# ======================================================================

def build_risk_features(
    df,
    columns=None,
):
    """
    Policy-conditioned RiskKV predictor.

    The online calibrator does NOT replace this model.
    It only recalibrates its output probabilities.
    """

    X = pd.get_dummies(
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

        X[
            "memory_ratio"
        ] = pd.to_numeric(
            df[
                "memory_ratio"
            ],
            errors="coerce",
        )

    for col in (
        PREDICTOR_FEATURES
    ):

        if col in df.columns:

            X[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    X = X.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    X = X.fillna(
        X.median(
            numeric_only=True
        )
    )

    if columns is not None:

        X = X.reindex(
            columns=columns,
            fill_value=0.0,
        )

    return X


def train_risk_ensemble(
    train,
    evaluation_sets,
):
    """
    Request-level bootstrap ensemble.

    Outputs:
        prob_safe_mean
        risk_uncertainty
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

    for member in range(
        ENSEMBLE_SIZE
    ):

        rng = np.random.default_rng(
            1000
            + member
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

        X_train = (
            build_risk_features(
                boot
            )
        )

        columns = (
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

            X = (
                build_risk_features(
                    df,
                    columns=columns,
                )
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

        outputs[
            name
        ] = out

    return outputs


# ======================================================================
# REQUEST GROUPS / REPLAY
# ======================================================================

def request_groups(
    df,
):
    """
    Cache request-policy groups.

    This avoids repeatedly filtering the large DataFrame inside every
    feedback simulation step.
    """

    return {
        rid: g.reset_index(
            drop=True
        )
        for rid, g
        in df.groupby(
            "request_id",
            sort=False,
        )
    }


def replay_sequence(
    request_ids,
    repeats,
    seed,
):
    """
    Controlled bootstrap-style replay.

    Every unique request appears exactly `repeats` times, but ordering
    is randomized.
    """

    ids = list(
        request_ids
    ) * int(
        repeats
    )

    rng = np.random.default_rng(
        seed
    )

    rng.shuffle(ids)

    return ids


# ======================================================================
# FULLKV FALLBACK
# ======================================================================

def fullkv_row(
    g,
):
    """
    FullKV is safe by construction relative to the degradation
    definition used in RiskKV:

        D(full, full) = 0.

    This does NOT mean the FullKV answer is objectively correct.
    """

    row = g.iloc[0].copy()

    row[
        "policy"
    ] = "full"

    row[
        "memory_ratio"
    ] = 1.0

    row[
        "degradation"
    ] = 0.0

    row[
        "safe"
    ] = 1

    row[
        "prob_safe_mean"
    ] = 1.0

    row[
        "risk_uncertainty"
    ] = 0.0

    return row


# ======================================================================
# POLICY SELECTION HELPERS
# ======================================================================

def select_threshold_policy(
    g,
    probabilities,
    threshold,
):
    """
    Select the most memory-aggressive compressed configuration that
    exceeds the current safety-probability threshold.
    """

    work = g.copy()

    work[
        "_decision_probability"
    ] = probabilities

    feasible = work[
        work[
            "_decision_probability"
        ]
        >= threshold
    ]

    if len(feasible) == 0:

        return fullkv_row(
            g
        ), 1.0

    selected = (
        feasible.sort_values(
            [
                "memory_ratio",
                "_decision_probability",
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

    return (
        selected,
        float(
            selected[
                "_decision_probability"
            ]
        ),
    )


def select_budget_policy(
    g,
    calibrated_probabilities,
    allowed_risk,
):
    """
    Risk-budget controller.

    Choose the lowest-memory configuration whose calibrated predicted
    violation probability fits the CURRENT remaining risk allowance.

        1 - p_tilde <= epsilon_t
    """

    work = g.copy()

    work[
        "_calibrated_probability"
    ] = (
        calibrated_probabilities
    )

    work[
        "_predicted_risk"
    ] = (
        1.0
        - work[
            "_calibrated_probability"
        ]
    )

    feasible = work[
        work[
            "_predicted_risk"
        ]
        <= allowed_risk
        + 1e-12
    ]

    if len(feasible) == 0:

        return (
            fullkv_row(
                g
            ),
            1.0,
            0.0,
        )

    selected = (
        feasible.sort_values(
            [
                "memory_ratio",
                "_predicted_risk",
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
        selected,
        float(
            selected[
                "_calibrated_probability"
            ]
        ),
        float(
            selected[
                "_predicted_risk"
            ]
        ),
    )


# ======================================================================
# FEEDBACK STREAM
# ======================================================================

def simulate_stream(
    deployment,
    sequence,
    method,
    epsilon,
    audit_policy=None,
    base_threshold=None,
    calibrated_threshold=None,
    calibrator_template=None,
    calibrator_learning_rate=None,
    global_params=None,
    seed=0,
    save_trace=False,
):
    """
    Run one complete closed-loop deployment replay.

    ------------------------------------------------------------
    METHOD: fixed
    ------------------------------------------------------------

        base predictor
            +
        frozen threshold

    No feedback.


    ------------------------------------------------------------
    METHOD: global_ewma
    ------------------------------------------------------------

        base predictor
            +
        one global EWMA/IPW feedback state
            +
        adaptive scalar threshold


    ------------------------------------------------------------
    METHOD: online_calibrator
    ------------------------------------------------------------

        base predictor
            +
        shared contextual online calibrator
            +
        frozen calibrated threshold


    ------------------------------------------------------------
    METHOD: calibrated_budget
    ------------------------------------------------------------

        base predictor
            +
        online calibrator
            +
        explicit remaining violation budget

    This is the proposed full quality-feedback architecture.
    """

    groups = request_groups(
        deployment
    )

    rng = np.random.default_rng(
        seed
    )

    calibrator = None

    if method in [
        "online_calibrator",
        "calibrated_budget",
    ]:

        calibrator = (
            calibrator_template
            .clone()
        )

    global_controller = None

    if method == "global_ewma":

        global_controller = (
            GlobalEWMAThresholdController(
                epsilon=epsilon,
                **global_params,
            )
        )

    budget = None

    if method == "calibrated_budget":

        budget = (
            ViolationBudgetManager(
                epsilon=epsilon,
                horizon=len(
                    sequence
                ),
            )
        )

    violations = 0
    memory_saved = []
    fullkv_count = 0

    audits = 0
    compressed_count = 0

    audit_probability_sum = 0.0

    trace_rows = []

    for step, rid in enumerate(
        sequence
    ):

        g = groups[
            rid
        ]

        # --------------------------------------------------------
        # DECISION
        # --------------------------------------------------------

        if method == "fixed":

            probabilities = (
                g[
                    "prob_safe_mean"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            chosen, selected_prob = (
                select_threshold_policy(
                    g,
                    probabilities,
                    base_threshold,
                )
            )

            predicted_risk = (
                0.0
                if chosen[
                    "policy"
                ]
                == "full"
                else (
                    1.0
                    - selected_prob
                )
            )

            threshold_used = (
                base_threshold
            )

            allowed_risk = np.nan


        elif method == "global_ewma":

            probabilities = (
                g[
                    "prob_safe_mean"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            threshold_used = float(
                np.clip(
                    base_threshold
                    + global_controller.correction,
                    0.0,
                    1.000001,
                )
            )

            chosen, selected_prob = (
                select_threshold_policy(
                    g,
                    probabilities,
                    threshold_used,
                )
            )

            predicted_risk = (
                0.0
                if chosen[
                    "policy"
                ]
                == "full"
                else (
                    1.0
                    - selected_prob
                )
            )

            allowed_risk = np.nan


        elif method == "online_calibrator":

            probabilities = (
                calibrator
                .predict_proba(
                    g
                )
            )

            threshold_used = (
                calibrated_threshold
            )

            chosen, selected_prob = (
                select_threshold_policy(
                    g,
                    probabilities,
                    threshold_used,
                )
            )

            predicted_risk = (
                0.0
                if chosen[
                    "policy"
                ]
                == "full"
                else (
                    1.0
                    - selected_prob
                )
            )

            allowed_risk = np.nan


        elif method == "calibrated_budget":

            probabilities = (
                calibrator
                .predict_proba(
                    g
                )
            )

            allowed_risk = (
                budget.allowance(
                    step
                )
            )

            (
                chosen,
                selected_prob,
                predicted_risk,
            ) = (
                select_budget_policy(
                    g,
                    probabilities,
                    allowed_risk,
                )
            )

            threshold_used = np.nan


        else:

            raise ValueError(
                "Unknown method: "
                + method
            )

        policy = (
            chosen[
                "policy"
            ]
        )

        is_full = (
            policy
            == "full"
        )

        if is_full:

            fullkv_count += 1

        else:

            compressed_count += 1

        actual_violation = bool(
            chosen[
                "degradation"
            ]
            > chosen[
                "delta"
            ]
            + 1e-12
        )

        violations += int(
            actual_violation
        )

        memory_saved.append(
            1.0
            - float(
                chosen[
                    "memory_ratio"
                ]
            )
        )

        # --------------------------------------------------------
        # VERIFIED AUDIT
        # --------------------------------------------------------

        audited = False
        q = 0.0

        ipw_weight = 0.0
        gradient_norm = 0.0

        # FullKV quality degradation is defined as zero relative to
        # itself, so additional quality auditing gives no useful
        # compression-risk information.
        if (
            not is_full
            and method
            != "fixed"
        ):

            q = (
                audit_policy
                .probability(
                    chosen[
                        "risk_uncertainty"
                    ]
                )
            )

            audit_probability_sum += q

            audited = bool(
                rng.random()
                < q
            )

            if audited:

                audits += 1

                y_safe = (
                    0
                    if actual_violation
                    else 1
                )

                if (
                    method
                    == "global_ewma"
                ):

                    global_controller.update(
                        violation=(
                            actual_violation
                        ),
                        audit_probability=q,
                    )

                elif method in [
                    "online_calibrator",
                    "calibrated_budget",
                ]:

                    diagnostic = (
                        calibrator.update(
                            row=chosen,
                            y_safe=y_safe,
                            audit_probability=q,
                            learning_rate=(
                                calibrator_learning_rate
                            ),
                        )
                    )

                    ipw_weight = (
                        diagnostic[
                            "ipw_weight"
                        ]
                    )

                    gradient_norm = (
                        diagnostic[
                            "gradient_norm"
                        ]
                    )

        # --------------------------------------------------------
        # VIOLATION-BUDGET UPDATE
        # --------------------------------------------------------

        # Consume the PRE-UPDATE calibrated expected risk associated
        # with the policy actually executed.
        if (
            method
            == "calibrated_budget"
        ):

            budget.consume(
                predicted_risk
            )

        # --------------------------------------------------------
        # TRACE
        # --------------------------------------------------------

        if save_trace:

            trace_rows.append(
                {
                    "step": step,
                    "request_id": rid,
                    "task": (
                        chosen[
                            "task"
                        ]
                    ),
                    "method": method,
                    "policy": policy,

                    "actual_violation": (
                        actual_violation
                    ),

                    "degradation": (
                        chosen[
                            "degradation"
                        ]
                    ),

                    "selected_probability": (
                        selected_prob
                    ),

                    "predicted_risk": (
                        predicted_risk
                    ),

                    "threshold": (
                        threshold_used
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

                    "ipw_weight": (
                        ipw_weight
                    ),

                    "gradient_norm": (
                        gradient_norm
                    ),

                    "memory_saved": (
                        memory_saved[-1]
                    ),

                    "fullkv": (
                        is_full
                    ),

                    "global_rate_estimate": (
                        np.nan
                        if global_controller
                        is None
                        else global_controller
                        .estimated_rate
                    ),

                    "global_correction": (
                        np.nan
                        if global_controller
                        is None
                        else global_controller
                        .correction
                    ),

                    "remaining_risk_budget": (
                        np.nan
                        if budget is None
                        else budget
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

    compressed_audit_rate = (
        audits
        / compressed_count
        if compressed_count
        else 0.0
    )

    all_request_audit_rate = (
        audits
        / n
    )

    summary = {
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
            - epsilon,
        ),

        "absolute_target_error": (
            abs(
                violation_rate
                - epsilon
            )
        ),

        "mean_memory_saved": float(
            np.mean(
                memory_saved
            )
        ),

        "fullkv_rate": (
            fullkv_count
            / n
        ),

        "audit_count": (
            audits
        ),

        "audit_rate_all_requests": (
            all_request_audit_rate
        ),

        "audit_rate_compressed": (
            compressed_audit_rate
        ),

        # If each verified audit required one extra quality-evaluation
        # execution, this is the corresponding execution-count proxy.
        # Actual latency/energy cost depends on the deployed verifier.
        "audit_overhead_generation_fraction": (
            all_request_audit_rate
        ),

        "expected_risk_consumed": (
            np.nan
            if budget is None
            else budget
            .consumed_expected_risk
        ),

        "initial_risk_budget": (
            np.nan
            if budget is None
            else budget
            .initial_budget
        ),

        "remaining_risk_budget": (
            np.nan
            if budget is None
            else budget
            .remaining_budget
        ),

        "actual_violation_budget_excess": max(
            0.0,
            violations
            - epsilon
            * n,
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
# HYPERPARAMETER OBJECTIVE
# ======================================================================

def score_candidate(
    summaries,
):
    """
    Controller tuning objective.

    Priority:

        1. avoid exceeding desired violation rate;
        2. maximize memory saving;
        3. minimize absolute deviation from target.

    This avoids choosing a controller that achieves zero violations
    simply by using FullKV everywhere.
    """

    df = pd.DataFrame(
        summaries
    )

    mean_violation = float(
        df[
            "violation_rate"
        ].mean()
    )

    mean_saving = float(
        df[
            "mean_memory_saved"
        ].mean()
    )

    violation_excess = max(
        0.0,
        mean_violation
        - RISK_TARGET,
    )

    absolute_error = abs(
        mean_violation
        - RISK_TARGET
    )

    return (
        violation_excess,
        -mean_saving,
        absolute_error,
    )


# ======================================================================
# TUNE STRONG GLOBAL BASELINE
# ======================================================================

def tune_global_controller(
    tuning_df,
    base_threshold,
    audit_policy,
):
    groups = request_groups(
        tuning_df
    )

    ids = list(
        groups.keys()
    )

    candidates = []

    for kp in GLOBAL_KP:
        for ki in GLOBAL_KI:
            for beta in GLOBAL_BETA:

                seed_summaries = []

                params = {
                    "kp": kp,
                    "ki": ki,
                    "beta": beta,
                }

                for seed in (
                    TUNING_SEEDS
                ):

                    sequence = (
                        replay_sequence(
                            ids,
                            repeats=(
                                TUNING_REPEATS
                            ),
                            seed=seed,
                        )
                    )

                    summary, _ = (
                        simulate_stream(
                            deployment=(
                                tuning_df
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
                                params
                            ),
                            seed=(
                                seed
                                + 10000
                            ),
                        )
                    )

                    seed_summaries.append(
                        summary
                    )

                candidates.append(
                    (
                        score_candidate(
                            seed_summaries
                        ),
                        params,
                    )
                )

    candidates.sort(
        key=lambda x:
            x[0]
    )

    return candidates[
        0
    ][1]


# ======================================================================
# TUNE ONLINE CALIBRATION LEARNING RATE
# ======================================================================

def tune_calibrator_learning_rate(
    tuning_df,
    calibrated_threshold,
    calibrator_template,
    audit_policy,
):
    groups = request_groups(
        tuning_df
    )

    ids = list(
        groups.keys()
    )

    candidates = []

    for learning_rate in (
        CALIBRATOR_LR
    ):

        seed_summaries = []

        for seed in (
            TUNING_SEEDS
        ):

            sequence = (
                replay_sequence(
                    ids,
                    repeats=(
                        TUNING_REPEATS
                    ),
                    seed=seed,
                )
            )

            summary, _ = (
                simulate_stream(
                    deployment=(
                        tuning_df
                    ),
                    sequence=(
                        sequence
                    ),
                    method=(
                        "online_calibrator"
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
                        calibrator_template
                    ),
                    calibrator_learning_rate=(
                        learning_rate
                    ),
                    seed=(
                        seed
                        + 20000
                    ),
                )
            )

            seed_summaries.append(
                summary
            )

        candidates.append(
            (
                score_candidate(
                    seed_summaries
                ),
                learning_rate,
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

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        ORACLE
    )

    # Base risk model predicts compressed-policy safety.
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

    seed_rows = []
    hyperparameter_rows = []
    calibration_rows = []
    representative_traces = []


    # =================================================================
    # LEAVE-ONE-TASK-OUT DEPLOYMENT SHIFT
    # =================================================================

    for heldout_task in tasks:

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

        print(
            "\n"
            + "=" * 80
        )

        print(
            "HELD-OUT TASK:",
            heldout_task
        )

        print(
            "=" * 80
        )


        # =============================================================
        # DELTA
        # =============================================================

        for delta in DELTAS:

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


            # ---------------------------------------------------------
            # Four disjoint source subsets.
            # ---------------------------------------------------------

            (
                risk_train,
                calibrator_fit,
                threshold_cal,
                controller_tune,
            ) = split_source_four_way(
                source,
                seed=42,
            )


            # ---------------------------------------------------------
            # Train base risk predictor.
            # ---------------------------------------------------------

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


            # ---------------------------------------------------------
            # Frozen base threshold.
            # ---------------------------------------------------------

            base_threshold = (
                calibrate_threshold(
                    threshold_pred[
                        "safe"
                    ],
                    threshold_pred[
                        "prob_safe_mean"
                    ],
                    RISK_TARGET,
                )
            )


            # ---------------------------------------------------------
            # Fit shared contextual calibrator.
            # ---------------------------------------------------------

            calibrator = (
                ContextualOnlineCalibrator()
            )

            calibrator.fit(
                calfit_pred
            )

            threshold_calibrated_prob = (
                calibrator.predict_proba(
                    threshold_pred
                )
            )

            calibrated_threshold = (
                calibrate_threshold(
                    threshold_pred[
                        "safe"
                    ],
                    threshold_calibrated_prob,
                    RISK_TARGET,
                )
            )


            # ---------------------------------------------------------
            # Calibration diagnostics before deployment.
            # ---------------------------------------------------------

            heldout_initial_calibrated = (
                calibrator.predict_proba(
                    heldout_pred
                )
            )

            calibration_rows.append(
                {
                    "delta": delta,
                    "heldout_task": (
                        heldout_task
                    ),

                    "base_brier": (
                        brier_score_loss(
                            heldout_pred[
                                "safe"
                            ],
                            heldout_pred[
                                "prob_safe_mean"
                            ],
                        )
                    ),

                    "calibrated_brier": (
                        brier_score_loss(
                            heldout_pred[
                                "safe"
                            ],
                            heldout_initial_calibrated,
                        )
                    ),

                    "base_ece": (
                        expected_calibration_error(
                            heldout_pred[
                                "safe"
                            ],
                            heldout_pred[
                                "prob_safe_mean"
                            ],
                        )
                    ),

                    "calibrated_ece": (
                        expected_calibration_error(
                            heldout_pred[
                                "safe"
                            ],
                            heldout_initial_calibrated,
                        )
                    ),

                    "base_threshold": (
                        base_threshold
                    ),

                    "calibrated_threshold": (
                        calibrated_threshold
                    ),
                }
            )


            # ---------------------------------------------------------
            # FIXED BASELINE
            # ---------------------------------------------------------

            heldout_ids = (
                heldout_pred[
                    "request_id"
                ]
                .drop_duplicates()
                .tolist()
            )

            fixed_sequence = (
                replay_sequence(
                    heldout_ids,
                    repeats=(
                        DEPLOYMENT_REPEATS
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
                    "audit_strategy": (
                        "none"
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


            # =========================================================
            # AUDIT CONFIGURATIONS
            # =========================================================

            for strategy in (
                AUDIT_STRATEGIES
            ):

                for audit_rate in (
                    AUDIT_RATES
                ):

                    # -------------------------------------------------
                    # Audit-probability normalization is learned from
                    # SOURCE threshold-calibration data only.
                    # -------------------------------------------------

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
                                strategy
                            ),
                        )
                    )


                    # -------------------------------------------------
                    # Tune the stronger global baseline under exactly
                    # the same sparse-feedback regime.
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
                    # Tune online calibration LR under exactly the same
                    # sparse audit regime.
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


                    hyperparameter_rows.append(
                        {
                            "delta": delta,
                            "heldout_task": (
                                heldout_task
                            ),
                            "audit_strategy": (
                                strategy
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

                            "online_learning_rate": (
                                online_lr
                            ),
                        }
                    )


                    # -------------------------------------------------
                    # HELD-OUT DEPLOYMENT
                    # -------------------------------------------------

                    for audit_seed in range(
                        AUDIT_SEEDS
                    ):

                        sequence = (
                            replay_sequence(
                                heldout_ids,
                                repeats=(
                                    DEPLOYMENT_REPEATS
                                ),
                                seed=(
                                    10000
                                    + audit_seed
                                ),
                            )
                        )

                        for method in [
                            "global_ewma",
                            "online_calibrator",
                            "calibrated_budget",
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

                            summary, trace = (
                                simulate_stream(
                                    deployment=(
                                        heldout_pred
                                    ),
                                    sequence=(
                                        sequence
                                    ),
                                    method=method,
                                    epsilon=(
                                        RISK_TARGET
                                    ),
                                    audit_policy=(
                                        audit_policy
                                    ),
                                    base_threshold=(
                                        base_threshold
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
                                    global_params=(
                                        global_params
                                    ),
                                    seed=(
                                        20000
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
                                    "audit_strategy": (
                                        strategy
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

                            if (
                                trace
                                is not None
                            ):

                                trace[
                                    "delta"
                                ] = delta

                                trace[
                                    "heldout_task"
                                ] = heldout_task

                                trace[
                                    "audit_strategy"
                                ] = strategy

                                trace[
                                    "audit_rate_target"
                                ] = audit_rate

                                representative_traces.append(
                                    trace
                                )


    # =================================================================
    # RESULTS
    # =================================================================

    seed_results = pd.DataFrame(
        seed_rows
    )

    hyperparams = pd.DataFrame(
        hyperparameter_rows
    )

    calibration = pd.DataFrame(
        calibration_rows
    )


    # -------------------------------------------------------------
    # Per-task mean across stochastic audit seeds.
    # -------------------------------------------------------------

    per_task = (
        seed_results.groupby(
            [
                "delta",
                "heldout_task",
                "method",
                "audit_strategy",
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

            expected_risk_consumed=(
                "expected_risk_consumed",
                "mean",
            ),

            initial_risk_budget=(
                "initial_risk_budget",
                "mean",
            ),
        )
    )


    # -------------------------------------------------------------
    # Overall across held-out tasks.
    # -------------------------------------------------------------

    overall = (
        per_task.groupby(
            [
                "delta",
                "method",
                "audit_strategy",
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
        )
    )


    # -------------------------------------------------------------
    # Save.
    # -------------------------------------------------------------

    seed_results.to_csv(
        OUTDIR
        / "verified_feedback_seed_results.csv",
        index=False,
    )

    per_task.to_csv(
        OUTDIR
        / "verified_feedback_by_task.csv",
        index=False,
    )

    overall.to_csv(
        OUTDIR
        / "verified_feedback_overall.csv",
        index=False,
    )

    hyperparams.to_csv(
        OUTDIR
        / "verified_feedback_hyperparameters.csv",
        index=False,
    )

    calibration.to_csv(
        OUTDIR
        / "verified_feedback_calibration.csv",
        index=False,
    )

    if representative_traces:

        trace = pd.concat(
            representative_traces,
            ignore_index=True,
        )

        trace.to_csv(
            OUTDIR
            / "verified_feedback_representative_trace.csv",
            index=False,
        )


    # -------------------------------------------------------------
    # Console summary.
    # -------------------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        "VERIFIED QUALITY FEEDBACK - OVERALL"
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
        "\n"
        + "=" * 100
    )

    print(
        "INITIAL CALIBRATION DIAGNOSTICS"
    )

    print(
        "=" * 100
    )

    print(
        calibration.round(
            5
        ).to_string(
            index=False
        )
    )

    print(
        "\nOutputs:",
        OUTDIR
    )

    print(
        "\nEXPERIMENT 9A: PASS"
    )


if __name__ == "__main__":
    main()
