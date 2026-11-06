"""
EXPERIMENT 8F
=============

Context-conditioned quality feedback under cross-task distribution shift.

======================================================================
CORE RESEARCH QUESTION
======================================================================

The previous global quality-feedback experiment showed that violation
drift is structured:

    some held-out workload regions are already safe,
    while others remain highly miscalibrated.

A single global threshold correction therefore has a problem:

        bad region becomes risky
               |
               v
        raise threshold for EVERY request

This can unnecessarily reduce compression for already-safe requests.

This experiment tests a more localized feedback mechanism:

        request
           |
           v
      context vector
           |
           v
       context cluster k
           |
           v
    cluster-specific correction b_k
           |
           v
     adjusted safety threshold


The method is intentionally NOT policy-conditioned yet.

We maintain:

        one correction per REQUEST CONTEXT CLUSTER,

rather than:

        one correction per (cluster, policy).

That avoids severe data sparsity and avoids introducing a second
research question about the internal mechanisms of every compression
family.

======================================================================
THREE STAGES
======================================================================

STAGE 1 -- REQUEST CONTEXT VECTOR

For each request:

    x_r = [
        log(context length),
        prefill NLL,
        last-logit entropy,
        last-logit margin
    ]

These signals are observable before compressed decoding.

We deliberately begin with a small feature set because the previous
leave-one-task-out experiment showed that larger engineered prefill
feature sets did not consistently improve transfer.


STAGE 2 -- CONTEXT CLUSTERING

Fit StandardScaler + KMeans on SOURCE TRAINING REQUESTS ONLY.

Candidate cluster counts:

    K in {2, 4, 6}

K is chosen by maximum silhouette score on the source training
requests.

NO held-out-task labels or outcomes are used to choose K.


STAGE 3 -- LOCAL QUALITY FEEDBACK

Each cluster k maintains an independent PI correction:

    b_k,t

For request r_t assigned to cluster k:

    tau_t(r_t)
        =
    clip(
        tau_0 + b_k,t,
        0,
        1
    )

where:

    tau_0 = offline calibrated base threshold.

When an audited request reveals whether:

    D(r_t, p_t) > delta

we update ONLY the controller belonging to that request's cluster.

If cluster k is violating too frequently:

    v_hat_k > epsilon

then:

    b_k increases

and future requests from that region require a higher predicted
safety probability before compression is accepted.

Other clusters are unchanged.


======================================================================
COMPARISON
======================================================================

1. fixed_threshold
       no runtime quality feedback

2. global_pi
       one PI correction shared by every request

3. context_pi
       one PI correction for each learned context cluster


======================================================================
IMPORTANT EVALUATION RULE
======================================================================

The held-out LongBench task is NEVER used to:

    train the predictor,
    fit the scaler,
    fit KMeans,
    choose K,
    calibrate the initial threshold,
    tune PI gains.

Ground-truth degradation on the held-out task becomes visible only
AFTER a decision and only when that request is selected for a simulated
runtime audit.

Therefore the feedback label behaves as delayed runtime information,
not privileged selection-time information.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from riskkv.feedback_control import PIDController
from riskkv.synthesis_eval import calibrate_threshold


# ======================================================================
# PATHS
# ======================================================================

ORACLE = (
    "experiments/full_oracle_qwen3_8b/"
    "oracle_longbench_qwen3_8b.csv"
)

OUTDIR = Path(
    "experiments/full_oracle_qwen3_8b/"
    "analysis/27_context_quality_feedback"
)


# ======================================================================
# EXPERIMENT SETTINGS
# ======================================================================

DELTAS = [
    0.01,
    0.05,
    0.10,
]

RISK_TARGET = 0.05


# Include a somewhat larger audit rate here because cluster-specific
# feedback naturally receives fewer labels than a global controller.
#
# The goal is initially to answer:
#
#     "Does localization help at all?"
#
# rather than pretending 1% feedback must already be sufficient.
AUDIT_RATES = [
    0.05,
    0.10,
    0.20,
]


AUDIT_SEEDS = 30

ENSEMBLE_SIZE = 5

K_CANDIDATES = [
    2,
    4,
    6,
]


# PI gains are tuned ONLY using source-task calibration requests.
GAIN_GRID = [
    (kp, ki)
    for kp in [
        0.05,
        0.10,
        0.20,
        0.30,
        0.50,
    ]
    for ki in [
        0.0,
        0.005,
        0.01,
        0.02,
        0.05,
    ]
]


# ======================================================================
# SMALL ROBUST CONTEXT FEATURE SET
# ======================================================================

CONTEXT_FEATURES = [
    "context_tokens",
    "prefill_nll",
    "last_logit_entropy",
    "last_logit_margin",
]


# Predictor can use the same wider prefill information as the previous
# RiskKV cross-task experiment.
PREDICTOR_PREFILL_FEATURES = [
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
# REQUEST-LEVEL TRAIN / CALIBRATION SPLIT
# ======================================================================

def split_source_requests(
    source: pd.DataFrame,
    seed: int = 42,
    calibration_fraction: float = 0.20,
):
    """
    Split SOURCE tasks at REQUEST level.

    All seven compressed-policy rows belonging to one request stay
    together.

    This prevents the same request from appearing simultaneously in
    predictor training and threshold/controller calibration.
    """

    ids = (
        source["request_id"]
        .drop_duplicates()
        .to_numpy()
    )

    rng = np.random.default_rng(seed)
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
        ~source["request_id"].isin(
            cal_ids
        )
    ].copy()

    calibration = source[
        source["request_id"].isin(
            cal_ids
        )
    ].copy()

    return train, calibration


# ======================================================================
# RISK-PREDICTOR FEATURES
# ======================================================================

def build_predictor_features(
    df: pd.DataFrame,
    columns=None,
):
    """
    Build policy-conditioned RiskKV predictor features.

    Policy identity is intentionally included because RiskKV predicts:

        P(
            D(request, policy) <= delta
        )

    rather than merely classifying the request itself.
    """

    x = pd.get_dummies(
        df[["policy"]],
        columns=["policy"],
        dtype=float,
    )

    if "memory_ratio" in df.columns:
        x["memory_ratio"] = (
            pd.to_numeric(
                df["memory_ratio"],
                errors="coerce",
            )
        )

    for col in PREDICTOR_PREFILL_FEATURES:

        if col in df.columns:

            x[col] = pd.to_numeric(
                df[col],
                errors="coerce",
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


# ======================================================================
# BOOTSTRAP RISK ENSEMBLE
# ======================================================================

def train_risk_ensemble(
    train: pd.DataFrame,
    evaluation_sets: dict[str, pd.DataFrame],
):
    """
    Train a request-level bootstrap ensemble.

    WHY BOOTSTRAP REQUESTS?
    -----------------------
    All policy rows for one request must remain together.

    Each ensemble member therefore bootstraps REQUEST IDs rather than
    independently bootstrapping request-policy rows.

    The output used in this experiment is:

        prob_safe_mean

    Ensemble uncertainty is retained in the output because it may be
    useful in subsequent auditing experiments.
    """

    request_ids = (
        train["request_id"]
        .drop_duplicates()
        .to_numpy()
    )

    prediction_stacks = {
        name: []
        for name in evaluation_sets
    }

    for member in range(
        ENSEMBLE_SIZE
    ):

        rng = np.random.default_rng(
            1000 + member
        )

        bootstrap_ids = (
            rng.choice(
                request_ids,
                size=len(
                    request_ids
                ),
                replace=True,
            )
        )

        boot = pd.concat(
            [
                train[
                    train["request_id"]
                    == rid
                ]
                for rid in bootstrap_ids
            ],
            ignore_index=True,
        )

        X_train = (
            build_predictor_features(
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
                    1000 + member
                ),
            )
        )

        model.fit(
            X_train,
            boot["safe"],
        )

        for name, df in (
            evaluation_sets.items()
        ):

            X = (
                build_predictor_features(
                    df,
                    columns=columns,
                )
            )

            p = (
                model.predict_proba(
                    X
                )[:, 1]
            )

            prediction_stacks[
                name
            ].append(p)

    outputs = {}

    for name, df in (
        evaluation_sets.items()
    ):

        stack = np.vstack(
            prediction_stacks[
                name
            ]
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


# ======================================================================
# STAGE 1: REQUEST CONTEXT VECTOR
# ======================================================================

def request_context_table(
    df: pd.DataFrame,
):
    """
    Convert policy-level rows into ONE context vector per request.

    Context vector:

        x_r = [
            log(1 + context_tokens),
            prefill_nll,
            last_logit_entropy,
            last_logit_margin
        ]

    We intentionally avoid policy-specific quantities here.

    Therefore the clustering describes the REQUEST, not the compression
    mechanism.
    """

    missing = [
        col
        for col in CONTEXT_FEATURES
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing context features: "
            + str(missing)
        )

    cols = [
        "request_id",
        "task",
    ] + CONTEXT_FEATURES

    x = (
        df[cols]
        .drop_duplicates(
            "request_id"
        )
        .copy()
    )

    x[
        "log_context_tokens"
    ] = np.log1p(
        pd.to_numeric(
            x[
                "context_tokens"
            ],
            errors="coerce",
        )
    )

    for col in [
        "prefill_nll",
        "last_logit_entropy",
        "last_logit_margin",
    ]:

        x[col] = pd.to_numeric(
            x[col],
            errors="coerce",
        )

    return x[
        [
            "request_id",
            "task",
            "log_context_tokens",
            "prefill_nll",
            "last_logit_entropy",
            "last_logit_margin",
        ]
    ]


CLUSTER_FEATURE_COLUMNS = [
    "log_context_tokens",
    "prefill_nll",
    "last_logit_entropy",
    "last_logit_margin",
]


# ======================================================================
# STAGE 2: FIT CONTEXT CLUSTERS
# ======================================================================

def fit_context_clusterer(
    train: pd.DataFrame,
):
    """
    Fit the context representation on SOURCE TRAINING REQUESTS ONLY.

    ------------------------------------------------------------
    Why StandardScaler?
    ------------------------------------------------------------

    Context length, NLL, entropy and margin have different numerical
    ranges.

    Without scaling, KMeans could be dominated merely by the largest
    numerical feature.

    ------------------------------------------------------------
    Why silhouette score?
    ------------------------------------------------------------

    Instead of arbitrarily deciding K=4 or K=6, evaluate:

        K in {2,4,6}

    using ONLY the geometry of source training contexts.

    Higher silhouette score means:
        tighter within-cluster grouping
        + stronger separation between clusters.

    No quality label and no held-out-task outcome is used to choose K.
    """

    request_context = (
        request_context_table(
            train
        )
    )

    feature_df = (
        request_context[
            CLUSTER_FEATURE_COLUMNS
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .copy()
    )

    medians = (
        feature_df.median()
    )

    feature_df = (
        feature_df.fillna(
            medians
        )
    )

    scaler = StandardScaler()

    X = scaler.fit_transform(
        feature_df
    )

    candidates = []

    for k in K_CANDIDATES:

        if len(X) <= k:
            continue

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20,
        )

        labels = (
            model.fit_predict(
                X
            )
        )

        # Silhouette is undefined when all points receive one label.
        if len(
            np.unique(
                labels
            )
        ) < 2:
            continue

        score = silhouette_score(
            X,
            labels,
        )

        candidates.append(
            {
                "k": k,
                "silhouette": (
                    float(score)
                ),
                "model": model,
            }
        )

    if not candidates:
        raise RuntimeError(
            "No valid KMeans configuration."
        )

    best = max(
        candidates,
        key=lambda x:
            x["silhouette"],
    )

    diagnostics = pd.DataFrame(
        [
            {
                "k": item["k"],
                "silhouette": (
                    item["silhouette"]
                ),
            }
            for item in candidates
        ]
    )

    return {
        "scaler": scaler,
        "model": best["model"],
        "k": int(best["k"]),
        "silhouette": float(
            best["silhouette"]
        ),
        "medians": medians,
        "diagnostics": diagnostics,
    }


def assign_context_clusters(
    rows: pd.DataFrame,
    clusterer,
):
    """
    Assign each request to the nearest SOURCE-TRAINED context cluster.

    The resulting cluster ID is copied back to every policy row
    belonging to that request.
    """

    request_context = (
        request_context_table(
            rows
        )
    )

    X = (
        request_context[
            CLUSTER_FEATURE_COLUMNS
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(
            clusterer[
                "medians"
            ]
        )
    )

    X_scaled = (
        clusterer[
            "scaler"
        ].transform(
            X
        )
    )

    request_context[
        "context_cluster"
    ] = (
        clusterer[
            "model"
        ].predict(
            X_scaled
        )
    )

    cluster_map = (
        request_context[
            [
                "request_id",
                "context_cluster",
            ]
        ]
    )

    result = rows.merge(
        cluster_map,
        on="request_id",
        how="left",
        validate="many_to_one",
    )

    return (
        result,
        request_context,
    )


# ======================================================================
# FULLKV FALLBACK
# ======================================================================

def make_fullkv_row(
    g: pd.DataFrame,
):
    """
    FullKV is the reference configuration.

    By construction:

        degradation = 0
        safe = 1
        memory ratio = 1

    It does not need a learned compression-risk prediction.
    """

    r = g.iloc[0].copy()

    r["policy"] = "full"
    r["memory_ratio"] = 1.0
    r["degradation"] = 0.0
    r["safe"] = 1
    r["prob_safe_mean"] = 1.0
    r["risk_uncertainty"] = 0.0

    return r


# ======================================================================
# STAGE 3: FEEDBACK STREAM
# ======================================================================

def simulate_feedback_stream(
    deployment: pd.DataFrame,
    base_threshold: float,
    audit_rate: float,
    method: str,
    kp: float,
    ki: float,
    risk_target: float,
    seed: int,
):
    """
    Simulate one deployment stream.

    ------------------------------------------------------------
    METHODS
    ------------------------------------------------------------

    fixed_threshold:
        tau(r) = tau_0

    global_pi:
        tau_t(r)
            =
        tau_0 + b_global,t

    context_pi:
        tau_t(r)
            =
        tau_0 + b_cluster(r),t


    ------------------------------------------------------------
    AUDIT MECHANISM
    ------------------------------------------------------------

    After a compressed request is selected:

        with probability audit_rate

    reveal its true quality-budget violation label:

        z_t = 1[D(r,p) > delta]

    This information becomes available only AFTER selection.

    ------------------------------------------------------------
    GLOBAL UPDATE
    ------------------------------------------------------------

    Maintain:

        v_hat_global

    and update one global PI controller.


    ------------------------------------------------------------
    CONTEXT UPDATE
    ------------------------------------------------------------

    Maintain independent:

        v_hat_0
        v_hat_1
        ...
        v_hat_K-1

    An audit from cluster 2 updates ONLY cluster 2.

    This is the central experimental intervention.
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

    cluster_ids = sorted(
        deployment[
            "context_cluster"
        ]
        .drop_duplicates()
        .astype(int)
        .tolist()
    )

    global_controller = (
        PIDController(
            kp=kp,
            ki=ki,
            kd=0.0,
            output_min=-0.40,
            output_max=0.40,
            integral_limit=5.0,
        )
    )

    cluster_controllers = {
        k: PIDController(
            kp=kp,
            ki=ki,
            kd=0.0,
            output_min=-0.40,
            output_max=0.40,
            integral_limit=5.0,
        )
        for k in cluster_ids
    }

    global_audits = []

    cluster_audits = {
        k: []
        for k in cluster_ids
    }

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

        cluster = int(
            g[
                "context_cluster"
            ].iloc[0]
        )

        # --------------------------------------------------------
        # CURRENT THRESHOLD
        # --------------------------------------------------------

        if method == "fixed_threshold":

            correction = 0.0

        elif method == "global_pi":

            correction = (
                global_controller.value
            )

        elif method == "context_pi":

            correction = (
                cluster_controllers[
                    cluster
                ].value
            )

        else:

            raise ValueError(
                "Unknown method: "
                + method
            )

        threshold = float(
            np.clip(
                base_threshold
                + correction,
                0.0,
                1.000001,
            )
        )

        # --------------------------------------------------------
        # CONFIGURATION SELECTION
        #
        # Among predicted-safe compressed policies, choose the most
        # memory-aggressive configuration.
        #
        # Ties prefer higher predicted safety.
        # --------------------------------------------------------

        accepted = g[
            g[
                "prob_safe_mean"
            ]
            >= threshold
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
                make_fullkv_row(
                    g
                )
            )

        violation = bool(
            chosen[
                "degradation"
            ]
            > chosen[
                "delta"
            ]
            + 1e-12
        )

        # --------------------------------------------------------
        # SELECTIVE RUNTIME AUDIT
        #
        # FullKV has degradation zero by definition, so auditing it
        # provides no useful compression-risk information.
        # --------------------------------------------------------

        audited = False
        observed_violation_rate = np.nan

        if (
            chosen["policy"]
            != "full"
            and audit_rate > 0
            and rng.random()
            < audit_rate
        ):

            audited = True

            z = float(
                violation
            )

            if method == "global_pi":

                global_audits.append(
                    z
                )

                observed_violation_rate = float(
                    np.mean(
                        global_audits
                    )
                )

                error = (
                    observed_violation_rate
                    - risk_target
                )

                global_controller.update(
                    error
                )

            elif method == "context_pi":

                cluster_audits[
                    cluster
                ].append(
                    z
                )

                observed_violation_rate = float(
                    np.mean(
                        cluster_audits[
                            cluster
                        ]
                    )
                )

                error = (
                    observed_violation_rate
                    - risk_target
                )

                # ------------------------------------------------
                # THE KEY CONTEXT-CONDITIONED UPDATE
                #
                # ONLY the current request's cluster controller is
                # modified.
                #
                # Every other context region preserves its previous
                # calibration state.
                # ------------------------------------------------

                cluster_controllers[
                    cluster
                ].update(
                    error
                )

        rows.append(
            {
                "step": step,
                "request_id": rid,
                "task": (
                    chosen[
                        "task"
                    ]
                ),
                "context_cluster": (
                    cluster
                ),
                "method": method,
                "audit_rate_target": (
                    audit_rate
                ),
                "policy": (
                    chosen[
                        "policy"
                    ]
                ),
                "threshold": (
                    threshold
                ),
                "correction": (
                    correction
                ),
                "audited": (
                    audited
                ),
                "observed_violation_rate": (
                    observed_violation_rate
                ),
                "actual_violation": (
                    violation
                ),
                "degradation": (
                    chosen[
                        "degradation"
                    ]
                ),
                "memory_saved": (
                    1.0
                    - float(
                        chosen[
                            "memory_ratio"
                        ]
                    )
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


# ======================================================================
# STREAM SUMMARY
# ======================================================================

def summarize_stream(
    x: pd.DataFrame,
):
    """
    Summarize both GLOBAL and WORST-CONTEXT behavior.

    The worst-context metric is important because the previous global
    feedback experiment showed that a reasonable average could hide
    severe workload-specific failures.
    """

    cluster_summary = (
        x.groupby(
            "context_cluster"
        )
        .agg(
            violation_rate=(
                "actual_violation",
                "mean",
            ),
            requests=(
                "request_id",
                "nunique",
            ),
        )
        .reset_index()
    )

    worst_cluster_violation = float(
        cluster_summary[
            "violation_rate"
        ].max()
    )

    return {
        "requests": len(x),

        "violation_rate": (
            x[
                "actual_violation"
            ].mean()
        ),

        "worst_cluster_violation_rate": (
            worst_cluster_violation
        ),

        "mean_memory_saved": (
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


# ======================================================================
# PI GAIN TUNING
# ======================================================================

def tune_pi(
    calibration: pd.DataFrame,
    base_threshold: float,
    method: str,
    risk_target: float,
):
    """
    Tune PI gains on SOURCE CALIBRATION REQUESTS ONLY.

    We use full feedback during gain tuning because this is an offline
    controller-design step.

    IMPORTANT:
    Held-out-task data is not used here.

    ------------------------------------------------------------
    GLOBAL PI OBJECTIVE
    ------------------------------------------------------------

    1. minimize overall violation excess above epsilon
    2. maximize memory saving

    ------------------------------------------------------------
    CONTEXT PI OBJECTIVE
    ------------------------------------------------------------

    1. minimize WORST-CLUSTER violation excess
    2. minimize overall violation excess
    3. maximize memory saving

    This reflects the actual motivation for context-conditioned
    feedback: preventing one request region from remaining severely
    miscalibrated while the global average looks acceptable.
    """

    rows = []

    for kp, ki in GAIN_GRID:

        seed_metrics = []

        # Multiple orderings reduce dependence on one calibration
        # sequence.
        for seed in [
            101,
            202,
            303,
        ]:

            trace = (
                simulate_feedback_stream(
                    deployment=(
                        calibration
                    ),
                    base_threshold=(
                        base_threshold
                    ),
                    audit_rate=1.0,
                    method=method,
                    kp=kp,
                    ki=ki,
                    risk_target=(
                        risk_target
                    ),
                    seed=seed,
                )
            )

            m = summarize_stream(
                trace
            )

            seed_metrics.append(m)

        m = pd.DataFrame(
            seed_metrics
        ).mean(
            numeric_only=True
        )

        rows.append(
            {
                "kp": kp,
                "ki": ki,

                "violation_rate": (
                    m[
                        "violation_rate"
                    ]
                ),

                "worst_cluster_violation_rate": (
                    m[
                        "worst_cluster_violation_rate"
                    ]
                ),

                "overall_excess": max(
                    0.0,
                    m[
                        "violation_rate"
                    ]
                    - risk_target,
                ),

                "worst_cluster_excess": max(
                    0.0,
                    m[
                        "worst_cluster_violation_rate"
                    ]
                    - risk_target,
                ),

                "mean_memory_saved": (
                    m[
                        "mean_memory_saved"
                    ]
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    if method == "context_pi":

        result = result.sort_values(
            [
                "worst_cluster_excess",
                "overall_excess",
                "mean_memory_saved",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )

    else:

        result = result.sort_values(
            [
                "overall_excess",
                "mean_memory_saved",
                "worst_cluster_excess",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )

    return result.iloc[0]


# ======================================================================
# CLUSTER CHARACTERIZATION
# ======================================================================

def cluster_profile_table(
    request_context: pd.DataFrame,
    split_name: str,
    delta: float,
    heldout_task: str,
):
    """
    Produce interpretable cluster summaries.

    We do NOT manually assign names such as "retrieval" or "hard QA".

    Instead the table reports what each learned cluster actually looks
    like in terms of the observable request signals.
    """

    rows = []

    for cluster, g in (
        request_context.groupby(
            "context_cluster"
        )
    ):

        rows.append(
            {
                "delta": delta,
                "heldout_task": (
                    heldout_task
                ),
                "split": split_name,
                "context_cluster": (
                    int(cluster)
                ),
                "requests": len(g),

                "mean_context_tokens": (
                    np.expm1(
                        g[
                            "log_context_tokens"
                        ]
                    ).mean()
                ),

                "mean_prefill_nll": (
                    g[
                        "prefill_nll"
                    ].mean()
                ),

                "mean_entropy": (
                    g[
                        "last_logit_entropy"
                    ].mean()
                ),

                "mean_margin": (
                    g[
                        "last_logit_margin"
                    ].mean()
                ),
            }
        )

    return rows


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

    # Risk model operates on compressed policies only.
    df = df[
        df["policy"]
        != "full"
    ].copy()

    tasks = sorted(
        df[
            "task"
        ].unique()
    )

    all_summaries = []
    all_traces = []
    all_cluster_profiles = []
    all_cluster_diagnostics = []
    all_gain_rows = []


    # ================================================================
    # LEAVE-ONE-TASK-OUT EVALUATION
    # ================================================================

    for heldout_task in tasks:

        source_all = df[
            df["task"]
            != heldout_task
        ].copy()

        heldout_all = df[
            df["task"]
            == heldout_task
        ].copy()


        # ------------------------------------------------------------
        # Cluster fitting does not depend on delta.
        #
        # Fit it once using only SOURCE TRAINING requests.
        # ------------------------------------------------------------

        cluster_train, _ = (
            split_source_requests(
                source_all,
                seed=42,
            )
        )

        clusterer = (
            fit_context_clusterer(
                cluster_train
            )
        )

        print(
            "\nHELD-OUT TASK:",
            heldout_task
        )

        print(
            "Selected K:",
            clusterer["k"],
            "silhouette:",
            round(
                clusterer[
                    "silhouette"
                ],
                4,
            )
        )

        diagnostics = (
            clusterer[
                "diagnostics"
            ].copy()
        )

        diagnostics[
            "heldout_task"
        ] = heldout_task

        all_cluster_diagnostics.append(
            diagnostics
        )


        # ============================================================
        # QUALITY BUDGETS
        # ============================================================

        for delta in DELTAS:

            source = source_all.copy()
            heldout = heldout_all.copy()

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
                <= delta + 1e-12
            ).astype(int)

            heldout[
                "safe"
            ] = (
                heldout[
                    "degradation"
                ]
                <= delta + 1e-12
            ).astype(int)


            # --------------------------------------------------------
            # Predictor train/calibration split.
            # --------------------------------------------------------

            train, calibration = (
                split_source_requests(
                    source,
                    seed=42,
                )
            )


            # --------------------------------------------------------
            # Train leave-one-task-out risk predictor.
            # --------------------------------------------------------

            predicted = (
                train_risk_ensemble(
                    train=train,
                    evaluation_sets={
                        "calibration": (
                            calibration
                        ),
                        "heldout": (
                            heldout
                        ),
                    },
                )
            )

            calibration_pred = (
                predicted[
                    "calibration"
                ]
            )

            heldout_pred = (
                predicted[
                    "heldout"
                ]
            )


            # --------------------------------------------------------
            # Assign SOURCE-TRAINED context clusters.
            # --------------------------------------------------------

            (
                calibration_pred,
                calibration_context,
            ) = (
                assign_context_clusters(
                    calibration_pred,
                    clusterer,
                )
            )

            (
                heldout_pred,
                heldout_context,
            ) = (
                assign_context_clusters(
                    heldout_pred,
                    clusterer,
                )
            )


            # --------------------------------------------------------
            # Initial global threshold.
            #
            # Same starting point for all three methods.
            # --------------------------------------------------------

            base_threshold = (
                calibrate_threshold(
                    calibration_pred[
                        "safe"
                    ],
                    calibration_pred[
                        "prob_safe_mean"
                    ],
                    RISK_TARGET,
                )
            )


            # --------------------------------------------------------
            # Offline PI gain tuning.
            # --------------------------------------------------------

            global_gain = tune_pi(
                calibration=(
                    calibration_pred
                ),
                base_threshold=(
                    base_threshold
                ),
                method="global_pi",
                risk_target=(
                    RISK_TARGET
                ),
            )

            context_gain = tune_pi(
                calibration=(
                    calibration_pred
                ),
                base_threshold=(
                    base_threshold
                ),
                method="context_pi",
                risk_target=(
                    RISK_TARGET
                ),
            )

            all_gain_rows.extend(
                [
                    {
                        "delta": delta,
                        "heldout_task": (
                            heldout_task
                        ),
                        "method": (
                            "global_pi"
                        ),
                        "base_threshold": (
                            base_threshold
                        ),
                        "kp": float(
                            global_gain[
                                "kp"
                            ]
                        ),
                        "ki": float(
                            global_gain[
                                "ki"
                            ]
                        ),
                    },
                    {
                        "delta": delta,
                        "heldout_task": (
                            heldout_task
                        ),
                        "method": (
                            "context_pi"
                        ),
                        "base_threshold": (
                            base_threshold
                        ),
                        "kp": float(
                            context_gain[
                                "kp"
                            ]
                        ),
                        "ki": float(
                            context_gain[
                                "ki"
                            ]
                        ),
                    },
                ]
            )


            # --------------------------------------------------------
            # Cluster characterization.
            # --------------------------------------------------------

            all_cluster_profiles.extend(
                cluster_profile_table(
                    calibration_context,
                    split_name=(
                        "source_calibration"
                    ),
                    delta=delta,
                    heldout_task=(
                        heldout_task
                    ),
                )
            )

            all_cluster_profiles.extend(
                cluster_profile_table(
                    heldout_context,
                    split_name=(
                        "heldout_deployment"
                    ),
                    delta=delta,
                    heldout_task=(
                        heldout_task
                    ),
                )
            )


            # ========================================================
            # BASELINE: FIXED THRESHOLD
            # ========================================================

            fixed = (
                simulate_feedback_stream(
                    deployment=(
                        heldout_pred
                    ),
                    base_threshold=(
                        base_threshold
                    ),
                    audit_rate=0.0,
                    method=(
                        "fixed_threshold"
                    ),
                    kp=0.0,
                    ki=0.0,
                    risk_target=(
                        RISK_TARGET
                    ),
                    seed=10000,
                )
            )

            fixed[
                "delta"
            ] = delta

            fixed[
                "heldout_task"
            ] = heldout_task

            fixed[
                "audit_seed"
            ] = 0

            all_traces.append(
                fixed
            )

            m = summarize_stream(
                fixed
            )

            m.update(
                {
                    "delta": delta,
                    "heldout_task": (
                        heldout_task
                    ),
                    "method": (
                        "fixed_threshold"
                    ),
                    "audit_rate_target": (
                        0.0
                    ),
                    "std_violation_rate": (
                        0.0
                    ),
                    "selected_k": (
                        clusterer["k"]
                    ),
                }
            )

            all_summaries.append(m)


            # ========================================================
            # FEEDBACK METHODS
            # ========================================================

            for method, gain in [
                (
                    "global_pi",
                    global_gain,
                ),
                (
                    "context_pi",
                    context_gain,
                ),
            ]:

                for audit_rate in (
                    AUDIT_RATES
                ):

                    seed_metrics = []

                    for audit_seed in range(
                        AUDIT_SEEDS
                    ):

                        trace = (
                            simulate_feedback_stream(
                                deployment=(
                                    heldout_pred
                                ),
                                base_threshold=(
                                    base_threshold
                                ),
                                audit_rate=(
                                    audit_rate
                                ),
                                method=method,
                                kp=float(
                                    gain[
                                        "kp"
                                    ]
                                ),
                                ki=float(
                                    gain[
                                        "ki"
                                    ]
                                ),
                                risk_target=(
                                    RISK_TARGET
                                ),
                                seed=(
                                    10000
                                    + audit_seed
                                ),
                            )
                        )

                        trace[
                            "delta"
                        ] = delta

                        trace[
                            "heldout_task"
                        ] = heldout_task

                        trace[
                            "audit_seed"
                        ] = audit_seed

                        all_traces.append(
                            trace
                        )

                        seed_metrics.append(
                            summarize_stream(
                                trace
                            )
                        )

                    seed_df = (
                        pd.DataFrame(
                            seed_metrics
                        )
                    )

                    all_summaries.append(
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

                            "requests": (
                                seed_df[
                                    "requests"
                                ].mean()
                            ),

                            "violation_rate": (
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
                            ),

                            "worst_cluster_violation_rate": (
                                seed_df[
                                    "worst_cluster_violation_rate"
                                ].mean()
                            ),

                            "mean_memory_saved": (
                                seed_df[
                                    "mean_memory_saved"
                                ].mean()
                            ),

                            "fullkv_rate": (
                                seed_df[
                                    "fullkv_rate"
                                ].mean()
                            ),

                            "actual_audit_rate": (
                                seed_df[
                                    "actual_audit_rate"
                                ].mean()
                            ),

                            "selected_k": (
                                clusterer["k"]
                            ),
                        }
                    )


    # =================================================================
    # SAVE RESULTS
    # =================================================================

    summary = pd.DataFrame(
        all_summaries
    )

    trace = pd.concat(
        all_traces,
        ignore_index=True,
    )

    cluster_profiles = (
        pd.DataFrame(
            all_cluster_profiles
        )
    )

    cluster_diagnostics = (
        pd.concat(
            all_cluster_diagnostics,
            ignore_index=True,
        )
    )

    gains = pd.DataFrame(
        all_gain_rows
    )


    # -----------------------------------------------------------------
    # Overall average across held-out tasks.
    # -----------------------------------------------------------------

    overall = (
        summary.groupby(
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

            mean_worst_cluster_violation_rate=(
                "worst_cluster_violation_rate",
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

            mean_actual_audit_rate=(
                "actual_audit_rate",
                "mean",
            ),
        )
    )


    # -----------------------------------------------------------------
    # Cluster-level held-out violation analysis.
    #
    # This lets us inspect whether context feedback specifically repairs
    # the problematic regions instead of only improving the global mean.
    # -----------------------------------------------------------------

    cluster_results = (
        trace.groupby(
            [
                "delta",
                "heldout_task",
                "method",
                "audit_rate_target",
                "audit_seed",
                "context_cluster",
            ],
            as_index=False,
        )
        .agg(
            violation_rate=(
                "actual_violation",
                "mean",
            ),

            memory_saved=(
                "memory_saved",
                "mean",
            ),

            fullkv_rate=(
                "fullkv",
                "mean",
            ),

            requests=(
                "request_id",
                "nunique",
            ),
        )
    )


    # -----------------------------------------------------------------
    # Write all diagnostic artifacts.
    # -----------------------------------------------------------------

    summary.to_csv(
        OUTDIR
        / "context_feedback_by_task.csv",
        index=False,
    )

    overall.to_csv(
        OUTDIR
        / "context_feedback_overall.csv",
        index=False,
    )

    cluster_results.to_csv(
        OUTDIR
        / "context_feedback_by_cluster.csv",
        index=False,
    )

    cluster_profiles.to_csv(
        OUTDIR
        / "context_cluster_profiles.csv",
        index=False,
    )

    cluster_diagnostics.to_csv(
        OUTDIR
        / "context_cluster_k_selection.csv",
        index=False,
    )

    gains.to_csv(
        OUTDIR
        / "context_feedback_pi_gains.csv",
        index=False,
    )

    trace.to_csv(
        OUTDIR
        / "context_feedback_trace.csv",
        index=False,
    )


    # =================================================================
    # CONSOLE REPORT
    # =================================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "CONTEXT-CONDITIONED QUALITY FEEDBACK"
    )

    print(
        "=" * 90
    )

    print(
        overall.round(5)
        .to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "PER HELD-OUT TASK"
    )

    print(
        "=" * 90
    )

    print(
        summary[
            [
                "delta",
                "heldout_task",
                "method",
                "audit_rate_target",
                "violation_rate",
                "worst_cluster_violation_rate",
                "mean_memory_saved",
                "fullkv_rate",
                "actual_audit_rate",
                "selected_k",
            ]
        ]
        .round(5)
        .to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "CLUSTER K SELECTION"
    )

    print(
        "=" * 90
    )

    print(
        cluster_diagnostics
        .round(5)
        .to_string(
            index=False
        )
    )

    print(
        "\nOutputs:",
        OUTDIR
    )

    print(
        "\nEXPERIMENT 8F: PASS"
    )


if __name__ == "__main__":
    main()
