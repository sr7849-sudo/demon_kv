from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DELTAS = [0.01, 0.05, 0.10]


LENGTH_FEATURES = [
    "context_tokens",
]


UNCERTAINTY_FEATURES = [
    "last_logit_entropy",
    "last_logit_top1",
    "last_logit_margin",
    "prefill_nll",
    "prefill_perplexity_capped",
]


KV_FEATURES = [
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
]


POLICY_NUMERIC = [
    "memory_ratio",
    "compression_ratio",
]


POLICY_CATEGORICAL = [
    "policy",
    "family",
]


FEATURE_GROUPS = {
    "policy_only": [],
    "length": LENGTH_FEATURES,
    "uncertainty": UNCERTAINTY_FEATURES,
    "kv_stats": KV_FEATURES,
    "length_uncertainty": (
        LENGTH_FEATURES
        + UNCERTAINTY_FEATURES
    ),
    "all_prefill": (
        LENGTH_FEATURES
        + UNCERTAINTY_FEATURES
        + KV_FEATURES
    ),
}


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input",
        default=(
            "experiments/full_oracle_qwen3_8b/"
            "oracle_longbench_qwen3_8b.csv"
        ),
    )

    p.add_argument(
        "--outdir",
        default=(
            "experiments/full_oracle_qwen3_8b/"
            "analysis/05_risk_classification"
        ),
    )

    p.add_argument(
        "--folds",
        type=int,
        default=5,
    )

    return p.parse_args()


def make_model(
    numeric_features,
    categorical_features,
):
    transformer = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    clf = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        [
            ("features", transformer),
            ("classifier", clf),
        ]
    )


def safe_metrics(
    y_true,
    prob_safe,
):
    # y=1 means SAFE.
    result = {}

    if len(np.unique(y_true)) == 2:
        result["auroc"] = roc_auc_score(
            y_true,
            prob_safe,
        )

        result["auprc"] = average_precision_score(
            y_true,
            prob_safe,
        )
    else:
        result["auroc"] = np.nan
        result["auprc"] = np.nan

    result["brier"] = brier_score_loss(
        y_true,
        prob_safe,
    )

    return result


def choose_threshold(
    y_true,
    prob_safe,
    max_violation_rate=0.01,
):
    """
    Choose the lowest probability threshold that
    maximizes safe coverage while keeping empirical
    violation rate among accepted samples <= target.

    Accepted == model says compression policy is safe.
    Violation == accepted sample is actually unsafe.
    """

    thresholds = np.unique(
        np.concatenate(
            [
                prob_safe,
                np.array([0.0, 1.0]),
            ]
        )
    )

    thresholds = np.sort(
        thresholds
    )[::-1]

    best = None

    for threshold in thresholds:
        accepted = (
            prob_safe >= threshold
        )

        n_accept = int(
            accepted.sum()
        )

        if n_accept == 0:
            continue

        unsafe_accepts = int(
            (
                accepted
                & (y_true == 0)
            ).sum()
        )

        violation_rate = (
            unsafe_accepts
            / n_accept
        )

        coverage = (
            n_accept
            / len(y_true)
        )

        if (
            violation_rate
            <= max_violation_rate
            + 1e-12
        ):
            candidate = {
                "threshold": float(
                    threshold
                ),
                "coverage": float(
                    coverage
                ),
                "violation_rate": float(
                    violation_rate
                ),
                "n_accept": n_accept,
            }

            if (
                best is None
                or candidate["coverage"]
                > best["coverage"]
            ):
                best = candidate

    if best is None:
        return {
            "threshold": 1.000001,
            "coverage": 0.0,
            "violation_rate": 0.0,
            "n_accept": 0,
        }

    return best


def run_grouped_cv(
    df,
    delta,
    feature_group,
    folds,
):
    work = df.copy()

    work["safe"] = (
        work["degradation"]
        <= delta + 1e-12
    ).astype(int)

    request_features = [
        x
        for x in FEATURE_GROUPS[
            feature_group
        ]
        if x in work.columns
    ]

    numeric = (
        [
            x
            for x in POLICY_NUMERIC
            if x in work.columns
        ]
        + request_features
    )

    categorical = [
        x
        for x in POLICY_CATEGORICAL
        if x in work.columns
    ]

    features = (
        numeric
        + categorical
    )

    groups = work[
        "request_id"
    ].values

    X = work[features].copy()

    y = work["safe"].values

    outer = GroupKFold(
        n_splits=folds
    )

    oof_prob = np.full(
        len(work),
        np.nan,
    )

    oof_threshold_1 = np.full(
        len(work),
        np.nan,
    )

    oof_threshold_5 = np.full(
        len(work),
        np.nan,
    )

    fold_metrics = []

    for fold, (
        outer_train,
        outer_test,
    ) in enumerate(
        outer.split(
            X,
            y,
            groups,
        ),
        start=1,
    ):
        train_groups = groups[
            outer_train
        ]

        inner = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=1000 + fold,
        )

        inner_train_rel, calib_rel = next(
            inner.split(
                X.iloc[
                    outer_train
                ],
                y[
                    outer_train
                ],
                train_groups,
            )
        )

        inner_train = outer_train[
            inner_train_rel
        ]

        calib = outer_train[
            calib_rel
        ]

        model = make_model(
            numeric,
            categorical,
        )

        model.fit(
            X.iloc[
                inner_train
            ],
            y[
                inner_train
            ],
        )

        calib_prob = (
            model.predict_proba(
                X.iloc[calib]
            )[:, 1]
        )

        threshold_1 = choose_threshold(
            y[calib],
            calib_prob,
            max_violation_rate=0.01,
        )

        threshold_5 = choose_threshold(
            y[calib],
            calib_prob,
            max_violation_rate=0.05,
        )

        test_prob = (
            model.predict_proba(
                X.iloc[
                    outer_test
                ]
            )[:, 1]
        )

        oof_prob[
            outer_test
        ] = test_prob

        oof_threshold_1[
            outer_test
        ] = threshold_1[
            "threshold"
        ]

        oof_threshold_5[
            outer_test
        ] = threshold_5[
            "threshold"
        ]

        metrics = safe_metrics(
            y[
                outer_test
            ],
            test_prob,
        )

        fold_metrics.append(
            {
                "delta": delta,
                "feature_group": (
                    feature_group
                ),
                "fold": fold,
                "n_train": len(
                    inner_train
                ),
                "n_calibration": len(
                    calib
                ),
                "n_test": len(
                    outer_test
                ),
                "safe_rate_test": (
                    y[
                        outer_test
                    ].mean()
                ),
                **metrics,
                "threshold_1pct": (
                    threshold_1[
                        "threshold"
                    ]
                ),
                "calib_coverage_1pct": (
                    threshold_1[
                        "coverage"
                    ]
                ),
                "threshold_5pct": (
                    threshold_5[
                        "threshold"
                    ]
                ),
                "calib_coverage_5pct": (
                    threshold_5[
                        "coverage"
                    ]
                ),
            }
        )

    if np.isnan(
        oof_prob
    ).any():
        raise RuntimeError(
            "OOF probabilities missing"
        )

    overall = safe_metrics(
        y,
        oof_prob,
    )

    result = work[
        [
            "request_id",
            "task",
            "policy",
            "family",
            "memory_ratio",
            "compression_ratio",
            "degradation",
            "quality",
            "full_quality",
        ]
    ].copy()

    result["delta"] = delta
    result[
        "feature_group"
    ] = feature_group

    result["safe"] = y
    result["prob_safe"] = (
        oof_prob
    )

    result[
        "threshold_1pct"
    ] = oof_threshold_1

    result[
        "threshold_5pct"
    ] = oof_threshold_5

    for label, threshold_col in [
        (
            "1pct",
            "threshold_1pct",
        ),
        (
            "5pct",
            "threshold_5pct",
        ),
    ]:
        accepted = (
            result[
                "prob_safe"
            ]
            >= result[
                threshold_col
            ]
        )

        result[
            f"accepted_{label}"
        ] = accepted

        unsafe_accepted = (
            accepted
            & (result["safe"] == 0)
        )

        result[
            f"false_safe_{label}"
        ] = unsafe_accepted

    summary = {
        "delta": delta,
        "feature_group": (
            feature_group
        ),
        "n_rows": len(work),
        "n_requests": work[
            "request_id"
        ].nunique(),
        "safe_rate": y.mean(),
        **overall,
    }

    for label in [
        "1pct",
        "5pct",
    ]:
        accepted = result[
            f"accepted_{label}"
        ]

        n_accept = int(
            accepted.sum()
        )

        false_safe = int(
            result[
                f"false_safe_{label}"
            ].sum()
        )

        summary[
            f"coverage_{label}"
        ] = (
            n_accept
            / len(result)
        )

        summary[
            f"violation_rate_{label}"
        ] = (
            false_safe
            / n_accept
            if n_accept
            else 0.0
        )

        unsafe = (
            result["safe"] == 0
        )

        summary[
            f"false_safe_rate_among_unsafe_{label}"
        ] = (
            (
                result[
                    f"false_safe_{label}"
                ]
                & unsafe
            ).sum()
            / unsafe.sum()
            if unsafe.sum()
            else 0.0
        )

    return (
        pd.DataFrame(
            fold_metrics
        ),
        pd.DataFrame(
            [summary]
        ),
        result,
    )


def main():
    args = parse_args()

    out = Path(
        args.outdir
    )
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        args.input
    )

    # FullKV is trivially safe by definition.
    # Risk prediction concerns candidate
    # compressed policies.
    df = df[
        df["policy"]
        != "full"
    ].copy()

    if (
        df[
            "request_id"
        ].nunique()
        != 290
    ):
        raise RuntimeError(
            "Expected 290 requests"
        )

    all_fold = []
    all_summary = []
    all_predictions = []

    for delta in DELTAS:
        print(
            "\n"
            + "=" * 70
        )
        print(
            f"DELTA = {delta}"
        )
        print(
            "=" * 70
        )

        for feature_group in (
            FEATURE_GROUPS
        ):
            print(
                f"Running "
                f"{feature_group}..."
            )

            fold, summary, pred = (
                run_grouped_cv(
                    df,
                    delta,
                    feature_group,
                    args.folds,
                )
            )

            all_fold.append(
                fold
            )
            all_summary.append(
                summary
            )
            all_predictions.append(
                pred
            )

            row = (
                summary.iloc[0]
            )

            print(
                f"  AUROC="
                f"{row.auroc:.4f} "
                f"AUPRC="
                f"{row.auprc:.4f} "
                f"Brier="
                f"{row.brier:.4f}"
            )

            print(
                f"  coverage@1%="
                f"{row.coverage_1pct:.3f} "
                f"viol="
                f"{row.violation_rate_1pct:.3f}"
            )

    fold_df = pd.concat(
        all_fold,
        ignore_index=True,
    )

    summary_df = pd.concat(
        all_summary,
        ignore_index=True,
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    fold_df.to_csv(
        out
        / "risk_classification_folds.csv",
        index=False,
    )

    summary_df.to_csv(
        out
        / "risk_classification_summary.csv",
        index=False,
    )

    predictions_df.to_csv(
        out
        / "risk_classification_oof_predictions.csv",
        index=False,
    )

    print(
        "\n"
        + "=" * 70
    )
    print(
        "RISK CLASSIFICATION SUMMARY"
    )
    print(
        "=" * 70
    )

    display_cols = [
        "delta",
        "feature_group",
        "safe_rate",
        "auroc",
        "auprc",
        "brier",
        "coverage_1pct",
        "violation_rate_1pct",
        "coverage_5pct",
        "violation_rate_5pct",
    ]

    print(
        summary_df[
            display_cols
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        "\nRISK CLASSIFICATION: PASS"
    )


if __name__ == "__main__":
    main()
