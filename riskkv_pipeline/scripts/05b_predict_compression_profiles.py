from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DELTAS = [0.01, 0.05, 0.10]

ENSEMBLE_SEEDS = [
    11,
    23,
    37,
    53,
    71,
    89,
    107,
]


REQUEST_FEATURES = [
    "context_tokens",

    "last_logit_entropy",
    "last_logit_top1",
    "last_logit_margin",

    "prefill_nll",
    "prefill_perplexity_capped",

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


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--oracle",
        default=(
            "experiments/full_oracle_qwen3_8b/"
            "oracle_longbench_qwen3_8b.csv"
        ),
    )

    p.add_argument(
        "--profiles",
        default=(
            "experiments/full_oracle_qwen3_8b/"
            "analysis/08_compression_profiles/"
            "oracle_request_profiles.csv"
        ),
    )

    p.add_argument(
        "--outdir",
        default=(
            "experiments/full_oracle_qwen3_8b/"
            "analysis/09_profile_prediction"
        ),
    )

    p.add_argument(
        "--folds",
        type=int,
        default=5,
    )

    p.add_argument(
        "--convenience-lcb-lambda",
        type=float,
        default=1.0,
        help=(
            "Lambda used only to emit convenience LCB columns. "
            "Raw ensemble mean/std are always saved and should "
            "be used for downstream lambda sweeps."
        ),
    )

    return p.parse_args()


def make_preprocessor(
    numeric,
    categorical,
):
    return ColumnTransformer(
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
                numeric,
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
                categorical,
            ),
        ]
    )


def classifier_pipeline(
    numeric,
    categorical,
    seed,
):
    return Pipeline(
        [
            (
                "preprocess",
                make_preprocessor(
                    numeric,
                    categorical,
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=350,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    bootstrap=True,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def regressor_pipeline(
    numeric,
    categorical,
    seed,
):
    return Pipeline(
        [
            (
                "preprocess",
                make_preprocessor(
                    numeric,
                    categorical,
                ),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=350,
                    min_samples_leaf=2,
                    bootstrap=True,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def ensemble_classifier(
    X_train,
    y_train,
    X_test,
    numeric,
    categorical,
):
    predictions = []

    for seed in ENSEMBLE_SEEDS:
        model = classifier_pipeline(
            numeric,
            categorical,
            seed,
        )

        model.fit(
            X_train,
            y_train,
        )

        p = model.predict_proba(
            X_test
        )[:, 1]

        predictions.append(p)

    pred = np.vstack(predictions)

    return (
        pred.mean(axis=0),
        pred.std(axis=0),
    )


def ensemble_regressor(
    X_train,
    y_train,
    X_test,
    numeric,
    categorical,
):
    predictions = []

    for seed in ENSEMBLE_SEEDS:
        model = regressor_pipeline(
            numeric,
            categorical,
            seed,
        )

        model.fit(
            X_train,
            y_train,
        )

        predictions.append(
            model.predict(X_test)
        )

    pred = np.vstack(predictions)

    return (
        pred.mean(axis=0),
        pred.std(axis=0),
    )


def entropy_binary(p):
    p = np.clip(
        p,
        1e-8,
        1.0 - 1e-8,
    )

    return -(
        p * np.log(p)
        + (1.0 - p)
        * np.log(1.0 - p)
    )


def main():
    args = parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    oracle = pd.read_csv(
        args.oracle
    )

    profiles = pd.read_csv(
        args.profiles
    )

    if profiles["request_id"].nunique() != 290:
        raise RuntimeError(
            "Expected 290 request profiles."
        )

    # -----------------------------------------------------
    # REQUEST FEATURES
    # -----------------------------------------------------

    available_request_features = [
        x
        for x in REQUEST_FEATURES
        if x in oracle.columns
    ]

    request_features = (
        oracle
        .sort_values("request_id")
        .groupby(
            "request_id",
            as_index=False,
        )[available_request_features]
        .first()
    )

    request_table = profiles.merge(
        request_features,
        on="request_id",
        how="left",
        suffixes=("", "_oracle"),
    )

    # -----------------------------------------------------
    # FIXED OUTER REQUEST FOLDS
    # -----------------------------------------------------

    request_ids = (
        request_table[
            "request_id"
        ]
        .values
    )

    outer = GroupKFold(
        n_splits=args.folds
    )

    fold_map = {}

    dummy = np.zeros(
        len(request_table)
    )

    for fold, (_, test) in enumerate(
        outer.split(
            dummy,
            groups=request_ids,
        ),
        start=1,
    ):
        for idx in test:
            fold_map[
                request_ids[idx]
            ] = fold

    request_table["fold"] = (
        request_table[
            "request_id"
        ].map(fold_map)
    )

    # =====================================================
    # REQUEST-LEVEL HEADROOM + FRAGILITY PREDICTION
    # =====================================================

    request_numeric = [
        x
        for x in available_request_features
        if x in request_table.columns
    ]

    request_categorical = []

    request_prediction_rows = []

    for fold in sorted(
        request_table["fold"].unique()
    ):
        train = request_table[
            request_table["fold"] != fold
        ].copy()

        test = request_table[
            request_table["fold"] == fold
        ].copy()

        X_train = train[
            request_numeric
        ]

        X_test = test[
            request_numeric
        ]

        result = test[
            [
                "request_id",
                "task",
                "fold",
                "fragility_score",
                "max_adjacent_jump",
                "cliff",
            ]
        ].copy()

        # ---------------------------------------------
        # Headroom prediction for each delta
        # ---------------------------------------------
        for delta in DELTAS:
            tag = (
                f"{delta:.2f}"
                .replace(".", "p")
            )

            target = (
                f"headroom_{tag}"
            )

            mean_pred, std_pred = (
                ensemble_regressor(
                    X_train,
                    train[target],
                    X_test,
                    request_numeric,
                    request_categorical,
                )
            )

            mean_pred = np.clip(
                mean_pred,
                0.0,
                0.75,
            )

            std_pred = np.maximum(
                std_pred,
                0.0,
            )

            result[
                f"headroom_true_{tag}"
            ] = test[target].values

            result[
                f"headroom_pred_{tag}"
            ] = mean_pred

            result[
                f"headroom_std_{tag}"
            ] = std_pred

            # Raw headroom prediction and uncertainty are the
            # primary outputs. This LCB is convenience-only.
            result[
                f"headroom_lcb_{tag}"
            ] = np.clip(
                mean_pred
                - args.convenience_lcb_lambda
                * std_pred,
                0.0,
                0.75,
            )

            result[
                f"headroom_lcb_lambda_{tag}"
            ] = args.convenience_lcb_lambda

        # ---------------------------------------------
        # Fragility regression
        # ---------------------------------------------
        frag_mean, frag_std = (
            ensemble_regressor(
                X_train,
                train["fragility_score"],
                X_test,
                request_numeric,
                request_categorical,
            )
        )

        result[
            "fragility_pred"
        ] = frag_mean

        result[
            "fragility_std"
        ] = frag_std

        # ---------------------------------------------
        # Compression-cliff probability
        # ---------------------------------------------
        cliff_mean, cliff_std = (
            ensemble_classifier(
                X_train,
                train["cliff"].astype(int),
                X_test,
                request_numeric,
                request_categorical,
            )
        )

        result[
            "cliff_probability"
        ] = cliff_mean

        result[
            "cliff_uncertainty"
        ] = cliff_std

        request_prediction_rows.append(
            result
        )

    request_predictions = pd.concat(
        request_prediction_rows,
        ignore_index=True,
    )

    # =====================================================
    # REQUEST × POLICY RISK + UNCERTAINTY
    # =====================================================

    compressed = oracle[
        oracle["policy"] != "full"
    ].copy()

    compressed["fold"] = (
        compressed[
            "request_id"
        ].map(fold_map)
    )

    policy_numeric = [
        x
        for x in POLICY_NUMERIC
        if x in compressed.columns
    ]

    policy_categorical = [
        x
        for x in POLICY_CATEGORICAL
        if x in compressed.columns
    ]

    risk_numeric = (
        policy_numeric
        + available_request_features
    )

    risk_features = (
        risk_numeric
        + policy_categorical
    )

    risk_prediction_rows = []

    for delta in DELTAS:
        work = compressed.copy()

        work["safe"] = (
            work["degradation"]
            <= delta + 1e-12
        ).astype(int)

        for fold in sorted(
            work["fold"].unique()
        ):
            train = work[
                work["fold"] != fold
            ].copy()

            test = work[
                work["fold"] == fold
            ].copy()

            # Internal calibration subset is kept separate
            # from model fitting.
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=0.20,
                random_state=1000 + int(fold),
            )

            fit_rel, calib_rel = next(
                splitter.split(
                    train,
                    groups=train[
                        "request_id"
                    ],
                )
            )

            fit = train.iloc[
                fit_rel
            ]

            calib = train.iloc[
                calib_rel
            ]

            calib_mean, calib_std = (
                ensemble_classifier(
                    fit[risk_features],
                    fit["safe"],
                    calib[risk_features],
                    risk_numeric,
                    policy_categorical,
                )
            )

            test_mean, test_std = (
                ensemble_classifier(
                    fit[risk_features],
                    fit["safe"],
                    test[risk_features],
                    risk_numeric,
                    policy_categorical,
                )
            )

            calib_lcb = np.clip(
                calib_mean
                - args.convenience_lcb_lambda
                * calib_std,
                0.0,
                1.0,
            )

            test_lcb = np.clip(
                test_mean
                - args.convenience_lcb_lambda
                * test_std,
                0.0,
                1.0,
            )

            result = test[
                [
                    "request_id",
                    "task",
                    "policy",
                    "family",
                    "memory_ratio",
                    "compression_ratio",
                    "quality",
                    "full_quality",
                    "degradation",
                    "fold",
                ]
            ].copy()

            result["delta"] = delta
            result["safe"] = (
                test["safe"].values
            )

            result[
                "prob_safe_mean"
            ] = test_mean

            # Primary uncertainty outputs.
            #
            # These raw quantities are what downstream controller
            # sweeps must use. No uncertainty weight is baked into
            # them.
            result[
                "prob_safe_mean"
            ] = test_mean

            result[
                "risk_uncertainty"
            ] = test_std

            # Convenience-only conservative score.
            #
            # This column is NOT the final controller decision rule.
            # We retain it only for quick inspection. The final
            # uncertainty coefficient is swept downstream.
            result[
                "prob_safe_lcb"
            ] = test_lcb

            result[
                "convenience_lcb_lambda"
            ] = args.convenience_lcb_lambda

            result[
                "predictive_entropy"
            ] = entropy_binary(
                test_mean
            )

            # Store calibration predictions as separate
            # sidecar file below.
            result[
                "calibration_fold"
            ] = fold

            risk_prediction_rows.append(
                result
            )

            calib_output = calib[
                [
                    "request_id",
                    "task",
                    "policy",
                    "family",
                    "memory_ratio",
                    "degradation",
                    "safe",
                ]
            ].copy()

            calib_output["delta"] = delta
            calib_output["fold"] = fold
            calib_output[
                "prob_safe_mean"
            ] = calib_mean

            calib_output[
                "risk_uncertainty"
            ] = calib_std

            calib_output[
                "prob_safe_lcb"
            ] = calib_lcb

            calib_path = (
                out
                / (
                    f"calibration_delta_"
                    f"{delta:.2f}_"
                    f"fold_{int(fold)}.csv"
                )
            )

            calib_output.to_csv(
                calib_path,
                index=False,
            )

    risk_predictions = pd.concat(
        risk_prediction_rows,
        ignore_index=True,
    )

    # =====================================================
    # METRICS
    # =====================================================

    risk_summary_rows = []

    for delta, g in risk_predictions.groupby(
        "delta"
    ):
        y = g["safe"].to_numpy(int)
        p = g["prob_safe_mean"].to_numpy(float)

        risk_summary_rows.append(
            {
                "delta": delta,
                "n_rows": len(g),
                "safe_rate": y.mean(),
                "auroc": (
                    roc_auc_score(y, p)
                    if len(np.unique(y)) == 2
                    else np.nan
                ),
                "auprc": (
                    average_precision_score(
                        y,
                        p,
                    )
                    if len(np.unique(y)) == 2
                    else np.nan
                ),
                "mean_risk_uncertainty": (
                    g[
                        "risk_uncertainty"
                    ].mean()
                ),
                "mean_predictive_entropy": (
                    g[
                        "predictive_entropy"
                    ].mean()
                ),
            }
        )

    headroom_summary_rows = []

    for delta in DELTAS:
        tag = (
            f"{delta:.2f}"
            .replace(".", "p")
        )

        true = request_predictions[
            f"headroom_true_{tag}"
        ]

        pred = request_predictions[
            f"headroom_pred_{tag}"
        ]

        lcb = request_predictions[
            f"headroom_lcb_{tag}"
        ]

        unsafe_overprediction = (
            lcb > true + 1e-12
        )

        headroom_summary_rows.append(
            {
                "delta": delta,
                "mae": mean_absolute_error(
                    true,
                    pred,
                ),
                "rmse": (
                    mean_squared_error(
                        true,
                        pred,
                    ) ** 0.5
                ),
                "r2": r2_score(
                    true,
                    pred,
                ),
                "spearman": (
                    pd.Series(true)
                    .corr(
                        pd.Series(pred),
                        method="spearman",
                    )
                ),
                "mean_true_headroom": (
                    true.mean()
                ),
                "mean_pred_headroom": (
                    pred.mean()
                ),
                "mean_conservative_headroom": (
                    lcb.mean()
                ),
                "unsafe_headroom_overprediction_rate": (
                    unsafe_overprediction.mean()
                ),
            }
        )

    frag_true = request_predictions[
        "fragility_score"
    ]

    frag_pred = request_predictions[
        "fragility_pred"
    ]

    fragility_summary = pd.DataFrame(
        [
            {
                "mae": mean_absolute_error(
                    frag_true,
                    frag_pred,
                ),
                "rmse": (
                    mean_squared_error(
                        frag_true,
                        frag_pred,
                    ) ** 0.5
                ),
                "r2": r2_score(
                    frag_true,
                    frag_pred,
                ),
                "spearman": (
                    frag_true.corr(
                        frag_pred,
                        method="spearman",
                    )
                ),
                "cliff_rate": (
                    request_predictions[
                        "cliff"
                    ].mean()
                ),
                "cliff_auroc": (
                    roc_auc_score(
                        request_predictions[
                            "cliff"
                        ],
                        request_predictions[
                            "cliff_probability"
                        ],
                    )
                    if request_predictions[
                        "cliff"
                    ].nunique() == 2
                    else np.nan
                ),
                "cliff_auprc": (
                    average_precision_score(
                        request_predictions[
                            "cliff"
                        ],
                        request_predictions[
                            "cliff_probability"
                        ],
                    )
                    if request_predictions[
                        "cliff"
                    ].nunique() == 2
                    else np.nan
                ),
            }
        ]
    )

    risk_summary = pd.DataFrame(
        risk_summary_rows
    )

    headroom_summary = pd.DataFrame(
        headroom_summary_rows
    )

    request_predictions.to_csv(
        out / "request_profile_oof.csv",
        index=False,
    )

    risk_predictions.to_csv(
        out / "policy_risk_oof.csv",
        index=False,
    )

    risk_summary.to_csv(
        out / "risk_uncertainty_summary.csv",
        index=False,
    )

    headroom_summary.to_csv(
        out / "headroom_prediction_summary.csv",
        index=False,
    )

    fragility_summary.to_csv(
        out / "fragility_prediction_summary.csv",
        index=False,
    )

    print("\nRISK + UNCERTAINTY")
    print(
        risk_summary
        .round(4)
        .to_string(index=False)
    )

    print("\nHEADROOM PREDICTION")
    print(
        headroom_summary
        .round(4)
        .to_string(index=False)
    )

    print("\nFRAGILITY / CLIFF PREDICTION")
    print(
        fragility_summary
        .round(4)
        .to_string(index=False)
    )

    print(
        "\nREQUEST COMPRESSION PROFILE PREDICTION: PASS"
    )


if __name__ == "__main__":
    main()
