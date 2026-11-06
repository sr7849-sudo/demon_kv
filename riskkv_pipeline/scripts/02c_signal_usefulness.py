from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


REQUEST_FEATURES = [
    "context_tokens",
    "prefill_seconds",
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
            "analysis/03_signal_usefulness"
        ),
    )
    p.add_argument(
        "--folds",
        type=int,
        default=5,
    )
    return p.parse_args()


def make_pipeline(
    numeric_features,
    categorical_features,
):
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
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
                    steps=[
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
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        [
            ("preprocess", pre),
            ("model", model),
        ]
    )


def grouped_cv(
    df,
    numeric,
    categorical,
    folds,
    name,
):
    groups = df["request_id"].values
    target = df["degradation"].values

    columns = numeric + categorical
    X = df[columns].copy()

    splitter = GroupKFold(
        n_splits=folds,
    )

    preds = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    fold_rows = []

    for fold, (train, test) in enumerate(
        splitter.split(
            X,
            target,
            groups,
        ),
        start=1,
    ):
        pipe = make_pipeline(
            numeric,
            categorical,
        )

        pipe.fit(
            X.iloc[train],
            target[train],
        )

        pred = pipe.predict(
            X.iloc[test]
        )

        preds[test] = pred

        mae = mean_absolute_error(
            target[test],
            pred,
        )
        rmse = mean_squared_error(
            target[test],
            pred,
        ) ** 0.5
        r2 = r2_score(
            target[test],
            pred,
        )

        fold_rows.append(
            {
                "model": name,
                "fold": fold,
                "n_train_rows": len(train),
                "n_test_rows": len(test),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            }
        )

    if np.isnan(preds).any():
        raise RuntimeError(
            f"Missing CV predictions for {name}"
        )

    overall = {
        "model": name,
        "fold": "ALL",
        "n_train_rows": np.nan,
        "n_test_rows": len(df),
        "mae": mean_absolute_error(
            target,
            preds,
        ),
        "rmse": mean_squared_error(
            target,
            preds,
        ) ** 0.5,
        "r2": r2_score(
            target,
            preds,
        ),
    }

    return (
        pd.DataFrame(
            fold_rows + [overall]
        ),
        preds,
    )


def main():
    args = parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    # FullKV is excluded because its degradation
    # is defined to be exactly zero.
    df = df[
        df["policy"] != "full"
    ].copy()

    available = [
        f
        for f in REQUEST_FEATURES
        if f in df.columns
    ]

    missing = sorted(
        set(REQUEST_FEATURES)
        - set(available)
    )

    if missing:
        print(
            "Warning: unavailable request features:",
            missing,
        )

    # Correlations are computed separately for each policy
    # to avoid confusing policy severity with request sensitivity.
    corr_rows = []

    for policy, g in df.groupby("policy"):
        for feature in available:
            x = g[feature]
            y = g["degradation"]

            valid = x.notna() & y.notna()

            if valid.sum() < 10:
                continue

            rho = (
                x[valid]
                .corr(
                    y[valid],
                    method="spearman",
                )
            )

            corr_rows.append(
                {
                    "policy": policy,
                    "feature": feature,
                    "spearman_rho": rho,
                    "abs_spearman_rho": abs(rho),
                    "n": int(valid.sum()),
                }
            )

    correlations = pd.DataFrame(
        corr_rows
    ).sort_values(
        [
            "policy",
            "abs_spearman_rho",
        ],
        ascending=[True, False],
    )

    correlations.to_csv(
        out / "policy_feature_spearman.csv",
        index=False,
    )

    # Policy-only baseline:
    # "How much can we predict simply from which compression
    # configuration was chosen?"
    baseline_numeric = [
        "memory_ratio",
        "compression_ratio",
    ]

    baseline_numeric = [
        x
        for x in baseline_numeric
        if x in df.columns
    ]

    baseline_categorical = [
        "policy",
        "family",
    ]

    baseline_cv, baseline_pred = grouped_cv(
        df,
        baseline_numeric,
        baseline_categorical,
        args.folds,
        "policy_only",
    )

    # Add request-specific prefill features.
    full_numeric = (
        baseline_numeric
        + available
    )

    full_categorical = (
        baseline_categorical
    )

    full_cv, full_pred = grouped_cv(
        df,
        full_numeric,
        full_categorical,
        args.folds,
        "policy_plus_prefill",
    )

    cv = pd.concat(
        [
            baseline_cv,
            full_cv,
        ],
        ignore_index=True,
    )

    cv.to_csv(
        out / "grouped_cv_signal_test.csv",
        index=False,
    )

    prediction_rows = df[
        [
            "request_id",
            "task",
            "policy",
            "degradation",
        ]
    ].copy()

    prediction_rows[
        "pred_policy_only"
    ] = baseline_pred

    prediction_rows[
        "pred_policy_plus_prefill"
    ] = full_pred

    prediction_rows.to_csv(
        out
        / "grouped_cv_predictions.csv",
        index=False,
    )

    print("\nGROUPED REQUEST-LEVEL CV")
    print(
        cv[
            cv["fold"].astype(str) == "ALL"
        ]
        .round(5)
        .to_string(index=False)
    )

    top = (
        correlations
        .sort_values(
            "abs_spearman_rho",
            ascending=False,
        )
        .head(30)
    )

    print("\nSTRONGEST POLICY-SPECIFIC SIGNALS")
    print(
        top.round(4).to_string(index=False)
    )

    print("\nSIGNAL USEFULNESS: PASS")


if __name__ == "__main__":
    main()
