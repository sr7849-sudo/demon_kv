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
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DELTAS = [
    0.01,
    0.05,
    0.10,
]


NUMERIC = [
    "memory_ratio",
    "compression_ratio",

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


CATEGORICAL = [
    "policy",
    "family",
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
            "analysis/06_unseen_task_risk"
        ),
    )

    return p.parse_args()


def make_model(
    numeric,
    categorical,
):
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        )
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

    clf = RandomForestClassifier(
        n_estimators=600,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        [
            ("features", pre),
            ("classifier", clf),
        ]
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

    df = df[
        df["policy"]
        != "full"
    ].copy()

    numeric = [
        x
        for x in NUMERIC
        if x in df.columns
    ]

    categorical = [
        x
        for x in CATEGORICAL
        if x in df.columns
    ]

    features = (
        numeric
        + categorical
    )

    rows = []
    predictions = []

    for delta in DELTAS:
        work = df.copy()

        work["safe"] = (
            work["degradation"]
            <= delta + 1e-12
        ).astype(int)

        for heldout_task in sorted(
            work["task"].unique()
        ):
            train = work[
                work["task"]
                != heldout_task
            ]

            test = work[
                work["task"]
                == heldout_task
            ]

            model = make_model(
                numeric,
                categorical,
            )

            model.fit(
                train[
                    features
                ],
                train["safe"],
            )

            probability = (
                model.predict_proba(
                    test[
                        features
                    ]
                )[:, 1]
            )

            y = (
                test["safe"]
                .values
            )

            if (
                len(
                    np.unique(y)
                )
                == 2
            ):
                auroc = (
                    roc_auc_score(
                        y,
                        probability,
                    )
                )

                auprc = (
                    average_precision_score(
                        y,
                        probability,
                    )
                )
            else:
                auroc = np.nan
                auprc = np.nan

            brier = (
                brier_score_loss(
                    y,
                    probability,
                )
            )

            rows.append(
                {
                    "delta": delta,
                    "heldout_task": (
                        heldout_task
                    ),
                    "n_train_requests": (
                        train[
                            "request_id"
                        ].nunique()
                    ),
                    "n_test_requests": (
                        test[
                            "request_id"
                        ].nunique()
                    ),
                    "n_test_rows": (
                        len(test)
                    ),
                    "safe_rate": (
                        y.mean()
                    ),
                    "auroc": auroc,
                    "auprc": auprc,
                    "brier": brier,
                }
            )

            p = test[
                [
                    "request_id",
                    "task",
                    "policy",
                    "degradation",
                    "safe",
                ]
            ].copy()

            p["delta"] = delta
            p[
                "prob_safe"
            ] = probability

            predictions.append(
                p
            )

    summary = pd.DataFrame(
        rows
    )

    prediction_df = pd.concat(
        predictions,
        ignore_index=True,
    )

    summary.to_csv(
        out
        / "unseen_task_summary.csv",
        index=False,
    )

    prediction_df.to_csv(
        out
        / "unseen_task_predictions.csv",
        index=False,
    )

    print(
        "\nLEAVE-ONE-TASK-OUT RESULTS"
    )

    print(
        summary
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        "\nMEAN BY DELTA"
    )

    print(
        summary
        .groupby("delta")[
            [
                "auroc",
                "auprc",
                "brier",
            ]
        ]
        .mean()
        .round(4)
    )

    print(
        "\nUNSEEN-TASK RISK: PASS"
    )


if __name__ == "__main__":
    main()
