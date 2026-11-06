"""
EXPERIMENT 5
============

Leave-one-task-out compression-risk generalization.

Purpose
-------
Tests whether RiskKV learns general request/compression sensitivity,
rather than simply memorizing task identity.

For every LongBench task:
  - remove that task entirely from training;
  - train using all other tasks;
  - evaluate on the unseen task.

Two predictors are compared:
  A. policy-only
  B. policy + cheap prefill features

Task identity itself is deliberately excluded.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
)
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


ORACLE = (
    "experiments/full_oracle_qwen3_8b/"
    "oracle_longbench_qwen3_8b.csv"
)

OUTDIR = Path(
    "experiments/full_oracle_qwen3_8b/"
    "analysis/19_cross_task_generalization"
)


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


DELTAS = [
    0.01,
    0.05,
    0.10,
]


def build_features(
    df,
    use_prefill,
    columns=None,
):
    base = pd.DataFrame(
        index=df.index
    )

    # Candidate-policy identity must be known because
    # risk is explicitly policy-conditioned.
    policy = pd.get_dummies(
        df["policy"],
        prefix="policy",
        dtype=float,
    )

    base = pd.concat(
        [
            base,
            policy,
        ],
        axis=1,
    )

    for col in [
        "memory_ratio",
    ]:
        if col in df.columns:
            base[col] = (
                df[col]
                .astype(float)
            )

    if use_prefill:
        for col in NUMERIC_PREFILL:
            if col in df.columns:
                base[col] = pd.to_numeric(
                    df[col],
                    errors="coerce",
                )

    base = base.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    base = base.fillna(
        base.median(
            numeric_only=True
        )
    )

    # Match training feature columns exactly at test.
    if columns is not None:
        base = base.reindex(
            columns=columns,
            fill_value=0.0,
        )

    return base


def safe_metric(fn, y, p):
    if len(np.unique(y)) < 2:
        return np.nan

    return fn(y, p)


def main():
    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        ORACLE
    )

    # FullKV is not a compression action and has
    # degradation identically zero; exclude it from
    # compression-risk prediction.
    df = df[
        df.policy != "full"
    ].copy()

    tasks = sorted(
        df.task.unique()
    )

    rows = []
    preds = []

    for delta in DELTAS:

        df["safe"] = (
            df["degradation"]
            <= delta + 1e-12
        ).astype(int)

        for heldout in tasks:

            train = df[
                df.task != heldout
            ].copy()

            test = df[
                df.task == heldout
            ].copy()

            for feature_set, use_prefill in [
                (
                    "policy_only",
                    False,
                ),
                (
                    "all_prefill",
                    True,
                ),
            ]:
                X_train = build_features(
                    train,
                    use_prefill,
                )

                X_test = build_features(
                    test,
                    use_prefill,
                    columns=X_train.columns,
                )

                model = (
                    HistGradientBoostingClassifier(
                        max_iter=200,
                        learning_rate=0.05,
                        max_leaf_nodes=15,
                        l2_regularization=1.0,
                        random_state=42,
                    )
                )

                model.fit(
                    X_train,
                    train["safe"],
                )

                prob = model.predict_proba(
                    X_test
                )[:, 1]

                y = test[
                    "safe"
                ].to_numpy(int)

                rows.append(
                    {
                        "delta": delta,
                        "heldout_task": (
                            heldout
                        ),
                        "feature_set": (
                            feature_set
                        ),
                        "n_test": len(test),
                        "safe_rate": (
                            y.mean()
                        ),
                        "auroc": safe_metric(
                            roc_auc_score,
                            y,
                            prob,
                        ),
                        "auprc": safe_metric(
                            average_precision_score,
                            y,
                            prob,
                        ),
                        "brier": (
                            brier_score_loss(
                                y,
                                prob,
                            )
                        ),
                    }
                )

                for i, (_, r) in enumerate(
                    test.iterrows()
                ):
                    preds.append(
                        {
                            "delta": delta,
                            "heldout_task": (
                                heldout
                            ),
                            "feature_set": (
                                feature_set
                            ),
                            "request_id": (
                                r[
                                    "request_id"
                                ]
                            ),
                            "policy": (
                                r["policy"]
                            ),
                            "safe": int(
                                r["safe"]
                            ),
                            "prob_safe": (
                                prob[i]
                            ),
                        }
                    )

    result = pd.DataFrame(
        rows
    )

    overall = (
        result.groupby(
            [
                "delta",
                "feature_set",
            ]
        )
        .agg(
            mean_auroc=(
                "auroc",
                "mean",
            ),
            min_auroc=(
                "auroc",
                "min",
            ),
            mean_auprc=(
                "auprc",
                "mean",
            ),
            mean_brier=(
                "brier",
                "mean",
            ),
        )
        .reset_index()
    )

    result.to_csv(
        OUTDIR
        / "leave_one_task_out.csv",
        index=False,
    )

    overall.to_csv(
        OUTDIR
        / "cross_task_summary.csv",
        index=False,
    )

    pd.DataFrame(
        preds
    ).to_csv(
        OUTDIR
        / "cross_task_predictions.csv",
        index=False,
    )

    print(
        "\nLEAVE-ONE-TASK-OUT RESULTS"
    )

    print(
        result.round(4)
        .to_string(index=False)
    )

    print(
        "\nCROSS-TASK SUMMARY"
    )

    print(
        overall.round(4)
        .to_string(index=False)
    )

    print(
        "\nEXPERIMENT 5: PASS"
    )


if __name__ == "__main__":
    main()
