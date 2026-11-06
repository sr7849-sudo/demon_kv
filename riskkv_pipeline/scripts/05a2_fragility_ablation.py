from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


POLICY_CURVES = {
    "snapkv": [
        "full",
        "snapkv_25",
        "snapkv_50",
        "snapkv_75",
    ],
    "knorm": [
        "full",
        "knorm_25",
        "knorm_50",
        "knorm_75",
    ],
}


CLIFF_THRESHOLDS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
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
            "analysis/08b_fragility_ablation"
        ),
    )

    p.add_argument(
        "--catastrophic-threshold",
        type=float,
        default=0.25,
    )

    return p.parse_args()


def get_curve(group, policies):
    rows = []

    for policy in policies:
        x = group[
            group["policy"] == policy
        ]

        if len(x) != 1:
            return None

        r = x.iloc[0]

        rows.append(
            {
                "policy": policy,
                "severity": (
                    1.0
                    - float(
                        r["memory_ratio"]
                    )
                ),
                "degradation": float(
                    r["degradation"]
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("severity")
        .reset_index(drop=True)
    )


def compute_family_metrics(curve):
    severity = curve[
        "severity"
    ].to_numpy(float)

    degradation = curve[
        "degradation"
    ].to_numpy(float)

    ds = np.diff(severity)
    dd = np.diff(degradation)

    positive_jump = np.maximum(
        dd,
        0.0,
    )

    slopes = np.divide(
        positive_jump,
        ds,
        out=np.zeros_like(
            positive_jump
        ),
        where=ds > 0,
    )

    auc = np.trapezoid(
        degradation,
        severity,
    )

    return {
        "max_positive_slope": float(
            slopes.max()
        ),
        "max_adjacent_jump": float(
            positive_jump.max()
        ),
        "degradation_auc": float(
            auc
        ),
        "final_degradation": float(
            degradation[-1]
        ),
        "max_degradation": float(
            degradation.max()
        ),
        "monotonicity_violations": int(
            (dd < -1e-12).sum()
        ),
    }


def safe_auroc(y, score):
    if len(
        np.unique(y)
    ) != 2:
        return np.nan

    return roc_auc_score(
        y,
        score,
    )


def safe_auprc(y, score):
    if len(
        np.unique(y)
    ) != 2:
        return np.nan

    return average_precision_score(
        y,
        score,
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

    if (
        df[
            "request_id"
        ].nunique()
        != 290
    ):
        raise RuntimeError(
            "Expected 290 requests."
        )

    compressed = df[
        df["policy"] != "full"
    ].copy()

    rows = []

    # =====================================================
    # BUILD RAW FAMILY-WISE CURVE METRICS
    # =====================================================

    for request_id, group in df.groupby(
        "request_id",
        sort=False,
    ):
        compressed_group = group[
            group["policy"]
            != "full"
        ]

        base = {
            "request_id": request_id,
            "task": group[
                "task"
            ].iloc[0],
            "context_tokens": group[
                "context_tokens"
            ].iloc[0],
            "max_degradation_any": (
                compressed_group[
                    "degradation"
                ].max()
            ),
            "mean_degradation_any": (
                compressed_group[
                    "degradation"
                ].mean()
            ),
            "catastrophic_any": int(
                compressed_group[
                    "degradation"
                ].max()
                >= args.catastrophic_threshold
            ),
        }

        family_metrics = {}

        for family, policies in (
            POLICY_CURVES.items()
        ):
            curve = get_curve(
                group,
                policies,
            )

            if curve is None:
                continue

            family_metrics[
                family
            ] = compute_family_metrics(
                curve
            )

        if len(
            family_metrics
        ) != len(
            POLICY_CURVES
        ):
            raise RuntimeError(
                f"Incomplete family curves "
                f"for {request_id}"
            )

        for family, metrics in (
            family_metrics.items()
        ):
            for key, value in (
                metrics.items()
            ):
                base[
                    f"{family}_{key}"
                ] = value

        # ---------------------------------------------
        # Aggregate each candidate fragility definition
        # across policy families using BOTH max and mean.
        # ---------------------------------------------

        candidate_metrics = [
            "max_positive_slope",
            "max_adjacent_jump",
            "degradation_auc",
            "final_degradation",
        ]

        for metric in candidate_metrics:
            vals = np.array(
                [
                    family_metrics[
                        family
                    ][metric]
                    for family
                    in POLICY_CURVES
                ],
                dtype=float,
            )

            base[
                f"{metric}_agg_max"
            ] = float(
                vals.max()
            )

            base[
                f"{metric}_agg_mean"
            ] = float(
                vals.mean()
            )

        rows.append(
            base
        )

    profile = pd.DataFrame(
        rows
    )

    profile.to_csv(
        out
        / "raw_fragility_ablation_profiles.csv",
        index=False,
    )

    # =====================================================
    # FRAGILITY DEFINITION ABLATION
    # =====================================================

    definition_rows = []

    candidate_definitions = []

    for metric in [
        "max_positive_slope",
        "max_adjacent_jump",
        "degradation_auc",
        "final_degradation",
    ]:
        for aggregation in [
            "max",
            "mean",
        ]:
            candidate_definitions.append(
                (
                    metric,
                    aggregation,
                    f"{metric}_agg_{aggregation}",
                )
            )

    for (
        definition,
        aggregation,
        column,
    ) in candidate_definitions:

        score = profile[
            column
        ]

        y_cat = profile[
            "catastrophic_any"
        ].to_numpy(int)

        # Relationship to continuous maximum loss.
        spearman_max_deg = (
            score.corr(
                profile[
                    "max_degradation_any"
                ],
                method="spearman",
            )
        )

        spearman_mean_deg = (
            score.corr(
                profile[
                    "mean_degradation_any"
                ],
                method="spearman",
            )
        )

        # How useful is this scalar for identifying requests
        # that experience at least one catastrophic policy?
        auroc = safe_auroc(
            y_cat,
            score.to_numpy(float),
        )

        auprc = safe_auprc(
            y_cat,
            score.to_numpy(float),
        )

        # Stability across tasks:
        # compute task-wise Spearman and summarize.
        task_rhos = []

        for task, g in profile.groupby(
            "task"
        ):
            if (
                g[column].nunique()
                < 2
                or g[
                    "max_degradation_any"
                ].nunique()
                < 2
            ):
                continue

            rho = (
                g[column]
                .corr(
                    g[
                        "max_degradation_any"
                    ],
                    method="spearman",
                )
            )

            if not np.isnan(rho):
                task_rhos.append(
                    rho
                )

        definition_rows.append(
            {
                "definition": definition,
                "aggregation": aggregation,
                "column": column,
                "mean": score.mean(),
                "std": score.std(),
                "zero_rate": (
                    score == 0
                ).mean(),
                "spearman_max_degradation": (
                    spearman_max_deg
                ),
                "spearman_mean_degradation": (
                    spearman_mean_deg
                ),
                "catastrophic_auroc": auroc,
                "catastrophic_auprc": auprc,
                "mean_task_spearman": (
                    np.mean(
                        task_rhos
                    )
                    if task_rhos
                    else np.nan
                ),
                "min_task_spearman": (
                    np.min(
                        task_rhos
                    )
                    if task_rhos
                    else np.nan
                ),
                "std_task_spearman": (
                    np.std(
                        task_rhos
                    )
                    if task_rhos
                    else np.nan
                ),
            }
        )

    definitions = pd.DataFrame(
        definition_rows
    )

    definitions.to_csv(
        out
        / "fragility_definition_ablation.csv",
        index=False,
    )

    # =====================================================
    # CLIFF THRESHOLD ABLATION
    # =====================================================
    #
    # Cliff is currently defined from MAX adjacent
    # positive degradation jump across ordered families.
    #
    # We are not treating catastrophic_any as the definition
    # of cliff. It is an external severity signal used only
    # to understand how useful each candidate threshold is.
    # =====================================================

    cliff_score = profile[
        "max_adjacent_jump_agg_max"
    ]

    cliff_rows = []
    cliff_task_rows = []

    catastrophic = (
        profile[
            "catastrophic_any"
        ].to_numpy(int)
    )

    for tau in CLIFF_THRESHOLDS:
        cliff = (
            cliff_score
            >= tau
        ).astype(int)

        tp = int(
            (
                (cliff == 1)
                & (catastrophic == 1)
            ).sum()
        )

        fp = int(
            (
                (cliff == 1)
                & (catastrophic == 0)
            ).sum()
        )

        fn = int(
            (
                (cliff == 0)
                & (catastrophic == 1)
            ).sum()
        )

        tn = int(
            (
                (cliff == 0)
                & (catastrophic == 0)
            ).sum()
        )

        precision = precision_score(
            catastrophic,
            cliff,
            zero_division=0,
        )

        recall = recall_score(
            catastrophic,
            cliff,
            zero_division=0,
        )

        jaccard = (
            tp
            / (tp + fp + fn)
            if (
                tp + fp + fn
            ) > 0
            else 0.0
        )

        cliff_rows.append(
            {
                "cliff_threshold": tau,
                "cliff_rate": cliff.mean(),
                "catastrophic_rate": (
                    catastrophic.mean()
                ),
                "precision_vs_catastrophic": (
                    precision
                ),
                "recall_vs_catastrophic": (
                    recall
                ),
                "jaccard_vs_catastrophic": (
                    jaccard
                ),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )

        tmp = profile[
            [
                "request_id",
                "task",
            ]
        ].copy()

        tmp["cliff"] = (
            cliff.values
        )

        for task, g in tmp.groupby(
            "task"
        ):
            cliff_task_rows.append(
                {
                    "cliff_threshold": tau,
                    "task": task,
                    "n_requests": len(g),
                    "cliff_rate": (
                        g[
                            "cliff"
                        ].mean()
                    ),
                }
            )

    cliff_summary = pd.DataFrame(
        cliff_rows
    )

    cliff_by_task = pd.DataFrame(
        cliff_task_rows
    )

    cliff_summary.to_csv(
        out
        / "cliff_threshold_ablation.csv",
        index=False,
    )

    cliff_by_task.to_csv(
        out
        / "cliff_threshold_by_task.csv",
        index=False,
    )

    # =====================================================
    # PRINT
    # =====================================================

    print(
        "\nFRAGILITY DEFINITION ABLATION"
    )

    print(
        definitions
        .sort_values(
            [
                "catastrophic_auroc",
                "spearman_max_degradation",
            ],
            ascending=False,
        )
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        "\nCLIFF THRESHOLD ABLATION"
    )

    print(
        cliff_summary
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Do not choose a threshold solely because "
        "it maximizes one number. Prefer a definition "
        "that is interpretable, produces a non-degenerate "
        "label distribution, and is reasonably stable "
        "across tasks."
    )

    print(
        "\nFRAGILITY ABLATION: PASS"
    )


if __name__ == "__main__":
    main()
