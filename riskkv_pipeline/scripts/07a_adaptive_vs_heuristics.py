"""
EXPERIMENT 1
============

Static / task-aware / length-aware vs adaptive RiskKV.

Purpose
-------
Tests whether request-conditioned adaptation is actually necessary.

Baselines:
  1. Best global static policy.
  2. Best policy conditioned only on task.
  3. Best policy conditioned only on context-length bucket.
  4. Risk-only adaptive controller.
  5. Risk + uncertainty adaptive controller.
  6. Perfect oracle.

IMPORTANT:
Task and length baselines are selected using OTHER folds only.
They do not inspect held-out request outcomes.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from riskkv.synthesis_eval import (
    LAMBDAS,
    calibrate_threshold,
    conservative_score,
)


RISK = (
    "experiments/full_oracle_qwen3_8b/"
    "analysis/09_profile_prediction/"
    "policy_risk_oof.csv"
)

OUTDIR = Path(
    "experiments/full_oracle_qwen3_8b/"
    "analysis/15_adaptive_vs_heuristics"
)


DELTAS = [0.01, 0.05, 0.10]
TARGETS = [0.01, 0.05]


def add_length_bucket(df):
    """
    Quartile-style length categories.

    Thresholds are derived globally here only from context length,
    never from quality labels.
    """
    x = df.copy()

    q = (
        x[
            ["request_id", "context_tokens"]
        ]
        .drop_duplicates()
        ["context_tokens"]
    )

    q1, q2, q3 = q.quantile(
        [0.25, 0.50, 0.75]
    )

    x["length_bucket"] = pd.cut(
        x["context_tokens"],
        bins=[
            -np.inf,
            q1,
            q2,
            q3,
            np.inf,
        ],
        labels=[
            "short",
            "medium",
            "long",
            "very_long",
        ],
    )

    return x


def choose_best_group_policy(
    train,
    group_col,
    delta,
    risk_target,
):
    """
    Learn a simple lookup baseline:

        task -> best policy

    or:

        length bucket -> best policy

    The selected policy minimizes nominal memory ratio among policies
    whose TRAINING violation rate satisfies the risk target.
    """
    table = {}

    for group_value, g in train.groupby(
        group_col,
        dropna=False,
        observed=True,
    ):
        stats = (
            g.groupby("policy")
            .agg(
                violation_rate=(
                    "safe",
                    lambda x: 1.0 - x.mean(),
                ),
                memory_ratio=(
                    "memory_ratio",
                    "mean",
                ),
            )
            .reset_index()
        )

        feasible = stats[
            stats["violation_rate"]
            <= risk_target + 1e-12
        ]

        if len(feasible):
            chosen = (
                feasible.sort_values(
                    [
                        "memory_ratio",
                        "violation_rate",
                        "policy",
                    ]
                )
                .iloc[0]["policy"]
            )
        else:
            chosen = "full"

        table[group_value] = chosen

    return table


def evaluate(rows, delta):
    rows = pd.DataFrame(rows)

    return {
        "n_requests": len(rows),

        "mean_memory_saved_ratio": (
            1.0
            - rows[
                "memory_ratio"
            ].mean()
        ),

        "violation_rate": (
            rows[
                "degradation"
            ]
            > delta + 1e-12
        ).mean(),

        "catastrophic_rate": (
            rows[
                "degradation"
            ]
            >= 0.25
        ).mean(),

        "mean_degradation": rows[
            "degradation"
        ].mean(),

        "fullkv_rate": rows[
            "policy"
        ].eq("full").mean(),
    }


def make_fullkv_row(g):
    """
    Construct the FullKV fallback for a request.

    policy_risk_oof.csv intentionally contains only the seven
    compressed candidate policies (290 requests x 7 = 2030 rows).
    FullKV therefore has no learned risk-prediction row.

    This is not an approximation of its quality:
    FullKV is the reference configuration against which degradation
    was defined, so by construction:

        degradation = 0
        memory_ratio = 1

    Other request metadata is copied from the current request.
    """
    if len(g) == 0:
        raise RuntimeError(
            "Cannot construct FullKV fallback from an empty request group."
        )

    r = g.iloc[0].copy()

    r["policy"] = "full"
    r["family"] = "full"
    r["memory_ratio"] = 1.0
    r["degradation"] = 0.0
    r["safe"] = 1

    # FullKV itself does not need a learned compression-risk score.
    if "prob_safe_mean" in r.index:
        r["prob_safe_mean"] = 1.0

    if "risk_uncertainty" in r.index:
        r["risk_uncertainty"] = 0.0

    return r


def pick_request_policy(g, policy):
    """
    Return a requested compressed policy when available.

    If the baseline chooses FullKV, or if a requested policy is not
    available for some reason, use the guaranteed FullKV fallback.
    """
    if policy == "full":
        return make_fullkv_row(g)

    x = g[
        g["policy"] == policy
    ]

    if len(x):
        return x.iloc[0]

    return make_fullkv_row(g)


def main():
    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    risk = pd.read_csv(RISK)

    # -----------------------------------------------------
    # Attach request context length.
    #
    # policy_risk_oof.csv contains the OOF predictions but
    # not the original context_tokens field needed for the
    # length-aware heuristic baseline.
    #
    # Pull context length from the frozen oracle dataset,
    # using request_id as the request-level key.
    # -----------------------------------------------------

    oracle = pd.read_csv(
        "experiments/full_oracle_qwen3_8b/"
        "oracle_longbench_qwen3_8b.csv"
    )

    request_meta = (
        oracle[
            [
                "request_id",
                "context_tokens",
            ]
        ]
        .drop_duplicates(
            subset=[
                "request_id",
            ]
        )
    )

    risk = risk.merge(
        request_meta,
        on="request_id",
        how="left",
        validate="many_to_one",
    )

    if risk["context_tokens"].isna().any():
        missing = (
            risk.loc[
                risk["context_tokens"].isna(),
                "request_id",
            ]
            .drop_duplicates()
        )

        print(
            "Missing context length for requests:"
        )

        print(
            missing.to_string(index=False)
        )

        raise RuntimeError(
            "Could not align OOF risk rows with "
            "oracle context lengths."
        )

    print(
        "CONTEXT LENGTH MERGE:",
        risk["request_id"].nunique(),
        "requests aligned"
    )

    risk = add_length_bucket(
        risk
    )

    results = []
    selections = []

    for delta in DELTAS:
        work = risk[
            np.isclose(
                risk["delta"],
                delta,
            )
        ].copy()

        for target in TARGETS:

            for fold in sorted(
                work["fold"].unique()
            ):
                train = work[
                    work["fold"] != fold
                ].copy()

                test = work[
                    work["fold"] == fold
                ].copy()

                # -------------------------------------
                # Static global baseline.
                # -------------------------------------
                global_map = choose_best_group_policy(
                    train.assign(
                        global_group="all"
                    ),
                    "global_group",
                    delta,
                    target,
                )

                # -------------------------------------
                # Task-aware lookup baseline.
                # -------------------------------------
                task_map = choose_best_group_policy(
                    train,
                    "task",
                    delta,
                    target,
                )

                # -------------------------------------
                # Length-aware lookup baseline.
                # -------------------------------------
                length_map = choose_best_group_policy(
                    train,
                    "length_bucket",
                    delta,
                    target,
                )

                # -------------------------------------
                # Nested lambda selection.
                #
                # lambda is chosen using TRAINING rows
                # only. The held-out fold is untouched.
                # -------------------------------------
                lambda_candidates = []

                for lam in LAMBDAS:
                    scores = conservative_score(
                        train[
                            "prob_safe_mean"
                        ],
                        train[
                            "risk_uncertainty"
                        ],
                        lam,
                    )

                    threshold = calibrate_threshold(
                        train["safe"],
                        scores,
                        target,
                    )

                    accepted = (
                        scores >= threshold
                    )

                    # Coverage acts as a cheap proxy for
                    # how often compression can be used.
                    coverage = accepted.mean()

                    lambda_candidates.append(
                        (
                            coverage,
                            -lam,
                            lam,
                            threshold,
                        )
                    )

                lambda_candidates.sort(
                    reverse=True
                )

                _, _, best_lambda, threshold = (
                    lambda_candidates[0]
                )

                # Risk-only threshold selected separately.
                risk_only_score = (
                    train[
                        "prob_safe_mean"
                    ].to_numpy(float)
                )

                risk_only_threshold = (
                    calibrate_threshold(
                        train["safe"],
                        risk_only_score,
                        target,
                    )
                )

                methods = {
                    "static_global": [],
                    "task_aware": [],
                    "length_aware": [],
                    "risk_only": [],
                    "risk_uncertainty": [],
                    "perfect_oracle": [],
                }

                for rid, g in test.groupby(
                    "request_id"
                ):
                    # Static.
                    methods[
                        "static_global"
                    ].append(
                        pick_request_policy(
                            g,
                            global_map["all"],
                        )
                    )

                    # Task.
                    task = g["task"].iloc[0]

                    methods[
                        "task_aware"
                    ].append(
                        pick_request_policy(
                            g,
                            task_map.get(
                                task,
                                "full",
                            ),
                        )
                    )

                    # Length.
                    lb = g[
                        "length_bucket"
                    ].iloc[0]

                    methods[
                        "length_aware"
                    ].append(
                        pick_request_policy(
                            g,
                            length_map.get(
                                lb,
                                "full",
                            ),
                        )
                    )

                    # Risk-only adaptive selection.
                    gg = g.copy()

                    accepted = gg[
                        gg[
                            "prob_safe_mean"
                        ]
                        >= risk_only_threshold
                    ]

                    if len(accepted):
                        chosen = (
                            accepted.sort_values(
                                [
                                    "memory_ratio",
                                    "prob_safe_mean",
                                ],
                                ascending=[
                                    True,
                                    False,
                                ],
                            )
                            .iloc[0]
                        )
                    else:
                        chosen = make_fullkv_row(
                            gg
                        )

                    methods[
                        "risk_only"
                    ].append(chosen)

                    # Risk + uncertainty.
                    gg[
                        "score"
                    ] = conservative_score(
                        gg[
                            "prob_safe_mean"
                        ],
                        gg[
                            "risk_uncertainty"
                        ],
                        best_lambda,
                    )

                    accepted = gg[
                        gg["score"]
                        >= threshold
                    ]

                    if len(accepted):
                        chosen = (
                            accepted.sort_values(
                                [
                                    "memory_ratio",
                                    "score",
                                ],
                                ascending=[
                                    True,
                                    False,
                                ],
                            )
                            .iloc[0]
                        )
                    else:
                        chosen = make_fullkv_row(
                            gg
                        )

                    methods[
                        "risk_uncertainty"
                    ].append(chosen)

                    # Perfect oracle.
                    safe = gg[
                        gg[
                            "degradation"
                        ]
                        <= delta + 1e-12
                    ].copy()

                    if len(safe):
                        chosen = (
                            safe.sort_values(
                                [
                                    "memory_ratio",
                                    "degradation",
                                ]
                            )
                            .iloc[0]
                        )
                    else:
                        # No compressed configuration is safe.
                        # The perfect oracle must therefore use FullKV.
                        chosen = make_fullkv_row(
                            gg
                        )

                    methods[
                        "perfect_oracle"
                    ].append(chosen)

                for method, rows in methods.items():
                    m = evaluate(
                        rows,
                        delta,
                    )

                    m.update(
                        {
                            "delta": delta,
                            "risk_target": target,
                            "fold": fold,
                            "method": method,
                            "selected_lambda": (
                                best_lambda
                                if method
                                == "risk_uncertainty"
                                else np.nan
                            ),
                        }
                    )

                    results.append(m)

                    for r in rows:
                        selections.append(
                            {
                                "delta": delta,
                                "risk_target": target,
                                "fold": fold,
                                "method": method,
                                "request_id": (
                                    r["request_id"]
                                ),
                                "task": r["task"],
                                "policy": r["policy"],
                                "degradation": (
                                    r["degradation"]
                                ),
                                "memory_ratio": (
                                    r["memory_ratio"]
                                ),
                            }
                        )

    result = pd.DataFrame(
        results
    )

    summary = (
        result.groupby(
            [
                "delta",
                "risk_target",
                "method",
            ],
            as_index=False,
        )
        .agg(
            mean_memory_saved_ratio=(
                "mean_memory_saved_ratio",
                "mean",
            ),
            violation_rate=(
                "violation_rate",
                "mean",
            ),
            catastrophic_rate=(
                "catastrophic_rate",
                "mean",
            ),
            mean_degradation=(
                "mean_degradation",
                "mean",
            ),
            fullkv_rate=(
                "fullkv_rate",
                "mean",
            ),
        )
    )

    result.to_csv(
        OUTDIR / "fold_results.csv",
        index=False,
    )

    summary.to_csv(
        OUTDIR
        / "adaptive_vs_heuristics_summary.csv",
        index=False,
    )

    pd.DataFrame(
        selections
    ).to_csv(
        OUTDIR / "selections.csv",
        index=False,
    )

    print(
        "\nSTATIC / HEURISTIC / ADAPTIVE COMPARISON"
    )

    print(
        summary.round(4)
        .to_string(index=False)
    )

    print(
        "\nEXPERIMENT 1: PASS"
    )


if __name__ == "__main__":
    main()
