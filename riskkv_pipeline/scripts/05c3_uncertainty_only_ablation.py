from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DELTAS = [0.01, 0.05, 0.10]

RISK_TARGETS = [0.01, 0.05]

UNCERTAINTY_LAMBDAS = [
    0.0,
    0.25,
    0.50,
    0.75,
    1.00,
    1.50,
    2.00,
]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--risk-oof",
        default=(
            "experiments/full_oracle_qwen3_8b/"
            "analysis/09_profile_prediction/"
            "policy_risk_oof.csv"
        ),
    )

    p.add_argument(
        "--calibration-dir",
        default=(
            "experiments/full_oracle_qwen3_8b/"
            "analysis/09_profile_prediction"
        ),
    )

    p.add_argument(
        "--outdir",
        default=(
            "experiments/full_oracle_qwen3_8b/"
            "analysis/10c_uncertainty_only"
        ),
    )

    return p.parse_args()


def conservative_score(mean, std, lam):
    return np.clip(
        mean - lam * std,
        0.0,
        1.0,
    )


def choose_threshold(
    y_true,
    score,
    target_violation,
):
    """
    Choose the score threshold giving the largest
    calibration-set coverage while satisfying the
    requested empirical violation target.
    """

    thresholds = np.unique(
        np.concatenate(
            [
                score,
                [0.0, 1.000001],
            ]
        )
    )

    thresholds = np.sort(
        thresholds
    )[::-1]

    best_threshold = 1.000001
    best_coverage = -1.0

    for threshold in thresholds:
        accepted = score >= threshold

        n_accept = int(
            accepted.sum()
        )

        if n_accept == 0:
            continue

        violations = int(
            (
                accepted
                & (y_true == 0)
            ).sum()
        )

        violation_rate = (
            violations
            / n_accept
        )

        coverage = (
            n_accept
            / len(y_true)
        )

        if (
            violation_rate
            <= target_violation + 1e-12
            and coverage > best_coverage
        ):
            best_threshold = float(
                threshold
            )

            best_coverage = float(
                coverage
            )

    return best_threshold


def main():
    args = parse_args()

    out = Path(args.outdir)

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    risk = pd.read_csv(
        args.risk_oof
    )

    required = {
        "request_id",
        "task",
        "policy",
        "family",
        "memory_ratio",
        "degradation",
        "delta",
        "fold",
        "safe",
        "prob_safe_mean",
        "risk_uncertainty",
    }

    missing = required - set(risk.columns)

    if missing:
        raise RuntimeError(
            f"Missing columns: {sorted(missing)}"
        )

    summary_rows = []
    selection_rows = []

    for delta in DELTAS:
        work = risk[
            np.isclose(
                risk["delta"],
                delta,
            )
        ].copy()

        if len(work) == 0:
            raise RuntimeError(
                f"No rows for delta={delta}"
            )

        for risk_target in RISK_TARGETS:

            for lam in UNCERTAINTY_LAMBDAS:

                # =====================================
                # CALIBRATE THRESHOLD PER OUTER FOLD
                # =====================================

                thresholds = {}

                for fold in sorted(
                    work["fold"].unique()
                ):
                    path = (
                        Path(
                            args.calibration_dir
                        )
                        / (
                            f"calibration_delta_"
                            f"{delta:.2f}_"
                            f"fold_{int(fold)}.csv"
                        )
                    )

                    if not path.exists():
                        raise FileNotFoundError(
                            path
                        )

                    calib = pd.read_csv(
                        path
                    )

                    score = conservative_score(
                        calib[
                            "prob_safe_mean"
                        ].to_numpy(float),
                        calib[
                            "risk_uncertainty"
                        ].to_numpy(float),
                        lam,
                    )

                    thresholds[
                        int(fold)
                    ] = choose_threshold(
                        calib[
                            "safe"
                        ].to_numpy(int),
                        score,
                        risk_target,
                    )

                # =====================================
                # SELECT POLICY REQUEST BY REQUEST
                # =====================================

                selections = []

                for request_id, g in work.groupby(
                    "request_id",
                    sort=False,
                ):
                    fold = int(
                        g["fold"].iloc[0]
                    )

                    candidates = g.copy()

                    candidates[
                        "controller_score"
                    ] = conservative_score(
                        candidates[
                            "prob_safe_mean"
                        ].to_numpy(float),
                        candidates[
                            "risk_uncertainty"
                        ].to_numpy(float),
                        lam,
                    )

                    accepted = candidates[
                        candidates[
                            "controller_score"
                        ]
                        >= thresholds[fold]
                    ].copy()

                    if len(accepted) == 0:

                        selections.append(
                            {
                                "request_id": request_id,
                                "task": g[
                                    "task"
                                ].iloc[0],
                                "policy": "full",
                                "memory_ratio": 1.0,
                                "degradation": 0.0,
                                "fallback_full": True,
                                "selected_score": 1.0,
                                "selected_uncertainty": 0.0,
                            }
                        )

                        continue

                    # Most aggressive accepted policy.
                    # For ties in memory ratio, choose the
                    # candidate with strongest safety score.
                    chosen = (
                        accepted
                        .sort_values(
                            [
                                "memory_ratio",
                                "controller_score",
                            ],
                            ascending=[
                                True,
                                False,
                            ],
                        )
                        .iloc[0]
                    )

                    selections.append(
                        {
                            "request_id": request_id,
                            "task": chosen[
                                "task"
                            ],
                            "policy": chosen[
                                "policy"
                            ],
                            "memory_ratio": float(
                                chosen[
                                    "memory_ratio"
                                ]
                            ),
                            "degradation": float(
                                chosen[
                                    "degradation"
                                ]
                            ),
                            "fallback_full": False,
                            "selected_score": float(
                                chosen[
                                    "controller_score"
                                ]
                            ),
                            "selected_uncertainty": float(
                                chosen[
                                    "risk_uncertainty"
                                ]
                            ),
                        }
                    )

                selected = pd.DataFrame(
                    selections
                )

                selected[
                    "memory_saved_ratio"
                ] = (
                    1.0
                    - selected[
                        "memory_ratio"
                    ]
                )

                selected[
                    "violation"
                ] = (
                    selected[
                        "degradation"
                    ]
                    > delta + 1e-12
                )

                selected[
                    "catastrophic"
                ] = (
                    selected[
                        "degradation"
                    ]
                    >= 0.25
                )

                selected["delta"] = delta
                selected[
                    "risk_target"
                ] = risk_target
                selected[
                    "uncertainty_lambda"
                ] = lam

                selection_rows.append(
                    selected
                )

                summary_rows.append(
                    {
                        "delta": delta,
                        "risk_target": risk_target,
                        "uncertainty_lambda": lam,
                        "controller": (
                            "risk_mean"
                            if lam == 0
                            else "risk_plus_uncertainty"
                        ),
                        "n_requests": len(
                            selected
                        ),
                        "mean_memory_saved_ratio": (
                            selected[
                                "memory_saved_ratio"
                            ].mean()
                        ),
                        "violation_rate": (
                            selected[
                                "violation"
                            ].mean()
                        ),
                        "catastrophic_rate": (
                            selected[
                                "catastrophic"
                            ].mean()
                        ),
                        "mean_degradation": (
                            selected[
                                "degradation"
                            ].mean()
                        ),
                        "max_degradation": (
                            selected[
                                "degradation"
                            ].max()
                        ),
                        "fallback_full_rate": (
                            selected[
                                "fallback_full"
                            ].mean()
                        ),
                        "mean_selected_uncertainty": (
                            selected[
                                "selected_uncertainty"
                            ].mean()
                        ),
                    }
                )

    summary = pd.DataFrame(
        summary_rows
    )

    selections = pd.concat(
        selection_rows,
        ignore_index=True,
    )

    summary.to_csv(
        out
        / "uncertainty_ablation_summary.csv",
        index=False,
    )

    selections.to_csv(
        out
        / "uncertainty_ablation_selections.csv",
        index=False,
    )

    # =========================================
    # BEST FEASIBLE λ FOR EACH OPERATING POINT
    # =========================================

    best_rows = []

    for (
        delta,
        target,
    ), g in summary.groupby(
        [
            "delta",
            "risk_target",
        ]
    ):

        feasible = g[
            g["violation_rate"]
            <= target + 1e-12
        ].copy()

        if len(feasible) == 0:
            continue

        best = (
            feasible
            .sort_values(
                [
                    "mean_memory_saved_ratio",
                    "catastrophic_rate",
                    "fallback_full_rate",
                ],
                ascending=[
                    False,
                    True,
                    True,
                ],
            )
            .iloc[0]
        )

        best_rows.append(
            best.to_dict()
        )

    best = pd.DataFrame(
        best_rows
    )

    best.to_csv(
        out
        / "best_feasible_uncertainty_settings.csv",
        index=False,
    )

    print(
        "\nUNCERTAINTY-ONLY ABLATION"
    )

    print(
        summary
        .round(4)
        .to_string(index=False)
    )

    print(
        "\nBEST FEASIBLE SETTINGS"
    )

    if len(best):
        print(
            best
            .round(4)
            .to_string(index=False)
        )

    print(
        "\nUNCERTAINTY-ONLY ABLATION: PASS"
    )


if __name__ == "__main__":
    main()
