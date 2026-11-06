from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


UNCERTAINTY_LAMBDAS = [
    0.0,
    0.25,
    0.50,
    0.75,
    1.00,
    1.50,
    2.00,
]


CLIFF_PROB_THRESHOLDS = [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
]


RISK_TARGETS = [
    0.01,
    0.05,
]


ALLOWED_HEADROOM = np.array(
    [
        0.00,
        0.25,
        0.50,
        0.75,
    ],
    dtype=float,
)


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
        "--request-oof",
        default=(
            "experiments/full_oracle_qwen3_8b/"
            "analysis/09_profile_prediction/"
            "request_profile_oof.csv"
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
            "analysis/10b_controller_sweep"
        ),
    )

    return p.parse_args()


def conservative_score(
    mean,
    uncertainty,
    lam,
):
    return np.clip(
        mean
        - lam * uncertainty,
        0.0,
        1.0,
    )


def conservative_headroom(
    mean,
    uncertainty,
    lam,
):
    return np.clip(
        mean
        - lam * uncertainty,
        0.0,
        0.75,
    )


def floor_headroom(value):
    feasible = ALLOWED_HEADROOM[
        ALLOWED_HEADROOM
        <= value + 1e-12
    ]

    if len(feasible) == 0:
        return 0.0

    return float(
        feasible.max()
    )


def choose_calibrated_threshold(
    y_true,
    score,
    max_violation,
):
    """
    Choose the lowest score threshold that maximizes
    accepted coverage while satisfying the requested
    empirical violation rate on calibration data.
    """

    thresholds = np.unique(
        np.concatenate(
            [
                score,
                np.array(
                    [
                        0.0,
                        1.000001,
                    ]
                ),
            ]
        )
    )

    thresholds = np.sort(
        thresholds
    )[::-1]

    best_threshold = 1.000001
    best_coverage = 0.0

    for threshold in thresholds:
        accepted = (
            score
            >= threshold
        )

        n_accept = int(
            accepted.sum()
        )

        if n_accept == 0:
            continue

        violation_rate = (
            (
                accepted
                & (y_true == 0)
            ).sum()
            / n_accept
        )

        coverage = (
            n_accept
            / len(y_true)
        )

        if (
            violation_rate
            <= max_violation
            + 1e-12
            and coverage
            > best_coverage
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

    out = Path(
        args.outdir
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    risk = pd.read_csv(
        args.risk_oof
    )

    request = pd.read_csv(
        args.request_oof
    )

    required_risk = {
        "request_id",
        "task",
        "policy",
        "memory_ratio",
        "degradation",
        "fold",
        "delta",
        "prob_safe_mean",
        "risk_uncertainty",
    }

    missing = (
        required_risk
        - set(
            risk.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"Missing risk columns: "
            f"{sorted(missing)}"
        )

    request_lookup = (
        request
        .set_index(
            "request_id"
        )
    )

    summary_rows = []
    selection_rows = []

    for delta in sorted(
        risk["delta"].unique()
    ):
        tag = (
            f"{delta:.2f}"
            .replace(
                ".",
                "p",
            )
        )

        work = risk[
            risk["delta"]
            == delta
        ].copy()

        for risk_target in (
            RISK_TARGETS
        ):
            for lam in (
                UNCERTAINTY_LAMBDAS
            ):

                # -------------------------------------
                # Fold-specific risk threshold
                # calibrated using the SAME lambda.
                # -------------------------------------
                risk_thresholds = {}

                for fold in sorted(
                    work["fold"].unique()
                ):
                    calib_path = (
                        Path(
                            args.calibration_dir
                        )
                        / (
                            f"calibration_delta_"
                            f"{delta:.2f}_"
                            f"fold_{int(fold)}.csv"
                        )
                    )

                    calib = pd.read_csv(
                        calib_path
                    )

                    calib_score = (
                        conservative_score(
                            calib[
                                "prob_safe_mean"
                            ].to_numpy(float),
                            calib[
                                "risk_uncertainty"
                            ].to_numpy(float),
                            lam,
                        )
                    )

                    risk_thresholds[
                        int(fold)
                    ] = (
                        choose_calibrated_threshold(
                            calib[
                                "safe"
                            ].to_numpy(int),
                            calib_score,
                            risk_target,
                        )
                    )

                for tau_p in (
                    CLIFF_PROB_THRESHOLDS
                ):
                    selections = []

                    for request_id, g in (
                        work.groupby(
                            "request_id",
                            sort=False,
                        )
                    ):
                        fold = int(
                            g[
                                "fold"
                            ].iloc[0]
                        )

                        req = (
                            request_lookup.loc[
                                request_id
                            ]
                        )

                        candidates = (
                            g.copy()
                        )

                        candidates[
                            "conservative_safe_score"
                        ] = (
                            conservative_score(
                                candidates[
                                    "prob_safe_mean"
                                ].to_numpy(float),
                                candidates[
                                    "risk_uncertainty"
                                ].to_numpy(float),
                                lam,
                            )
                        )

                        candidates = (
                            candidates[
                                candidates[
                                    "conservative_safe_score"
                                ]
                                >= risk_thresholds[
                                    fold
                                ]
                            ]
                            .copy()
                        )

                        # -----------------------------
                        # Headroom constraint using
                        # same uncertainty coefficient.
                        # -----------------------------
                        predicted_headroom = (
                            conservative_headroom(
                                float(
                                    req[
                                        f"headroom_pred_{tag}"
                                    ]
                                ),
                                float(
                                    req[
                                        f"headroom_std_{tag}"
                                    ]
                                ),
                                lam,
                            )
                        )

                        predicted_headroom = (
                            floor_headroom(
                                predicted_headroom
                            )
                        )

                        if len(
                            candidates
                        ):
                            candidates[
                                "memory_saved_ratio"
                            ] = (
                                1.0
                                - candidates[
                                    "memory_ratio"
                                ]
                            )

                            candidates = (
                                candidates[
                                    candidates[
                                        "memory_saved_ratio"
                                    ]
                                    <= predicted_headroom
                                    + 1e-12
                                ]
                                .copy()
                            )

                        # -----------------------------
                        # Compression-cliff guard.
                        # -----------------------------
                        cliff_triggered = (
                            float(
                                req[
                                    "cliff_probability"
                                ]
                            )
                            >= tau_p
                        )

                        if (
                            cliff_triggered
                            and len(
                                candidates
                            )
                        ):
                            # Conservative guard:
                            # forbid 75% memory removal.
                            candidates = (
                                candidates[
                                    candidates[
                                        "memory_ratio"
                                    ]
                                    >= 0.50
                                ]
                                .copy()
                            )

                        if len(
                            candidates
                        ) == 0:
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
                                    "cliff_guard_triggered": (
                                        cliff_triggered
                                    ),
                                    "selected_uncertainty": 0.0,
                                }
                            )

                            continue

                        chosen = (
                            candidates
                            .sort_values(
                                [
                                    "memory_ratio",
                                    "conservative_safe_score",
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
                                "cliff_guard_triggered": (
                                    cliff_triggered
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
                        > delta
                        + 1e-12
                    )

                    selected[
                        "catastrophic"
                    ] = (
                        selected[
                            "degradation"
                        ]
                        >= 0.25
                    )

                    summary_rows.append(
                        {
                            "delta": delta,
                            "risk_target": (
                                risk_target
                            ),
                            "uncertainty_lambda": (
                                lam
                            ),
                            "cliff_probability_threshold": (
                                tau_p
                            ),
                            "n_requests": (
                                len(
                                    selected
                                )
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
                            "cliff_guard_rate": (
                                selected[
                                    "cliff_guard_triggered"
                                ].mean()
                            ),
                            "mean_selected_uncertainty": (
                                selected[
                                    "selected_uncertainty"
                                ].mean()
                            ),
                        }
                    )

                    selected[
                        "delta"
                    ] = delta

                    selected[
                        "risk_target"
                    ] = risk_target

                    selected[
                        "uncertainty_lambda"
                    ] = lam

                    selected[
                        "cliff_probability_threshold"
                    ] = tau_p

                    selection_rows.append(
                        selected
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
        / "controller_hyperparameter_sweep.csv",
        index=False,
    )

    selections.to_csv(
        out
        / "controller_hyperparameter_selections.csv",
        index=False,
    )

    # =====================================================
    # Pareto-efficient settings
    # =====================================================

    pareto_rows = []

    for (
        delta,
        risk_target,
    ), g in summary.groupby(
        [
            "delta",
            "risk_target",
        ]
    ):
        for idx, row in (
            g.iterrows()
        ):
            dominated = (
                (
                    g[
                        "mean_memory_saved_ratio"
                    ]
                    >= row[
                        "mean_memory_saved_ratio"
                    ]
                )
                &
                (
                    g[
                        "violation_rate"
                    ]
                    <= row[
                        "violation_rate"
                    ]
                )
                &
                (
                    g[
                        "catastrophic_rate"
                    ]
                    <= row[
                        "catastrophic_rate"
                    ]
                )
                &
                (
                    (
                        g[
                            "mean_memory_saved_ratio"
                        ]
                        > row[
                            "mean_memory_saved_ratio"
                        ]
                    )
                    |
                    (
                        g[
                            "violation_rate"
                        ]
                        < row[
                            "violation_rate"
                        ]
                    )
                    |
                    (
                        g[
                            "catastrophic_rate"
                        ]
                        < row[
                            "catastrophic_rate"
                        ]
                    )
                )
            ).any()

            if not dominated:
                pareto_rows.append(
                    row.to_dict()
                )

    pareto = pd.DataFrame(
        pareto_rows
    )

    pareto.to_csv(
        out
        / "controller_pareto_frontier.csv",
        index=False,
    )

    print(
        "\nCONTROLLER SWEEP COMPLETE"
    )

    print(
        "\nTotal configurations:",
        len(
            summary
        ),
    )

    print(
        "\nPARETO-EFFICIENT CONFIGURATIONS"
    )

    display_cols = [
        "delta",
        "risk_target",
        "uncertainty_lambda",
        "cliff_probability_threshold",
        "mean_memory_saved_ratio",
        "violation_rate",
        "catastrophic_rate",
        "fallback_full_rate",
    ]

    print(
        pareto[
            display_cols
        ]
        .sort_values(
            [
                "delta",
                "risk_target",
                "violation_rate",
                "mean_memory_saved_ratio",
            ],
            ascending=[
                True,
                True,
                True,
                False,
            ],
        )
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        "\nCONTROLLER HYPERPARAMETER SWEEP: PASS"
    )


if __name__ == "__main__":
    main()
