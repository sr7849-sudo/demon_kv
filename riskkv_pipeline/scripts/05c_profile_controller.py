from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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
    ]
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
            "analysis/10_profile_controller"
        ),
    )

    p.add_argument(
        "--cliff-threshold",
        type=float,
        default=0.50,
    )

    return p.parse_args()


def choose_threshold(
    y_true,
    score,
    max_violation,
):
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

    best = None

    for threshold in thresholds:
        accepted = (
            score >= threshold
        )

        n = int(
            accepted.sum()
        )

        if n == 0:
            continue

        violations = (
            accepted
            & (y_true == 0)
        ).sum()

        violation_rate = (
            violations
            / n
        )

        coverage = (
            n
            / len(y_true)
        )

        if (
            violation_rate
            <= max_violation + 1e-12
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
            }

            if (
                best is None
                or candidate[
                    "coverage"
                ]
                > best[
                    "coverage"
                ]
            ):
                best = candidate

    if best is None:
        return 1.000001

    return best["threshold"]


def floor_headroom(value):
    allowed = ALLOWED_HEADROOM[
        ALLOWED_HEADROOM
        <= value + 1e-12
    ]

    if len(allowed) == 0:
        return 0.0

    return float(
        allowed.max()
    )


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

    request = pd.read_csv(
        args.request_oof
    )

    request_lookup = (
        request.set_index(
            "request_id"
        )
    )

    summary_rows = []
    all_selection_rows = []

    controllers = [
        "risk_mean",
        "risk_uncertainty",
        "profile",
        "profile_cliff_guard",
    ]

    for delta in sorted(
        risk["delta"].unique()
    ):
        tag = (
            f"{delta:.2f}"
            .replace(".", "p")
        )

        work = risk[
            risk["delta"] == delta
        ].copy()

        for risk_target in RISK_TARGETS:
            # -----------------------------------------
            # Calibrate thresholds fold-by-fold using
            # calibration data never used as outer test.
            # -----------------------------------------
            thresholds_mean = {}
            thresholds_lcb = {}

            for fold in sorted(
                work["fold"].unique()
            ):
                path = Path(
                    args.calibration_dir
                ) / (
                    f"calibration_delta_"
                    f"{delta:.2f}_"
                    f"fold_{int(fold)}.csv"
                )

                calib = pd.read_csv(
                    path
                )

                y = calib[
                    "safe"
                ].to_numpy(int)

                thresholds_mean[
                    fold
                ] = choose_threshold(
                    y,
                    calib[
                        "prob_safe_mean"
                    ].to_numpy(float),
                    risk_target,
                )

                thresholds_lcb[
                    fold
                ] = choose_threshold(
                    y,
                    calib[
                        "prob_safe_lcb"
                    ].to_numpy(float),
                    risk_target,
                )

            for controller in controllers:
                selections = []

                for request_id, g in work.groupby(
                    "request_id",
                    sort=False,
                ):
                    fold = int(
                        g["fold"].iloc[0]
                    )

                    req = request_lookup.loc[
                        request_id
                    ]

                    candidates = g.copy()

                    if controller == "risk_mean":
                        score_col = (
                            "prob_safe_mean"
                        )

                        threshold = (
                            thresholds_mean[
                                fold
                            ]
                        )

                    else:
                        score_col = (
                            "prob_safe_lcb"
                        )

                        threshold = (
                            thresholds_lcb[
                                fold
                            ]
                        )

                    candidates = candidates[
                        candidates[
                            score_col
                        ]
                        >= threshold
                    ].copy()

                    # ---------------------------------
                    # Headroom guard
                    # ---------------------------------
                    if controller in {
                        "profile",
                        "profile_cliff_guard",
                    }:
                        predicted_headroom = (
                            req[
                                f"headroom_lcb_{tag}"
                            ]
                        )

                        predicted_headroom = (
                            floor_headroom(
                                predicted_headroom
                            )
                        )

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
                        )

                    # ---------------------------------
                    # Fragility / cliff guard
                    #
                    # If the request is likely to
                    # contain a quality cliff, do not
                    # allow the most aggressive 75%
                    # removal profile.
                    # ---------------------------------
                    if (
                        controller
                        == "profile_cliff_guard"
                        and req[
                            "cliff_probability"
                        ]
                        >= args.cliff_threshold
                    ):
                        candidates = candidates[
                            candidates[
                                "memory_ratio"
                            ]
                            >= 0.50
                        ]

                    if len(candidates) == 0:
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
                                "prob_safe": 1.0,
                                "risk_uncertainty": 0.0,
                            }
                        )

                        continue

                    # Lowest memory first.
                    # For equal memory ratio, prefer
                    # the policy with strongest
                    # conservative safety probability.
                    chosen = (
                        candidates
                        .sort_values(
                            [
                                "memory_ratio",
                                score_col,
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
                            "memory_ratio": chosen[
                                "memory_ratio"
                            ],
                            "degradation": chosen[
                                "degradation"
                            ],
                            "fallback_full": False,
                            "prob_safe": chosen[
                                "prob_safe_mean"
                            ],
                            "risk_uncertainty": chosen[
                                "risk_uncertainty"
                            ],
                        }
                    )

                selected = pd.DataFrame(
                    selections
                )

                selected[
                    "delta"
                ] = delta

                selected[
                    "risk_target"
                ] = risk_target

                selected[
                    "controller"
                ] = controller

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

                oracle_headroom = (
                    request_lookup[
                        f"headroom_true_{tag}"
                    ]
                    .reindex(
                        selected[
                            "request_id"
                        ]
                    )
                    .to_numpy()
                )

                selected[
                    "oracle_headroom"
                ] = (
                    oracle_headroom
                )

                all_selection_rows.append(
                    selected
                )

                mean_oracle = (
                    selected[
                        "oracle_headroom"
                    ].mean()
                )

                mean_saved = (
                    selected[
                        "memory_saved_ratio"
                    ].mean()
                )

                summary_rows.append(
                    {
                        "delta": delta,
                        "risk_target": risk_target,
                        "controller": controller,
                        "n_requests": len(
                            selected
                        ),
                        "mean_memory_saved_ratio": (
                            mean_saved
                        ),
                        "mean_oracle_headroom": (
                            mean_oracle
                        ),
                        "oracle_recovery": (
                            mean_saved
                            / mean_oracle
                            if mean_oracle > 0
                            else np.nan
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
                                "risk_uncertainty"
                            ].mean()
                        ),
                    }
                )

    summary = pd.DataFrame(
        summary_rows
    )

    selections = pd.concat(
        all_selection_rows,
        ignore_index=True,
    )

    summary.to_csv(
        out
        / "profile_controller_summary.csv",
        index=False,
    )

    selections.to_csv(
        out
        / "profile_controller_selections.csv",
        index=False,
    )

    print("\nREQUEST COMPRESSION PROFILE CONTROLLER")
    print(
        summary
        .round(4)
        .to_string(
            index=False
        )
    )

    print("\nPROFILE CONTROLLER: PASS")


if __name__ == "__main__":
    main()
