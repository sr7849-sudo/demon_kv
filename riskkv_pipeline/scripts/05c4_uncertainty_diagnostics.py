from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


INFILE = (
    "experiments/full_oracle_qwen3_8b/"
    "analysis/09_profile_prediction/"
    "policy_risk_oof.csv"
)

OUTDIR = Path(
    "experiments/full_oracle_qwen3_8b/"
    "analysis/10d_uncertainty_diagnostics"
)


def main():
    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        INFILE
    )

    rows = []
    quantile_rows = []

    for delta, g in df.groupby(
        "delta"
    ):
        x = g.copy()

        x[
            "pred_safe_05"
        ] = (
            x[
                "prob_safe_mean"
            ]
            >= 0.5
        ).astype(int)

        x[
            "prediction_error"
        ] = (
            x[
                "pred_safe_05"
            ]
            != x[
                "safe"
            ]
        ).astype(int)

        # Dangerous error:
        # model predicts safe, actual is unsafe.
        x[
            "false_safe"
        ] = (
            (
                x[
                    "pred_safe_05"
                ] == 1
            )
            & (
                x["safe"] == 0
            )
        ).astype(int)

        if (
            x[
                "prediction_error"
            ].nunique()
            == 2
        ):
            error_auroc = (
                roc_auc_score(
                    x[
                        "prediction_error"
                    ],
                    x[
                        "risk_uncertainty"
                    ],
                )
            )
        else:
            error_auroc = np.nan

        if (
            x[
                "false_safe"
            ].nunique()
            == 2
        ):
            false_safe_auroc = (
                roc_auc_score(
                    x[
                        "false_safe"
                    ],
                    x[
                        "risk_uncertainty"
                    ],
                )
            )
        else:
            false_safe_auroc = (
                np.nan
            )

        rows.append(
            {
                "delta": delta,
                "n_rows": len(x),
                "error_rate": (
                    x[
                        "prediction_error"
                    ].mean()
                ),
                "false_safe_rate": (
                    x[
                        "false_safe"
                    ].mean()
                ),
                "mean_uncertainty_correct": (
                    x.loc[
                        x[
                            "prediction_error"
                        ] == 0,
                        "risk_uncertainty",
                    ].mean()
                ),
                "mean_uncertainty_error": (
                    x.loc[
                        x[
                            "prediction_error"
                        ] == 1,
                        "risk_uncertainty",
                    ].mean()
                ),
                "uncertainty_error_auroc": (
                    error_auroc
                ),
                "uncertainty_false_safe_auroc": (
                    false_safe_auroc
                ),
            }
        )

        x[
            "uncertainty_decile"
        ] = pd.qcut(
            x[
                "risk_uncertainty"
            ],
            10,
            labels=False,
            duplicates="drop",
        )

        q = (
            x.groupby(
                "uncertainty_decile",
                as_index=False,
            )
            .agg(
                n=(
                    "request_id",
                    "size",
                ),
                mean_uncertainty=(
                    "risk_uncertainty",
                    "mean",
                ),
                prediction_error_rate=(
                    "prediction_error",
                    "mean",
                ),
                false_safe_rate=(
                    "false_safe",
                    "mean",
                ),
                mean_degradation=(
                    "degradation",
                    "mean",
                ),
            )
        )

        q[
            "delta"
        ] = delta

        quantile_rows.append(
            q
        )

    summary = pd.DataFrame(
        rows
    )

    quantiles = pd.concat(
        quantile_rows,
        ignore_index=True,
    )

    summary.to_csv(
        OUTDIR
        / "uncertainty_diagnostic_summary.csv",
        index=False,
    )

    quantiles.to_csv(
        OUTDIR
        / "uncertainty_deciles.csv",
        index=False,
    )

    print(
        "\nUNCERTAINTY DIAGNOSTICS"
    )

    print(
        summary
        .round(4)
        .to_string(index=False)
    )

    print(
        "\nUNCERTAINTY DECILES"
    )

    print(
        quantiles
        .round(4)
        .to_string(index=False)
    )

    print(
        "\nUNCERTAINTY DIAGNOSTICS: PASS"
    )


if __name__ == "__main__":
    main()
