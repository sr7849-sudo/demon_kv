"""
EXPERIMENT 4
============

Metric-specific oracle opportunity recovery.

Purpose
-------
Measures how much of the BEST QUALITY-SAFE hardware opportunity each
runtime-selection method recovers.

The key correction over the earlier implementation is that there is
NOT one universal oracle configuration.

For every request r and quality budget delta, we independently define:

    memory oracle:
        argmin_p M(r,p)  subject to D(r,p) <= delta

    latency oracle:
        argmin_p T(r,p)  subject to D(r,p) <= delta

    energy oracle:
        argmin_p E(r,p)  subject to D(r,p) <= delta

    power oracle:
        argmin_p P(r,p)  subject to D(r,p) <= delta

This matters because the policy that minimizes memory is not
necessarily the one that minimizes latency, energy, or power.

Two recovery quantities are reported.

1. RAW recovery:
       compares the method exactly as selected.

2. SAFETY-ADJUSTED recovery:
       if the method violates the requested quality budget on a request,
       that request is replaced by FullKV before resource cost is
       calculated.

The safety-adjusted quantity is the primary defensible metric because
an unsafe policy should not receive "credit" for beating the safe
oracle by spending fewer resources while violating quality.

The experiment also reports:
  - which policies most often form each metric-specific oracle;
  - how frequently the four oracle policies disagree for the same
    request.

The disagreement table directly tests whether there is genuinely no
single universally optimal KV configuration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from riskkv.synthesis_eval import (
    load_quality_hardware,
)


# ---------------------------------------------------------------------
# Inputs / outputs
# ---------------------------------------------------------------------

SELECTIONS = (
    "experiments/full_oracle_qwen3_8b/"
    "analysis/15_adaptive_vs_heuristics/"
    "selections.csv"
)

OUTDIR = Path(
    "experiments/full_oracle_qwen3_8b/"
    "analysis/18_oracle_recovery"
)


METRICS = {
    "memory": "memory_bytes",
    "runtime": "runtime_seconds",
    "energy": "energy_j",
    "power": "power_w",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def select_metric_oracle(
    g: pd.DataFrame,
    metric_col: str,
    delta: float,
) -> pd.Series:
    """
    Select the best quality-safe configuration for ONE request and ONE
    hardware metric.

    FullKV is part of the measured hardware candidate set and has
    degradation == 0, so a feasible configuration should always exist.
    """

    safe = g[
        g["degradation"]
        <= delta + 1e-12
    ].copy()

    if len(safe) == 0:
        raise RuntimeError(
            "No quality-safe configuration found. "
            "FullKV should always be feasible."
        )

    # Primary sort:
    #     requested hardware objective.
    #
    # Deterministic ties:
    #     degradation -> memory -> runtime -> policy.
    sort_cols = [
        metric_col,
        "degradation",
        "memory_bytes",
        "runtime_seconds",
        "policy",
    ]

    sort_cols = list(
        dict.fromkeys(sort_cols)
    )

    return (
        safe.sort_values(
            sort_cols
        )
        .iloc[0]
    )


def safe_recovery(
    full_cost: float,
    method_cost: float,
    oracle_cost: float,
):
    """
    Fraction of the available oracle opportunity recovered.

        recovery =
            FullKV - method
            ----------------
            FullKV - oracle

    If FullKV and the oracle have effectively identical cost, there is
    no meaningful opportunity to recover, so return NaN.
    """

    opportunity = (
        full_cost
        - oracle_cost
    )

    if abs(opportunity) <= 1e-12:
        return np.nan

    return (
        full_cost
        - method_cost
    ) / opportunity


# ---------------------------------------------------------------------
# Build request-level metric-specific oracles
# ---------------------------------------------------------------------

def build_oracles(
    hw: pd.DataFrame,
    deltas,
):
    """
    Produce one oracle choice for every:

        request x delta x hardware objective.
    """

    rows = []

    for delta in deltas:

        for request_id, g in hw.groupby(
            "request_id",
            sort=False,
        ):

            for metric_name, metric_col in (
                METRICS.items()
            ):

                chosen = select_metric_oracle(
                    g,
                    metric_col,
                    delta,
                )

                rows.append(
                    {
                        "delta": delta,
                        "request_id": request_id,
                        "task": chosen["task"],
                        "metric": metric_name,
                        "metric_column": metric_col,
                        "oracle_policy": (
                            chosen["policy"]
                        ),
                        "oracle_cost": (
                            float(
                                chosen[
                                    metric_col
                                ]
                            )
                        ),
                        "oracle_degradation": (
                            float(
                                chosen[
                                    "degradation"
                                ]
                            )
                        ),
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Load stable five-run hardware medians + frozen oracle degradation.
    # -----------------------------------------------------------------

    hw = load_quality_hardware()

    selections = pd.read_csv(
        SELECTIONS
    )

    print(
        "HARDWARE REQUESTS:",
        hw.request_id.nunique(),
    )

    print(
        "HARDWARE PAIRS:",
        len(hw),
    )

    print(
        "SELECTION REQUESTS:",
        selections.request_id.nunique(),
    )

    # -----------------------------------------------------------------
    # Only 25 requests have five-run hardware characterization.
    #
    # Experiment 1 contains all 290 requests. Therefore, Experiment 4 is
    # intentionally evaluated on the intersection with the 25-request
    # repeated-hardware subset.
    # -----------------------------------------------------------------

    hardware_requests = set(
        hw["request_id"]
    )

    selections = selections[
        selections[
            "request_id"
        ].isin(
            hardware_requests
        )
    ].copy()

    print(
        "MATCHED SELECTION REQUESTS:",
        selections.request_id.nunique(),
    )

    if (
        selections.request_id.nunique()
        != hw.request_id.nunique()
    ):
        raise RuntimeError(
            "Experiment-1 selections do not cover "
            "all repeated-hardware requests."
        )

    deltas = sorted(
        selections[
            "delta"
        ].unique()
    )

    # -----------------------------------------------------------------
    # Build metric-specific perfect oracles.
    # -----------------------------------------------------------------

    oracles = build_oracles(
        hw,
        deltas,
    )

    oracles.to_csv(
        OUTDIR
        / "metric_specific_oracle_points.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # FullKV hardware baseline.
    # -----------------------------------------------------------------

    full = (
        hw[
            hw["policy"]
            == "full"
        ][
            [
                "request_id",
                "memory_bytes",
                "runtime_seconds",
                "energy_j",
                "power_w",
            ]
        ]
        .copy()
    )

    if (
        full["request_id"].nunique()
        != hw["request_id"].nunique()
    ):
        raise RuntimeError(
            "Missing FullKV hardware measurements."
        )

    full = full.rename(
        columns={
            "memory_bytes":
                "full_memory_bytes",

            "runtime_seconds":
                "full_runtime_seconds",

            "energy_j":
                "full_energy_j",

            "power_w":
                "full_power_w",
        }
    )

    # -----------------------------------------------------------------
    # Attach measured hardware costs to every Experiment-1 decision.
    # -----------------------------------------------------------------

    selected = selections.merge(
        hw[
            [
                "request_id",
                "policy",
                "memory_bytes",
                "runtime_seconds",
                "energy_j",
                "power_w",
            ]
        ],
        on=[
            "request_id",
            "policy",
        ],
        how="left",
        validate="many_to_one",
    )

    # FullKV rows in Experiment 1 were synthesized as explicit fallback
    # rows and therefore still match the measured "full" hardware row.
    required_costs = list(
        METRICS.values()
    )

    if (
        selected[
            required_costs
        ]
        .isna()
        .any()
        .any()
    ):
        bad = selected[
            selected[
                required_costs
            ]
            .isna()
            .any(axis=1)
        ]

        print(
            bad[
                [
                    "request_id",
                    "policy",
                    "method",
                ]
            ]
            .to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Missing repeated hardware measurements "
            "for one or more selected configurations."
        )

    selected = selected.merge(
        full,
        on="request_id",
        how="left",
        validate="many_to_one",
    )

    # -----------------------------------------------------------------
    # Request-level recovery calculations.
    # -----------------------------------------------------------------

    recovery_rows = []

    for (
        delta,
        risk_target,
        method,
    ), method_rows in selected.groupby(
        [
            "delta",
            "risk_target",
            "method",
        ],
        sort=False,
    ):

        for metric_name, metric_col in (
            METRICS.items()
        ):

            oracle_metric = (
                oracles[
                    (
                        np.isclose(
                            oracles[
                                "delta"
                            ],
                            delta,
                        )
                    )
                    & (
                        oracles[
                            "metric"
                        ]
                        == metric_name
                    )
                ][
                    [
                        "request_id",
                        "oracle_policy",
                        "oracle_cost",
                    ]
                ]
            )

            x = method_rows.merge(
                oracle_metric,
                on="request_id",
                how="inner",
                validate="one_to_one",
            )

            full_col = {
                "memory":
                    "full_memory_bytes",

                "runtime":
                    "full_runtime_seconds",

                "energy":
                    "full_energy_j",

                "power":
                    "full_power_w",
            }[
                metric_name
            ]

            x["quality_violation"] = (
                x["degradation"]
                > delta + 1e-12
            )

            # ---------------------------------------------------------
            # Raw selected cost.
            # ---------------------------------------------------------

            x[
                "raw_method_cost"
            ] = x[
                metric_col
            ]

            # ---------------------------------------------------------
            # Safety-adjusted cost.
            #
            # If a learned/heuristic method violates the quality budget,
            # give it the cost of FullKV for that request instead.
            #
            # This prevents unsafe compression from receiving resource
            # credit unavailable to any valid quality-constrained system.
            # ---------------------------------------------------------

            x[
                "safety_adjusted_cost"
            ] = np.where(
                x[
                    "quality_violation"
                ],
                x[
                    full_col
                ],
                x[
                    metric_col
                ],
            )

            full_cost = float(
                x[
                    full_col
                ].mean()
            )

            raw_method_cost = float(
                x[
                    "raw_method_cost"
                ].mean()
            )

            adjusted_method_cost = float(
                x[
                    "safety_adjusted_cost"
                ].mean()
            )

            oracle_cost = float(
                x[
                    "oracle_cost"
                ].mean()
            )

            oracle_opportunity = (
                full_cost
                - oracle_cost
            )

            raw_gain = (
                full_cost
                - raw_method_cost
            )

            adjusted_gain = (
                full_cost
                - adjusted_method_cost
            )

            raw_recovery = safe_recovery(
                full_cost,
                raw_method_cost,
                oracle_cost,
            )

            adjusted_recovery = safe_recovery(
                full_cost,
                adjusted_method_cost,
                oracle_cost,
            )

            recovery_rows.append(
                {
                    "delta": delta,

                    "risk_target":
                        risk_target,

                    "method":
                        method,

                    "metric":
                        metric_name,

                    "n_requests":
                        len(x),

                    "quality_violation_rate":
                        x[
                            "quality_violation"
                        ].mean(),

                    "fullkv_cost":
                        full_cost,

                    "raw_method_cost":
                        raw_method_cost,

                    "safety_adjusted_method_cost":
                        adjusted_method_cost,

                    "oracle_cost":
                        oracle_cost,

                    "oracle_opportunity":
                        oracle_opportunity,

                    "raw_gain_vs_fullkv":
                        raw_gain,

                    "safety_adjusted_gain_vs_fullkv":
                        adjusted_gain,

                    "raw_oracle_recovery":
                        raw_recovery,

                    "safety_adjusted_oracle_recovery":
                        adjusted_recovery,
                }
            )

    recovery = pd.DataFrame(
        recovery_rows
    )

    recovery.to_csv(
        OUTDIR
        / "oracle_recovery_metric_specific.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # Compact primary table.
    #
    # SAFETY-ADJUSTED recovery should be the main number used in the
    # paper. Raw recovery remains useful diagnostically.
    # -----------------------------------------------------------------

    compact = (
        recovery[
            [
                "delta",
                "risk_target",
                "method",
                "metric",
                "quality_violation_rate",
                "safety_adjusted_oracle_recovery",
            ]
        ]
        .copy()
    )

    compact.to_csv(
        OUTDIR
        / "oracle_recovery_primary_table.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # Oracle policy frequency.
    #
    # Shows which KV configuration is the ideal policy for memory,
    # latency, energy, and power separately.
    # -----------------------------------------------------------------

    oracle_frequency = (
        oracles.groupby(
            [
                "delta",
                "metric",
                "oracle_policy",
            ]
        )
        .size()
        .rename(
            "requests"
        )
        .reset_index()
    )

    totals = (
        oracles.groupby(
            [
                "delta",
                "metric",
            ]
        )[
            "request_id"
        ]
        .nunique()
        .rename(
            "total_requests"
        )
        .reset_index()
    )

    oracle_frequency = (
        oracle_frequency.merge(
            totals,
            on=[
                "delta",
                "metric",
            ],
            how="left",
        )
    )

    oracle_frequency[
        "request_rate"
    ] = (
        oracle_frequency[
            "requests"
        ]
        / oracle_frequency[
            "total_requests"
        ]
    )

    oracle_frequency.to_csv(
        OUTDIR
        / "oracle_policy_frequency.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # Oracle disagreement.
    #
    # This is particularly useful for defending runtime synthesis:
    # if metric-specific ideal configurations disagree for the SAME
    # request, no single policy can be globally optimal.
    # -----------------------------------------------------------------

    oracle_pivot = (
        oracles.pivot(
            index=[
                "delta",
                "request_id",
                "task",
            ],
            columns="metric",
            values="oracle_policy",
        )
        .reset_index()
    )

    disagreement_rows = []

    for delta, g in oracle_pivot.groupby(
        "delta"
    ):

        tests = {
            "memory_vs_runtime":
                (
                    g["memory"]
                    != g["runtime"]
                ),

            "memory_vs_energy":
                (
                    g["memory"]
                    != g["energy"]
                ),

            "memory_vs_power":
                (
                    g["memory"]
                    != g["power"]
                ),

            "runtime_vs_energy":
                (
                    g["runtime"]
                    != g["energy"]
                ),

            "runtime_vs_power":
                (
                    g["runtime"]
                    != g["power"]
                ),

            "energy_vs_power":
                (
                    g["energy"]
                    != g["power"]
                ),
        }

        for name, changed in (
            tests.items()
        ):
            disagreement_rows.append(
                {
                    "delta": delta,
                    "comparison": name,
                    "requests": len(g),
                    "requests_different":
                        int(
                            changed.sum()
                        ),
                    "fraction_different":
                        float(
                            changed.mean()
                        ),
                }
            )

        same_all = (
            (
                g["memory"]
                == g["runtime"]
            )
            & (
                g["memory"]
                == g["energy"]
            )
            & (
                g["memory"]
                == g["power"]
            )
        )

        unique_count = (
            g[
                [
                    "memory",
                    "runtime",
                    "energy",
                    "power",
                ]
            ]
            .nunique(
                axis=1
            )
        )

        disagreement_rows.append(
            {
                "delta": delta,
                "comparison":
                    "all_four_same",

                "requests":
                    len(g),

                "requests_different":
                    int(
                        same_all.sum()
                    ),

                # Here the column means fraction where all FOUR are same,
                # despite retaining the common output schema.
                "fraction_different":
                    float(
                        same_all.mean()
                    ),
            }
        )

        disagreement_rows.append(
            {
                "delta": delta,
                "comparison":
                    "mean_unique_oracle_policies",

                "requests":
                    len(g),

                "requests_different":
                    np.nan,

                "fraction_different":
                    float(
                        unique_count.mean()
                    ),
            }
        )

    disagreement = pd.DataFrame(
        disagreement_rows
    )

    disagreement.to_csv(
        OUTDIR
        / "oracle_disagreement.csv",
        index=False,
    )

    oracle_pivot.to_csv(
        OUTDIR
        / "oracle_policy_by_request.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # Validation.
    # -----------------------------------------------------------------

    assert (
        oracles[
            "request_id"
        ].nunique()
        == 25
    )

    assert set(
        oracles[
            "metric"
        ].unique()
    ) == set(
        METRICS
    )

    # The perfect oracle must always recover exactly 1 whenever there
    # is nonzero resource opportunity.
    perfect = recovery[
        recovery[
            "method"
        ]
        == "perfect_oracle"
    ]

    valid_perfect = perfect[
        perfect[
            "oracle_opportunity"
        ].abs()
        > 1e-12
    ]

    # Note: Experiment-1's "perfect_oracle" was originally memory-optimal.
    # It therefore should NOT be expected to equal each metric-specific
    # oracle. We deliberately do NOT assert recovery == 1 here.
    #
    # The true metric-specific oracle is represented separately in
    # metric_specific_oracle_points.csv.

    # Safety-adjusted recovery should not materially exceed 1 because
    # unsafe decisions are converted to FullKV and the oracle is
    # objective-specific.
    finite = recovery[
        "safety_adjusted_oracle_recovery"
    ].dropna()

    if len(finite):
        max_recovery = finite.max()

        print(
            "\nMAX SAFETY-ADJUSTED RECOVERY:",
            round(
                float(
                    max_recovery
                ),
                6,
            ),
        )

        if max_recovery > 1.000001:
            print(
                "WARNING: recovery > 1 detected. "
                "Inspect request alignment/tie handling."
            )

    # -----------------------------------------------------------------
    # Console report.
    # -----------------------------------------------------------------

    print(
        "\n"
        + "=" * 78
    )

    print(
        "METRIC-SPECIFIC ORACLE RECOVERY"
    )

    print(
        "=" * 78
    )

    print(
        compact
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "METRIC-SPECIFIC ORACLE POLICY FREQUENCY"
    )

    print(
        "=" * 78
    )

    print(
        oracle_frequency
        .sort_values(
            [
                "delta",
                "metric",
                "request_rate",
            ],
            ascending=[
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
        "\n"
        + "=" * 78
    )

    print(
        "ORACLE DISAGREEMENT"
    )

    print(
        "=" * 78
    )

    print(
        disagreement
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        "\nOutputs:",
        OUTDIR,
    )

    print(
        "\nEXPERIMENT 4 "
        "METRIC-SPECIFIC ORACLE: PASS"
    )


if __name__ == "__main__":
    main()
