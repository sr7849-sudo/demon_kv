#!/usr/bin/env python3

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


POLICY_ORDER = [
    "full",
    "snapkv_25",
    "snapkv_50",
    "snapkv_75",
    "knorm_25",
    "knorm_50",
    "knorm_75",
    "quant4",
]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def find_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            "Could not find any of these columns:\n  "
            + "\n  ".join(candidates)
            + "\n\nAvailable columns:\n  "
            + "\n  ".join(df.columns)
        )
    return None


def pareto_mask(values):
    """
    Return True for non-dominated rows.

    All objectives are minimized.

    Point j dominates i iff:
        j <= i on every objective
    and
        j < i on at least one objective.
    """
    x = np.asarray(values, dtype=float)

    if np.isnan(x).any():
        raise ValueError("NaN encountered in Pareto objective matrix.")

    n = len(x)
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        dominated = np.any(
            np.all(x <= x[i], axis=1)
            & np.any(x < x[i], axis=1)
        )
        if dominated:
            keep[i] = False

    return keep


def signature(policies):
    ordered = [p for p in POLICY_ORDER if p in set(policies)]
    extras = sorted(set(policies) - set(POLICY_ORDER))
    return "|".join(ordered + extras)


def safe_cv(std, median):
    if abs(median) < 1e-12:
        return np.nan
    return std / abs(median)


# ---------------------------------------------------------------------
# Load + normalize raw hardware measurements
# ---------------------------------------------------------------------

def load_raw(path):
    df = pd.read_csv(path)

    request_col = find_col(df, ["request_id", "request"])
    task_col = find_col(df, ["task"])
    policy_col = find_col(df, ["policy"])
    rep_col = find_col(df, ["repetition", "rep"])

    runtime_col = find_col(
        df,
        [
            "actual_seconds",
            "runtime_seconds",
            "total_seconds",
            "runtime",
        ],
    )

    memory_col = find_col(
        df,
        [
            "incremental_peak_allocated_gpu_bytes",
            "incremental_peak_reserved_gpu_bytes",
            "peak_incremental_gpu_bytes",
            "incremental_gpu_bytes",
            "incremental_memory_bytes",
            "pytorch_incremental_memory_bytes",
        ],
    )

    energy_col = find_col(
        df,
        [
            "gpu_energy_j",
            "energy_j",
            "energy_joules",
        ],
    )

    power_col = find_col(
        df,
        [
            "gpu_mean_power_w",
            "mean_gpu_power_w",
            "mean_power_w",
            "gpu_power_mean_w",
            "gpu_median_power_w",
        ],
    )

    out = pd.DataFrame({
        "request_id": df[request_col],
        "task": df[task_col],
        "policy": df[policy_col],
        "repetition": df[rep_col],
        "runtime": pd.to_numeric(df[runtime_col], errors="coerce"),
        "memory": pd.to_numeric(df[memory_col], errors="coerce"),
        "energy": pd.to_numeric(df[energy_col], errors="coerce"),
        "power": pd.to_numeric(df[power_col], errors="coerce"),
    })

    if out.isna().any().any():
        print("\nNaN counts after normalization:")
        print(out.isna().sum())
        raise ValueError("Missing values found in required hardware columns.")

    return out


# ---------------------------------------------------------------------
# Per request-policy statistics
# ---------------------------------------------------------------------

def aggregate_points(raw):
    rows = []

    for (rid, task, policy), g in raw.groupby(
        ["request_id", "task", "policy"],
        sort=False
    ):
        row = {
            "request_id": rid,
            "task": task,
            "policy": policy,
            "n_repetitions": len(g),
        }

        for metric in [
            "degradation",
            "memory",
            "runtime",
            "energy",
            "power",
        ]:
            x = g[metric].to_numpy(dtype=float)

            row[f"{metric}_median"] = np.median(x)
            row[f"{metric}_mean"] = np.mean(x)
            row[f"{metric}_std"] = np.std(x, ddof=1) if len(x) > 1 else 0.0

            # Empirical quantiles. With five runs these are sensitivity
            # summaries rather than high-confidence tail estimates.
            row[f"{metric}_p80"] = np.quantile(x, 0.80)
            row[f"{metric}_p90"] = np.quantile(x, 0.90)
            row[f"{metric}_max"] = np.max(x)

        row["runtime_cv"] = safe_cv(
            row["runtime_std"],
            row["runtime_median"],
        )
        row["energy_cv"] = safe_cv(
            row["energy_std"],
            row["energy_median"],
        )
        row["power_cv"] = safe_cv(
            row["power_std"],
            row["power_median"],
        )
        row["memory_cv"] = safe_cv(
            row["memory_std"],
            row["memory_median"],
        )

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Median objective spaces
# ---------------------------------------------------------------------

SPACES = {
    "S2_D_M": [
        "degradation_median",
        "memory_median",
    ],
    "S3_D_M_T": [
        "degradation_median",
        "memory_median",
        "runtime_median",
    ],
    "S4_D_M_T_E": [
        "degradation_median",
        "memory_median",
        "runtime_median",
        "energy_median",
    ],
    "S5_D_M_T_E_P": [
        "degradation_median",
        "memory_median",
        "runtime_median",
        "energy_median",
        "power_median",
    ],
}


def compute_frontiers(points, spaces):
    rows = []

    for space_name, objectives in spaces.items():
        for rid, g in points.groupby("request_id", sort=False):
            mask = pareto_mask(g[objectives].to_numpy())

            frontier = g.loc[mask]
            task = g["task"].iloc[0]
            sig = signature(frontier["policy"])

            for _, r in g.iterrows():
                rows.append({
                    "space": space_name,
                    "request_id": rid,
                    "task": task,
                    "policy": r["policy"],
                    "on_frontier": bool(
                        r["policy"] in set(frontier["policy"])
                    ),
                    "pareto_signature": sig,
                    "pareto_points": len(frontier),
                })

    return pd.DataFrame(rows)


def summarize_frontiers(frontiers):
    request_level = (
        frontiers.groupby(
            ["space", "request_id", "task"],
            as_index=False
        )
        .agg(
            pareto_signature=("pareto_signature", "first"),
            pareto_points=("pareto_points", "first"),
        )
    )

    summary = (
        request_level.groupby("space")
        .agg(
            requests=("request_id", "nunique"),
            distinct_pareto_sets=("pareto_signature", "nunique"),
            mean_pareto_points=("pareto_points", "mean"),
            median_pareto_points=("pareto_points", "median"),
            min_pareto_points=("pareto_points", "min"),
            max_pareto_points=("pareto_points", "max"),
        )
        .reset_index()
    )

    freq = (
        frontiers[frontiers["on_frontier"]]
        .groupby(["space", "policy"])
        .agg(
            requests_on_frontier=("request_id", "nunique")
        )
        .reset_index()
    )

    totals = (
        request_level.groupby("space")["request_id"]
        .nunique()
        .to_dict()
    )

    freq["frontier_request_rate"] = freq.apply(
        lambda r: r["requests_on_frontier"] / totals[r["space"]],
        axis=1,
    )

    return request_level, summary, freq


# ---------------------------------------------------------------------
# Objective-ablation transitions
# ---------------------------------------------------------------------

def compare_spaces(request_level):
    order = [
        "S2_D_M",
        "S3_D_M_T",
        "S4_D_M_T_E",
        "S5_D_M_T_E_P",
    ]

    rows = []

    for a, b in zip(order[:-1], order[1:]):
        xa = (
            request_level[request_level.space == a]
            [["request_id", "task", "pareto_signature", "pareto_points"]]
            .rename(columns={
                "pareto_signature": "signature_a",
                "pareto_points": "points_a",
            })
        )

        xb = (
            request_level[request_level.space == b]
            [["request_id", "pareto_signature", "pareto_points"]]
            .rename(columns={
                "pareto_signature": "signature_b",
                "pareto_points": "points_b",
            })
        )

        z = xa.merge(xb, on="request_id", how="inner")
        z["changed"] = z.signature_a != z.signature_b

        rows.append({
            "space_a": a,
            "space_b": b,
            "requests_total": len(z),
            "requests_changed": int(z.changed.sum()),
            "fraction_frontier_changed": z.changed.mean(),
            "mean_points_a": z.points_a.mean(),
            "mean_points_b": z.points_b.mean(),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Confidence-aware sensitivity
# ---------------------------------------------------------------------

def confidence_spaces(lam):
    """
    Quality degradation and memory remain at their median values.

    Runtime and energy are made conservative:
        T+ = median(T) + lambda * std(T)
        E+ = median(E) + lambda * std(E)

    Power is deliberately left at its median here so that this experiment
    isolates timing/energy measurement uncertainty.
    """
    return {
        "S4_D_M_T_E": [
            "degradation_median",
            "memory_median",
            f"runtime_lambda_{lam}",
            f"energy_lambda_{lam}",
        ],
        "S5_D_M_T_E_P": [
            "degradation_median",
            "memory_median",
            f"runtime_lambda_{lam}",
            f"energy_lambda_{lam}",
            "power_median",
        ],
    }


def confidence_sensitivity(points):
    all_frontiers = []

    tmp = points.copy()

    for lam in [0, 1, 2]:
        tmp[f"runtime_lambda_{lam}"] = (
            tmp.runtime_median + lam * tmp.runtime_std
        )
        tmp[f"energy_lambda_{lam}"] = (
            tmp.energy_median + lam * tmp.energy_std
        )

        f = compute_frontiers(
            tmp,
            confidence_spaces(lam)
        )
        f["lambda"] = lam
        all_frontiers.append(f)

    allf = pd.concat(all_frontiers, ignore_index=True)

    req = (
        allf.groupby(
            ["lambda", "space", "request_id", "task"],
            as_index=False
        )
        .agg(
            pareto_signature=("pareto_signature", "first"),
            pareto_points=("pareto_points", "first"),
        )
    )

    rows = []

    for space in ["S4_D_M_T_E", "S5_D_M_T_E_P"]:
        base = (
            req[
                (req["space"] == space)
                & (req["lambda"] == 0)
            ][["request_id", "pareto_signature"]]
            .rename(columns={
                "pareto_signature": "signature_lambda0"
            })
        )

        for lam in [1, 2]:
            cur = (
                req[
                    (req["space"] == space)
                    & (req["lambda"] == lam)
                ][["request_id", "pareto_signature"]]
                .rename(columns={
                    "pareto_signature": "signature_current"
                })
            )

            z = base.merge(cur, on="request_id")
            changed = (
                z.signature_lambda0
                != z.signature_current
            )

            rows.append({
                "space": space,
                "lambda": lam,
                "requests_total": len(z),
                "requests_changed_vs_lambda0": int(changed.sum()),
                "fraction_changed_vs_lambda0": changed.mean(),
            })

    return allf, req, pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Empirical upper-quantile robustness
# ---------------------------------------------------------------------

def quantile_spaces(q):
    suffix = {
        0.80: "p80",
        0.90: "p90",
        1.00: "max",
    }[q]

    return {
        "S4_D_M_T_E": [
            "degradation_median",
            "memory_median",
            f"runtime_{suffix}",
            f"energy_{suffix}",
        ],
        "S5_D_M_T_E_P": [
            "degradation_median",
            "memory_median",
            f"runtime_{suffix}",
            f"energy_{suffix}",
            f"power_{suffix}",
        ],
    }


def empirical_robustness(points, median_request_level):
    rows = []
    detailed = []

    labels = [
        ("p80", 0.80),
        ("p90", 0.90),
        ("max", 1.00),
    ]

    for label, q in labels:
        f = compute_frontiers(points, quantile_spaces(q))
        f["estimate"] = label
        detailed.append(f)

        req = (
            f.groupby(
                ["space", "request_id", "task"],
                as_index=False
            )
            .agg(
                pareto_signature=("pareto_signature", "first"),
                pareto_points=("pareto_points", "first"),
            )
        )

        for space in ["S4_D_M_T_E", "S5_D_M_T_E_P"]:
            base = (
                median_request_level[
                    median_request_level.space == space
                ][["request_id", "pareto_signature"]]
                .rename(columns={
                    "pareto_signature": "median_signature"
                })
            )

            cur = (
                req[req.space == space]
                [["request_id", "pareto_signature"]]
                .rename(columns={
                    "pareto_signature": "robust_signature"
                })
            )

            z = base.merge(cur, on="request_id")
            changed = (
                z.median_signature != z.robust_signature
            )

            rows.append({
                "space": space,
                "estimate": label,
                "requests_total": len(z),
                "requests_changed_vs_median": int(changed.sum()),
                "fraction_changed_vs_median": changed.mean(),
            })

    return (
        pd.concat(detailed, ignore_index=True),
        pd.DataFrame(rows),
    )


# ---------------------------------------------------------------------
# Per-task objective ablation
# ---------------------------------------------------------------------

def task_transition_summary(request_level):
    order = [
        "S2_D_M",
        "S3_D_M_T",
        "S4_D_M_T_E",
        "S5_D_M_T_E_P",
    ]

    rows = []

    for a, b in zip(order[:-1], order[1:]):
        xa = (
            request_level[request_level.space == a]
            [["request_id", "task", "pareto_signature"]]
            .rename(columns={"pareto_signature": "a"})
        )

        xb = (
            request_level[request_level.space == b]
            [["request_id", "pareto_signature"]]
            .rename(columns={"pareto_signature": "b"})
        )

        z = xa.merge(xb, on="request_id")
        z["changed"] = z.a != z.b

        for task, g in z.groupby("task"):
            rows.append({
                "task": task,
                "space_a": a,
                "space_b": b,
                "requests": len(g),
                "requests_changed": int(g.changed.sum()),
                "fraction_changed": g.changed.mean(),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        default=(
            "experiments/full_oracle_qwen3_8b/"
            "hardware/a100_profile/raw_runs.csv"
        ),
    )

    ap.add_argument(
        "--oracle",
        default=(
            "experiments/full_oracle_qwen3_8b/"
            "oracle_longbench_qwen3_8b.csv"
        ),
    )

    ap.add_argument(
        "--output-dir",
        default=(
            "experiments/full_oracle_qwen3_8b/"
            "analysis/13_multidimensional_pareto"
        ),
    )

    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(args.input)

    # ======================================================
    # Join oracle quality/degradation onto repeated hardware
    # measurements.
    #
    # Hardware measurements are repeated five times, while
    # degradation is a request-policy property from the
    # frozen oracle experiment.
    # ======================================================

    oracle = pd.read_csv(
        args.oracle
    )

    oracle_required = {
        "request_id",
        "policy",
        "degradation",
    }

    oracle_missing = (
        oracle_required
        - set(oracle.columns)
    )

    if oracle_missing:
        raise RuntimeError(
            "Oracle file missing columns: "
            f"{sorted(oracle_missing)}"
        )

    oracle_quality = (
        oracle[
            [
                "request_id",
                "policy",
                "degradation",
            ]
        ]
        .copy()
        .drop_duplicates(
            subset=[
                "request_id",
                "policy",
            ]
        )
    )

    hardware_pairs = (
        raw[
            [
                "request_id",
                "policy",
            ]
        ]
        .drop_duplicates()
    )

    alignment = hardware_pairs.merge(
        oracle_quality,
        on=[
            "request_id",
            "policy",
        ],
        how="left",
        validate="one_to_one",
    )

    missing_quality = alignment[
        "degradation"
    ].isna()

    if missing_quality.any():
        print(
            "\nMissing oracle degradation for:"
        )

        print(
            alignment.loc[
                missing_quality,
                [
                    "request_id",
                    "policy",
                ],
            ].to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Hardware/oracle alignment failed."
        )

    raw = raw.merge(
        oracle_quality,
        on=[
            "request_id",
            "policy",
        ],
        how="left",
        validate="many_to_one",
    )

    assert raw[
        "degradation"
    ].isna().sum() == 0

    print(
        "QUALITY MERGE:",
        hardware_pairs.shape[0],
        "hardware request-policy pairs aligned"
    )

    print("RAW ROWS:", len(raw))
    print("REQUESTS:", raw.request_id.nunique())
    print("POLICIES:", raw.policy.nunique())
    print("REPETITIONS:", sorted(raw.repetition.unique()))

    # Strict validation of the intended experiment.
    counts = raw.groupby(["request_id", "policy"]).size()

    assert raw.request_id.nunique() == 25
    assert raw.policy.nunique() == 8
    assert len(counts) == 200
    assert (counts == 5).all()

    points = aggregate_points(raw)
    points.to_csv(
        outdir / "request_policy_statistics.csv",
        index=False
    )

    assert len(points) == 200

    # --------------------------------------------------------------
    # PRIMARY RESULT: median Pareto spaces
    # --------------------------------------------------------------

    frontiers = compute_frontiers(points, SPACES)

    request_level, space_summary, policy_freq = (
        summarize_frontiers(frontiers)
    )

    frontiers.to_csv(
        outdir / "median_pareto_membership.csv",
        index=False
    )

    request_level.to_csv(
        outdir / "median_request_frontiers.csv",
        index=False
    )

    space_summary.to_csv(
        outdir / "median_space_summary.csv",
        index=False
    )

    policy_freq.to_csv(
        outdir / "median_policy_frontier_frequency.csv",
        index=False
    )

    # --------------------------------------------------------------
    # OBJECTIVE ABLATION
    # --------------------------------------------------------------

    transitions = compare_spaces(request_level)
    transitions.to_csv(
        outdir / "objective_ablation_transitions.csv",
        index=False
    )

    task_transitions = task_transition_summary(request_level)
    task_transitions.to_csv(
        outdir / "objective_ablation_by_task.csv",
        index=False
    )

    # --------------------------------------------------------------
    # LAMBDA SENSITIVITY
    # --------------------------------------------------------------

    conf_frontiers, conf_req, conf_summary = (
        confidence_sensitivity(points)
    )

    conf_frontiers.to_csv(
        outdir / "lambda_frontier_membership.csv",
        index=False
    )

    conf_req.to_csv(
        outdir / "lambda_request_frontiers.csv",
        index=False
    )

    conf_summary.to_csv(
        outdir / "lambda_sensitivity_summary.csv",
        index=False
    )

    # --------------------------------------------------------------
    # EMPIRICAL UPPER-QUANTILE ROBUSTNESS
    # --------------------------------------------------------------

    empirical_frontiers, empirical_summary = (
        empirical_robustness(points, request_level)
    )

    empirical_frontiers.to_csv(
        outdir / "empirical_robust_frontiers.csv",
        index=False
    )

    empirical_summary.to_csv(
        outdir / "empirical_robustness_summary.csv",
        index=False
    )

    # --------------------------------------------------------------
    # Measurement stability
    # --------------------------------------------------------------

    stability = pd.DataFrame({
        "metric": [
            "runtime",
            "energy",
            "power",
            "memory",
        ],
        "mean_cv": [
            points.runtime_cv.mean(),
            points.energy_cv.mean(),
            points.power_cv.mean(),
            points.memory_cv.mean(),
        ],
        "median_cv": [
            points.runtime_cv.median(),
            points.energy_cv.median(),
            points.power_cv.median(),
            points.memory_cv.median(),
        ],
        "p95_cv": [
            points.runtime_cv.quantile(.95),
            points.energy_cv.quantile(.95),
            points.power_cv.quantile(.95),
            points.memory_cv.quantile(.95),
        ],
        "max_cv": [
            points.runtime_cv.max(),
            points.energy_cv.max(),
            points.power_cv.max(),
            points.memory_cv.max(),
        ],
    })

    stability.to_csv(
        outdir / "measurement_stability.csv",
        index=False
    )

    # --------------------------------------------------------------
    # Console report
    # --------------------------------------------------------------

    print("\n" + "=" * 72)
    print("PRIMARY RESULT: MEDIAN PARETO SPACES")
    print("=" * 72)
    print(space_summary.to_string(index=False))

    print("\n" + "=" * 72)
    print("OBJECTIVE ABLATION")
    print("=" * 72)
    print(transitions.to_string(index=False))

    print("\n" + "=" * 72)
    print("POLICY FRONTIER FREQUENCY")
    print("=" * 72)
    print(
        policy_freq.sort_values(
            ["space", "frontier_request_rate"],
            ascending=[True, False]
        ).to_string(index=False)
    )

    print("\n" + "=" * 72)
    print("LAMBDA SENSITIVITY")
    print("=" * 72)
    print(conf_summary.to_string(index=False))

    print("\n" + "=" * 72)
    print("EMPIRICAL UPPER-QUANTILE ROBUSTNESS")
    print("=" * 72)
    print(empirical_summary.to_string(index=False))

    print("\n" + "=" * 72)
    print("MEASUREMENT STABILITY")
    print("=" * 72)
    print(stability.round(6).to_string(index=False))

    print("\nOutputs:", outdir)
    print("\nMULTIDIMENSIONAL PARETO ANALYSIS: PASS")


if __name__ == "__main__":
    main()
