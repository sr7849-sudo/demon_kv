from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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
            "analysis/01_oracle_landscape"
        ),
    )
    return p.parse_args()


def percentile(q):
    def f(x):
        return x.quantile(q)
    f.__name__ = f"p{int(q * 100)}"
    return f


def main():
    args = parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    required = {
        "request_id",
        "task",
        "policy",
        "family",
        "quality",
        "full_quality",
        "signed_delta",
        "degradation",
        "memory_ratio",
    }

    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df["request_id"].nunique() != 290:
        raise RuntimeError(
            f"Expected 290 requests, got "
            f"{df['request_id'].nunique()}"
        )

    print("Rows:", len(df))
    print("Requests:", df["request_id"].nunique())
    print("Policies:", df["policy"].nunique())
    print("Tasks:", df["task"].nunique())

    compressed = df[df["policy"] != "full"].copy()

    compressed["robust_001"] = (
        compressed["degradation"] <= 0.01
    )
    compressed["robust_005"] = (
        compressed["degradation"] <= 0.05
    )
    compressed["robust_010"] = (
        compressed["degradation"] <= 0.10
    )
    compressed["catastrophic_025"] = (
        compressed["degradation"] >= 0.25
    )
    compressed["improved"] = (
        compressed["signed_delta"] < 0
    )

    summary = (
        compressed
        .groupby(["task", "policy"], as_index=False)
        .agg(
            n=("request_id", "nunique"),
            mean_quality=("quality", "mean"),
            mean_full_quality=("full_quality", "mean"),
            mean_degradation=("degradation", "mean"),
            median_degradation=("degradation", "median"),
            std_degradation=("degradation", "std"),
            p90_degradation=("degradation", percentile(0.90)),
            p95_degradation=("degradation", percentile(0.95)),
            max_degradation=("degradation", "max"),
            mean_signed_delta=("signed_delta", "mean"),
            robust_001_rate=("robust_001", "mean"),
            robust_005_rate=("robust_005", "mean"),
            robust_010_rate=("robust_010", "mean"),
            catastrophic_025_rate=("catastrophic_025", "mean"),
            improved_rate=("improved", "mean"),
            mean_memory_ratio=("memory_ratio", "mean"),
        )
    )

    summary.to_csv(
        out / "task_policy_landscape.csv",
        index=False,
    )

    policy_summary = (
        compressed
        .groupby("policy", as_index=False)
        .agg(
            n=("request_id", "nunique"),
            mean_quality=("quality", "mean"),
            mean_degradation=("degradation", "mean"),
            median_degradation=("degradation", "median"),
            std_degradation=("degradation", "std"),
            p90_degradation=("degradation", percentile(0.90)),
            p95_degradation=("degradation", percentile(0.95)),
            max_degradation=("degradation", "max"),
            robust_001_rate=("robust_001", "mean"),
            robust_005_rate=("robust_005", "mean"),
            robust_010_rate=("robust_010", "mean"),
            catastrophic_025_rate=("catastrophic_025", "mean"),
            improved_rate=("improved", "mean"),
            mean_memory_ratio=("memory_ratio", "mean"),
        )
    )

    policy_summary.to_csv(
        out / "overall_policy_landscape.csv",
        index=False,
    )

    full_quality = (
        df[df["policy"] == "full"]
        .groupby("task", as_index=False)
        .agg(
            n=("request_id", "nunique"),
            mean=("quality", "mean"),
            median=("quality", "median"),
            std=("quality", "std"),
            min=("quality", "min"),
            max=("quality", "max"),
        )
    )

    full_quality.to_csv(
        out / "fullkv_quality_by_task.csv",
        index=False,
    )

    # Request × policy degradation matrix.
    matrix = compressed.pivot(
        index="request_id",
        columns="policy",
        values="degradation",
    )
    matrix.to_csv(
        out / "request_policy_degradation_matrix.csv"
    )

    print("\nOVERALL POLICY LANDSCAPE")
    print(
        policy_summary.round(4).to_string(index=False)
    )

    print("\nTASK/POLICY SUMMARY written to:")
    print(out)

    # Publication-oriented figures.
    try:
        import matplotlib.pyplot as plt

        pivot = summary.pivot(
            index="task",
            columns="policy",
            values="mean_degradation",
        )

        fig, ax = plt.subplots(figsize=(11, 5))
        im = ax.imshow(
            pivot.values,
            aspect="auto",
        )
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(
            pivot.columns,
            rotation=45,
            ha="right",
        )
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(
            "Mean KV Compression Degradation by Task and Policy"
        )
        ax.set_xlabel("KV policy")
        ax.set_ylabel("LongBench task")
        fig.colorbar(
            im,
            ax=ax,
            label="Mean quality degradation",
        )
        fig.tight_layout()
        fig.savefig(
            out / "mean_degradation_heatmap.png",
            dpi=250,
        )
        plt.close(fig)

        order = (
            compressed
            .groupby("policy")["degradation"]
            .median()
            .sort_values()
            .index
        )

        data = [
            compressed.loc[
                compressed["policy"] == p,
                "degradation",
            ].values
            for p in order
        ]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(
            data,
            tick_labels=order,
            showfliers=False,
        )
        ax.set_ylabel("Quality degradation")
        ax.set_xlabel("KV policy")
        ax.set_title(
            "Distribution of Request-Level Quality Degradation"
        )
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(
            out / "degradation_distribution.png",
            dpi=250,
        )
        plt.close(fig)

    except ImportError:
        print(
            "matplotlib not installed; CSV analyses completed."
        )

    print("\nORACLE LANDSCAPE: PASS")


if __name__ == "__main__":
    main()
