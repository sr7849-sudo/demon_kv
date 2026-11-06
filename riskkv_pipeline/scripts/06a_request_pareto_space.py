from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--oracle",
        default=(
            "experiments/full_oracle_qwen3_8b/"
            "oracle_longbench_qwen3_8b.csv"
        ),
    )
    p.add_argument(
        "--outdir",
        default=(
            "experiments/full_oracle_qwen3_8b/"
            "analysis/12_request_pareto_space"
        ),
    )
    return p.parse_args()


def pareto_mask(memory, degradation):
    n = len(memory)
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            dominates = (
                memory[j] <= memory[i] + 1e-12
                and degradation[j] <= degradation[i] + 1e-12
                and (
                    memory[j] < memory[i] - 1e-12
                    or degradation[j] < degradation[i] - 1e-12
                )
            )

            if dominates:
                keep[i] = False
                break

    return keep


def main():
    args = parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.oracle)

    rows = []
    request_summary = []

    for request_id, g in df.groupby("request_id", sort=False):
        g = g.copy().reset_index(drop=True)

        mask = pareto_mask(
            g["memory_ratio"].to_numpy(float),
            g["degradation"].to_numpy(float),
        )

        g["pareto"] = mask
        g["memory_saved_ratio"] = 1.0 - g["memory_ratio"]

        rows.append(g)

        frontier = g[g["pareto"]].sort_values(
            ["memory_ratio", "degradation", "policy"]
        )

        request_summary.append(
            {
                "request_id": request_id,
                "task": g["task"].iloc[0],
                "n_points": len(g),
                "n_pareto": len(frontier),
                "pareto_policies": "|".join(frontier["policy"].tolist()),
                "pareto_signature": "|".join(sorted(frontier["policy"].tolist())),
                "min_memory_ratio_on_frontier": frontier["memory_ratio"].min(),
                "max_memory_saved_on_frontier": frontier["memory_saved_ratio"].max(),
            }
        )

    point_df = pd.concat(rows, ignore_index=True)
    req_df = pd.DataFrame(request_summary)

    point_df.to_csv(
        out / "request_pareto_points.csv",
        index=False,
    )

    req_df.to_csv(
        out / "request_pareto_summary.csv",
        index=False,
    )

    policy_freq = (
        point_df[point_df["pareto"]]
        .groupby("policy")
        .agg(
            requests_on_frontier=("request_id", "nunique"),
            mean_degradation=("degradation", "mean"),
            mean_memory_ratio=("memory_ratio", "mean"),
        )
        .reset_index()
    )

    policy_freq["frontier_request_rate"] = (
        policy_freq["requests_on_frontier"]
        / req_df["request_id"].nunique()
    )

    policy_freq.to_csv(
        out / "pareto_policy_frequency.csv",
        index=False,
    )

    signatures = (
        req_df["pareto_signature"]
        .value_counts()
        .rename_axis("pareto_signature")
        .reset_index(name="requests")
    )

    signatures["rate"] = (
        signatures["requests"] / len(req_df)
    )

    signatures.to_csv(
        out / "pareto_signature_frequency.csv",
        index=False,
    )

    per_task = (
        req_df.groupby("task")
        .agg(
            requests=("request_id", "nunique"),
            mean_pareto_points=("n_pareto", "mean"),
            median_pareto_points=("n_pareto", "median"),
            distinct_frontiers=("pareto_signature", "nunique"),
        )
        .reset_index()
    )

    per_task.to_csv(
        out / "pareto_by_task.csv",
        index=False,
    )

    print("REQUESTS:", req_df.request_id.nunique())
    print("DISTINCT PARETO SETS:", req_df.pareto_signature.nunique())
    print("MEAN PARETO POINTS:", round(req_df.n_pareto.mean(), 4))

    print("\nPOLICY FRONTIER FREQUENCY")
    print(
        policy_freq
        .sort_values("frontier_request_rate", ascending=False)
        .round(4)
        .to_string(index=False)
    )

    print("\nMOST COMMON FRONTIERS")
    print(
        signatures.head(20)
        .round(4)
        .to_string(index=False)
    )

    print("\nPER TASK")
    print(
        per_task.round(4).to_string(index=False)
    )

    print("\nREQUEST PARETO ANALYSIS: PASS")


if __name__ == "__main__":
    main()
