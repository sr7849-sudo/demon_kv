from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DELTAS = [0.01, 0.05, 0.10]

ORDERED_FAMILIES = {
    "snapkv": ["full", "snapkv_25", "snapkv_50", "snapkv_75"],
    "knorm":  ["full", "knorm_25", "knorm_50", "knorm_75"],
}


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
            "analysis/08_compression_profiles"
        ),
    )

    p.add_argument(
        "--cliff-threshold",
        type=float,
        default=0.25,
        help="Adjacent quality-loss jump regarded as a compression cliff.",
    )

    return p.parse_args()


def family_curve(group, policies):
    rows = []

    for policy in policies:
        x = group[group["policy"] == policy]

        if len(x) != 1:
            continue

        row = x.iloc[0]

        rows.append(
            {
                "policy": policy,
                "memory_ratio": float(row["memory_ratio"]),
                "severity": 1.0 - float(row["memory_ratio"]),
                "degradation": float(row["degradation"]),
            }
        )

    if len(rows) < 2:
        return None

    curve = (
        pd.DataFrame(rows)
        .sort_values("severity")
        .reset_index(drop=True)
    )

    return curve


def curve_fragility(curve):
    severity = curve["severity"].to_numpy(float)
    degradation = curve["degradation"].to_numpy(float)

    ds = np.diff(severity)
    dd = np.diff(degradation)

    positive_jump = np.maximum(dd, 0.0)

    slope = np.divide(
        positive_jump,
        ds,
        out=np.zeros_like(positive_jump),
        where=ds > 0,
    )

    max_jump_idx = int(np.argmax(positive_jump))

    auc = float(
        np.trapezoid(
            degradation,
            severity,
        )
    )

    return {
        "max_adjacent_jump": float(positive_jump.max()),
        "max_positive_slope": float(slope.max()),
        "fragility_auc": auc,
        "cliff_start_severity": float(
            severity[max_jump_idx]
        ),
        "cliff_end_severity": float(
            severity[max_jump_idx + 1]
        ),
        "max_curve_degradation": float(
            degradation.max()
        ),
        "monotonicity_violations": int(
            (dd < -1e-12).sum()
        ),
    }


def main():
    args = parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    if df["request_id"].nunique() != 290:
        raise RuntimeError(
            f"Expected 290 requests, got "
            f"{df['request_id'].nunique()}"
        )

    request_rows = []
    family_rows = []

    for request_id, g in df.groupby("request_id", sort=False):
        base = g.iloc[0]

        row = {
            "request_id": request_id,
            "task": base["task"],
            "context_tokens": base.get("context_tokens", np.nan),
            "full_quality": base["full_quality"],
        }

        # -------------------------------------------------
        # HEADROOM
        # -------------------------------------------------
        #
        # headroom_delta = maximum removable KV fraction
        # among policies that satisfy degradation <= delta.
        #
        for delta in DELTAS:
            feasible = g[
                g["degradation"]
                <= delta + 1e-12
            ].copy()

            if len(feasible) == 0:
                raise RuntimeError(
                    f"No feasible policy for {request_id}"
                )

            feasible["memory_saved_ratio"] = (
                1.0
                - feasible["memory_ratio"]
            )

            best = feasible.sort_values(
                [
                    "memory_ratio",
                    "degradation",
                    "policy",
                ],
                ascending=[
                    True,
                    True,
                    True,
                ],
            ).iloc[0]

            tag = f"{delta:.2f}".replace(".", "p")

            row[f"headroom_{tag}"] = (
                1.0
                - float(best["memory_ratio"])
            )

            row[f"oracle_policy_{tag}"] = (
                best["policy"]
            )

            row[f"oracle_degradation_{tag}"] = (
                float(best["degradation"])
            )

        # -------------------------------------------------
        # FRAGILITY
        # -------------------------------------------------
        family_fragilities = []

        for family, policies in ORDERED_FAMILIES.items():
            curve = family_curve(
                g,
                policies,
            )

            if curve is None:
                continue

            frag = curve_fragility(curve)

            family_rows.append(
                {
                    "request_id": request_id,
                    "task": base["task"],
                    "family": family,
                    **frag,
                }
            )

            for key, value in frag.items():
                row[
                    f"{family}_{key}"
                ] = value

            family_fragilities.append(
                frag
            )

        if family_fragilities:
            row["fragility_score"] = max(
                x["max_positive_slope"]
                for x in family_fragilities
            )

            row["max_adjacent_jump"] = max(
                x["max_adjacent_jump"]
                for x in family_fragilities
            )

            row["fragility_auc_max"] = max(
                x["fragility_auc"]
                for x in family_fragilities
            )

            row["cliff"] = int(
                row["max_adjacent_jump"]
                >= args.cliff_threshold
            )

            row["catastrophic_any"] = int(
                g["degradation"].max()
                >= 0.25
            )

        request_rows.append(row)

    request_profiles = pd.DataFrame(
        request_rows
    )

    family_profiles = pd.DataFrame(
        family_rows
    )

    request_profiles.to_csv(
        out / "oracle_request_profiles.csv",
        index=False,
    )

    family_profiles.to_csv(
        out / "oracle_family_fragility.csv",
        index=False,
    )

    print("REQUESTS:", len(request_profiles))

    print("\nHEADROOM")
    for delta in DELTAS:
        tag = f"{delta:.2f}".replace(".", "p")
        col = f"headroom_{tag}"

        print(
            f"delta={delta:.2f}",
            request_profiles[col]
            .value_counts()
            .sort_index()
            .to_dict(),
        )

        print(
            " mean=",
            round(
                request_profiles[col].mean(),
                4,
            ),
        )

    print("\nFRAGILITY")
    print(
        request_profiles[
            [
                "fragility_score",
                "max_adjacent_jump",
                "fragility_auc_max",
                "cliff",
                "catastrophic_any",
            ]
        ]
        .describe()
        .round(4)
    )

    print(
        "\nCLIFF RATE:",
        round(
            request_profiles["cliff"].mean(),
            4,
        ),
    )

    print("\nCOMPRESSION PROFILE CONSTRUCTION: PASS")


if __name__ == "__main__":
    main()
