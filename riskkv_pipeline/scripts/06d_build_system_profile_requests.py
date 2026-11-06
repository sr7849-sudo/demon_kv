from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input",
        default="data/requests_longbench.jsonl",
    )

    p.add_argument(
        "--output",
        default="data/system_profile25.jsonl",
    )

    return p.parse_args()


def main():
    args = parse_args()

    rows = [
        json.loads(x)
        for x in open(
            args.input,
            encoding="utf-8",
        )
    ]

    by_task = defaultdict(list)

    for row in rows:
        by_task[
            row["task"]
        ].append(row)

    selected = []

    for task in sorted(
        by_task
    ):
        xs = sorted(
            by_task[task],
            key=lambda r: r[
                "context_tokens"
            ],
        )

        # Approximate min / Q1 / median / Q3 / max.
        fractions = [
            0.00,
            0.25,
            0.50,
            0.75,
            1.00,
        ]

        indices = []

        for f in fractions:
            i = int(
                round(
                    f
                    * (
                        len(xs)
                        - 1
                    )
                )
            )

            indices.append(i)

        # Remove accidental duplicates for tiny tasks.
        indices = sorted(
            set(indices)
        )

        for i in indices:
            row = dict(
                xs[i]
            )

            row[
                "hardware_profile_rank"
            ] = i

            selected.append(
                row
            )

    Path(
        args.output
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as f:
        for row in selected:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        "REQUESTS:",
        len(selected),
    )

    for row in selected:
        print(
            f"{row['task']:24s}",
            row[
                "context_tokens"
            ],
            row[
                "request_id"
            ],
        )

    if len(selected) != 25:
        print(
            "WARNING: expected 25 requests; "
            "check task sizes."
        )

    else:
        print(
            "SYSTEM PROFILE SET: PASS"
        )


if __name__ == "__main__":
    main()
