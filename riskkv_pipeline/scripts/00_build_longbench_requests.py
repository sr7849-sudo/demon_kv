from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


REVISION = "c05c7428de23cd8ad4df6ed748b118c602022972"

TASKS = {
    "qasper": {
        "category": "single_doc_qa",
        "metric": "official_longbench",
    },
    "hotpotqa": {
        "category": "multi_doc_qa",
        "metric": "official_longbench",
    },
    "passage_retrieval_en": {
        "category": "retrieval",
        "metric": "official_longbench",
    },
    "multifieldqa_en": {
        "category": "structured_qa",
        "metric": "official_longbench",
    },
    "triviaqa": {
        "category": "long_context_qa",
        "metric": "official_longbench",
    },
}


def load_task(task: str):
    filename = f"{task}/test-00000-of-00001.parquet"

    path = hf_hub_download(
        repo_id="zai-org/LongBench",
        filename=filename,
        repo_type="dataset",
        revision=REVISION,
    )

    ds = load_dataset(
        "parquet",
        data_files=path,
        split="train",
    )

    return ds, str(path)


def choose_reference(answers):
    if answers is None:
        return ""

    if isinstance(answers, str):
        return answers

    if isinstance(answers, (list, tuple)):
        if len(answers) == 0:
            return ""

        # For now use the first canonical reference.
        # The oracle evaluator can later be extended to max-over-references.
        return str(answers[0])

    return str(answers)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
    )

    ap.add_argument(
        "--per-task",
        type=int,
        default=60,
    )

    ap.add_argument(
        "--min-tokens",
        type=int,
        default=1500,
    )

    ap.add_argument(
        "--max-tokens",
        type=int,
        default=12000,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    ap.add_argument(
        "--output",
        default="data/requests_longbench.jsonl",
    )

    ap.add_argument(
        "--manifest",
        default="data/requests_longbench_manifest.json",
    )

    args = ap.parse_args()

    rng = random.Random(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    output_rows: List[Dict] = []
    manifest = {
        "model_tokenizer": args.model,
        "longbench_revision": REVISION,
        "seed": args.seed,
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "per_task_target": args.per_task,
        "tasks": {},
    }

    for task, meta in TASKS.items():
        print(f"\n=== Loading {task} ===")

        ds, source_path = load_task(task)

        candidates = []

        for row in ds:
            context = row["context"]
            question = row["input"]
            answers = row["answers"]

            if not context or not question:
                continue

            prompt_text = context + "\n" + question

            token_ids = tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]

            n_tokens = len(token_ids)

            if n_tokens < args.min_tokens:
                continue

            if n_tokens > args.max_tokens:
                continue

            reference = choose_reference(answers)

            if reference == "":
                continue

            req_id = f"{task}:{row['_id']}"

            candidates.append(
                {
                    "request_id": req_id,
                    "task": task,
                    "task_category": meta["category"],
                    "context": context,
                    "question": question,
                    "reference": reference,
                    "references": list(answers)
                    if isinstance(answers, (list, tuple))
                    else [reference],
                    "metric": meta["metric"],
                    "context_tokens": n_tokens,
                    "longbench_reported_length": row.get("length"),
                    "dataset": row.get("dataset", task),
                    "language": row.get("language", "en"),
                    "source_id": row["_id"],
                    "all_classes": row.get("all_classes"),
                }
            )

        rng.shuffle(candidates)

        selected = candidates[: args.per_task]

        output_rows.extend(selected)

        manifest["tasks"][task] = {
            "category": meta["category"],
            "metric": meta["metric"],
            "source_path": source_path,
            "available_rows": len(ds),
            "eligible_rows": len(candidates),
            "selected_rows": len(selected),
            "min_selected_tokens": min(
                [x["context_tokens"] for x in selected],
                default=None,
            ),
            "max_selected_tokens": max(
                [x["context_tokens"] for x in selected],
                default=None,
            ),
        }

        print(
            f"{task}: "
            f"available={len(ds)} "
            f"eligible={len(candidates)} "
            f"selected={len(selected)}"
        )

    rng.shuffle(output_rows)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest["total_requests"] = len(output_rows)

    manifest_path = Path(args.manifest)

    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print("\n====================================")
    print("BUILD COMPLETE")
    print("====================================")
    print("Requests:", len(output_rows))
    print("Output:", out_path)
    print("Manifest:", manifest_path)

    counts = {}
    for row in output_rows:
        counts[row["task"]] = counts.get(row["task"], 0) + 1

    print("\nTask counts:")
    for task, n in sorted(counts.items()):
        print(f"  {task}: {n}")


if __name__ == "__main__":
    main()
