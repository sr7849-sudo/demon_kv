from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import transformers
from tqdm import tqdm

from riskkv.backend import KVPressBackend
from riskkv.config import load_config, validate_config
from riskkv.features import extract_prefill_features
from riskkv.longbench_eval import (
    build_longbench_prompt,
    score_longbench,
)
from riskkv.metrics import score
from riskkv.kv_runtime_policies import PolicySpec


ROOT = Path(__file__).resolve().parents[1]

MAXLEN_PATH = (
    ROOT
    / "third_party"
    / "longbench"
    / "dataset2maxlen.json"
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--config",
        default="configs/default.yaml",
    )

    p.add_argument(
        "--input",
        default="data/requests.jsonl",
    )

    p.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    p.add_argument(
        "--output",
        default="results/oracle.csv",
    )

    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore an existing output CSV and start over.",
    )

    return p.parse_args()


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def load_maxlens():
    if not MAXLEN_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MAXLEN_PATH}. "
            "Download official LongBench dataset2maxlen.json first."
        )

    return json.load(
        open(MAXLEN_PATH, "r", encoding="utf-8")
    )


def evaluate_answer(row, answer):
    metric = row.get("metric", "token_f1")

    if metric == "official_longbench":
        refs = row.get("references") or [
            row["reference"]
        ]

        return float(
            score_longbench(
                dataset=row["task"],
                prediction=answer,
                references=refs,
                all_classes=row.get("all_classes"),
            )
        )

    return float(
        score(
            answer,
            row["reference"],
            metric,
        )
    )


def generation_inputs(row):
    if row.get("metric") == "official_longbench":
        formatted = build_longbench_prompt(
            dataset=row["task"],
            context=row["context"],
            user_input=row["question"],
        )

        return formatted, ""

    return row["context"], row["question"]


def max_new_tokens_for(row, model_cfg, maxlens):
    if row.get("metric") == "official_longbench":
        task = row["task"]

        if task not in maxlens:
            raise KeyError(
                f"Missing official LongBench maxlen for {task}"
            )

        return int(maxlens[task])

    return int(
        model_cfg.get("max_new_tokens", 64)
    )


def dtype_nbytes(model):
    dtype = next(model.parameters()).dtype

    return torch.empty(
        (),
        dtype=dtype,
    ).element_size()


def full_kv_bytes(model_config, seq_len, model):
    """
    Analytical full-KV tensor footprint.

    K and V are both stored:
      layers × tokens × KV heads × head_dim × 2 × dtype_bytes

    This is intentionally labelled theoretical rather than measured
    allocator memory.
    """

    n_layers = int(model_config.num_hidden_layers)

    n_kv_heads = int(
        getattr(
            model_config,
            "num_key_value_heads",
            model_config.num_attention_heads,
        )
    )

    head_dim = getattr(
        model_config,
        "head_dim",
        None,
    )

    if head_dim is None:
        head_dim = (
            model_config.hidden_size
            // model_config.num_attention_heads
        )

    return int(
        n_layers
        * int(seq_len)
        * n_kv_heads
        * int(head_dim)
        * 2
        * dtype_nbytes(model)
    )


def build_manifest(
    args,
    cfg,
    policies,
    rows,
    static_gpu_bytes,
):
    manifest = {
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "input": str(Path(args.input).resolve()),
        "input_sha256": sha256_file(args.input),

        "config": str(
            Path(args.config).resolve()
        ),
        "config_sha256": sha256_file(
            args.config
        ),

        "n_requests": len(rows),

        "model": cfg["model"]["name"],

        "policies": [
            {
                "name": p.name,
                "family": p.family,
                "compression_ratio": (
                    p.compression_ratio
                ),
                "memory_ratio": p.memory_ratio,
                "nbits": p.nbits,
                "backend": p.backend,
            }
            for p in policies
        ],

        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,

        "cuda_available": (
            torch.cuda.is_available()
        ),
        "torch_cuda": torch.version.cuda,

        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),

        "gpu_capability": (
            list(
                torch.cuda.get_device_capability(0)
            )
            if torch.cuda.is_available()
            else None
        ),

        "static_model_gpu_bytes": (
            int(static_gpu_bytes)
        ),

        "git_commit": git_commit(),

        "longbench_maxlen_file": str(
            MAXLEN_PATH.resolve()
        ),
        "longbench_maxlen_sha256": (
            sha256_file(MAXLEN_PATH)
        ),
    }

    return manifest


def atomic_write_csv(df, output):
    tmp = output.with_suffix(
        output.suffix + ".tmp"
    )

    df.to_csv(
        tmp,
        index=False,
    )

    tmp.replace(output)


def main():
    args = parse_args()

    cfg = load_config(args.config)
    validate_config(cfg)

    rows = read_jsonl(args.input)

    if args.limit is not None:
        rows = rows[:args.limit]

    maxlens = load_maxlens()

    m = cfg["model"]

    backend = KVPressBackend(
        m["name"],
        device=m.get("device", "cuda:0"),
        dtype=m.get("dtype", "auto"),
        device_map=m.get("device_map"),
        trust_remote_code=m.get(
            "trust_remote_code",
            False,
        ),
    )

    policies = [
        PolicySpec.from_dict(x)
        for x in cfg["policies"]
    ]

    expected_policy_names = {
        p.name
        for p in policies
    }

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        static_gpu_bytes = int(
            torch.cuda.memory_allocated()
        )
    else:
        static_gpu_bytes = 0

    output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = output.with_suffix(
        ".manifest.json"
    )

    manifest = build_manifest(
        args,
        cfg,
        policies,
        rows,
        static_gpu_bytes,
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
    )

    existing_records = []

    completed_requests = set()

    if (
        output.exists()
        and not args.no_resume
    ):
        existing = pd.read_csv(output)

        existing_records = (
            existing.to_dict("records")
        )

        # A request counts as complete only when every currently
        # configured policy is present.
        for request_id, g in existing.groupby(
            "request_id"
        ):
            present = set(g["policy"])

            if expected_policy_names.issubset(
                present
            ):
                completed_requests.add(
                    request_id
                )

        print(
            f"Resume: {len(completed_requests)} "
            f"requests already complete."
        )

    records = list(existing_records)

    pending_rows = [
        r
        for r in rows
        if r["request_id"]
        not in completed_requests
    ]

    print(
        f"Requests total={len(rows)} "
        f"complete={len(completed_requests)} "
        f"pending={len(pending_rows)}"
    )

    for row in tqdm(
        pending_rows,
        desc="requests",
    ):
        context_for_model, question_for_model = (
            generation_inputs(row)
        )

        max_new_tokens = max_new_tokens_for(
            row,
            m,
            maxlens,
        )

        feats = extract_prefill_features(
            backend.model,
            backend.tokenizer,
            context_for_model,
            question_for_model,
            max_tokens=int(
                m.get(
                    "feature_max_tokens",
                    32768,
                )
            ),
        )

        prompt_tokens = int(
            feats.get(
                "context_tokens",
                row.get(
                    "context_tokens",
                    0,
                ),
            )
        )

        kv_full = full_kv_bytes(
            backend.model.config,
            prompt_tokens,
            backend.model,
        )

        per_policy = []

        for policy in policies:
            gr = backend.generate(
                context_for_model,
                question_for_model,
                policy,
                max_new_tokens=max_new_tokens,
                do_sample=m.get(
                    "do_sample",
                    False,
                ),
                temperature=m.get(
                    "temperature",
                    0.0,
                ),
            )

            quality = evaluate_answer(
                row,
                gr.answer,
            )

            kv_estimated = int(
                kv_full
                * float(
                    policy.memory_ratio
                )
            )

            peak_incremental = max(
                0,
                int(gr.peak_gpu_bytes)
                - static_gpu_bytes,
            )

            rec = {
                "request_id": row[
                    "request_id"
                ],
                "task": row["task"],
                "task_category": row.get(
                    "task_category"
                ),
                "metric": row.get("metric"),

                "policy": policy.name,
                "family": policy.family,
                "compression_ratio": (
                    policy.compression_ratio
                ),
                "memory_ratio": (
                    policy.memory_ratio
                ),

                "reference": row[
                    "reference"
                ],
                "references": json.dumps(
                    row.get(
                        "references",
                        [row["reference"]],
                    ),
                    ensure_ascii=False,
                ),

                "answer": gr.answer,
                "quality": quality,

                "max_new_tokens": (
                    max_new_tokens
                ),

                "actual_seconds": (
                    gr.total_seconds
                ),

                "static_model_gpu_bytes": (
                    static_gpu_bytes
                ),

                "actual_peak_gpu_bytes": (
                    gr.peak_gpu_bytes
                ),

                "actual_allocated_gpu_bytes": (
                    gr.allocated_gpu_bytes
                ),

                "peak_incremental_gpu_bytes": (
                    peak_incremental
                ),

                # Analytical KV accounting.
                "kv_bytes_full_theoretical": (
                    kv_full
                ),

                "kv_bytes_policy_estimated": (
                    kv_estimated
                ),

                "kv_bytes_saved_estimated": (
                    kv_full
                    - kv_estimated
                ),

                **feats,
            }

            per_policy.append(rec)

        full_rows = [
            x
            for x in per_policy
            if (
                x["family"] == "full"
                or x["policy"] == "full"
            )
        ]

        if len(full_rows) != 1:
            raise RuntimeError(
                f"{row['request_id']}: "
                f"expected one FullKV result, "
                f"got {len(full_rows)}"
            )

        full_quality = float(
            full_rows[0]["quality"]
        )

        for rec in per_policy:
            signed_delta = (
                full_quality
                - float(rec["quality"])
            )

            rec["full_quality"] = (
                full_quality
            )

            rec["signed_delta"] = (
                signed_delta
            )

            rec["degradation"] = max(
                0.0,
                signed_delta,
            )

        # If a previous interrupted run left an incomplete request,
        # discard its partial rows before replacing it.
        request_id = row["request_id"]

        records = [
            r
            for r in records
            if r["request_id"]
            != request_id
        ]

        records.extend(per_policy)

        # Atomic checkpoint after EVERY request.
        atomic_write_csv(
            pd.DataFrame(records),
            output,
        )

    final = pd.DataFrame(records)

    print(
        f"Wrote {len(final)} rows "
        f"to {output}"
    )

    print(
        f"Complete requests: "
        f"{final['request_id'].nunique()}"
    )


if __name__ == "__main__":
    main()
