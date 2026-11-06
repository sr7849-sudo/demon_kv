from __future__ import annotations

import argparse
import gc
import json
import platform
import socket
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from riskkv.backend import KVPressBackend
from riskkv.gpu_telemetry import NVMLSampler
from riskkv.kv_runtime_policies import PolicySpec


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--config",
        default="configs/default.yaml",
    )

    p.add_argument(
        "--input",
        default="data/system_profile25.jsonl",
    )

    p.add_argument(
        "--output",
        default=(
            "experiments/full_oracle_qwen3_8b/"
            "hardware/a100_profile/raw_runs.csv"
        ),
    )

    p.add_argument(
        "--warmups",
        type=int,
        default=2,
    )

    p.add_argument(
        "--repetitions",
        type=int,
        default=5,
    )

    p.add_argument(
        "--telemetry-interval",
        type=float,
        default=0.02,
    )

    p.add_argument(
        "--cooldown",
        type=float,
        default=0.05,
    )

    return p.parse_args()


def read_jsonl(path):
    return [
        json.loads(x)
        for x in open(
            path,
            encoding="utf-8",
        )
        if x.strip()
    ]


def policy_from_dict(x):
    return PolicySpec(
        name=x["name"],
        family=x["family"],
        compression_ratio=x.get(
            "compression_ratio"
        ),
        memory_ratio=float(
            x["memory_ratio"]
        ),
        nbits=x.get("nbits"),
        backend=x.get("backend"),
    )


def model_inputs(row):
    # Your LongBench request builder stores the already
    # prepared question/input in "question".
    return (
        row["context"],
        row["question"],
    )


def generated_tokens(
    tokenizer,
    answer,
):
    return len(
        tokenizer(
            answer,
            add_special_tokens=False,
        )["input_ids"]
    )


def run_once(
    backend,
    row,
    policy,
    max_new_tokens,
    do_sample,
    temperature,
    telemetry_interval,
):
    context, question = model_inputs(
        row
    )

    sampler = NVMLSampler(
        device_index=0,
        interval_s=telemetry_interval,
    )

    sampler.start()

    gr = backend.generate(
        context,
        question,
        policy,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
    )

    telemetry = sampler.stop()

    n_generated = generated_tokens(
        backend.tokenizer,
        gr.answer,
    )

    tokens_per_second = (
        n_generated
        / gr.total_seconds
        if gr.total_seconds > 0
        else np.nan
    )

    return (
        gr,
        telemetry,
        n_generated,
        tokens_per_second,
    )


def main():
    args = parse_args()

    cfg = yaml.safe_load(
        open(
            args.config,
            encoding="utf-8",
        )
    )

    model_cfg = cfg["model"]

    rows = read_jsonl(
        args.input
    )

    policies = [
        policy_from_dict(x)
        for x in cfg["policies"]
    ]

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required."
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "Capability:",
        torch.cuda.get_device_capability(0),
    )

    print(
        "Requests:",
        len(rows),
    )

    print(
        "Policies:",
        len(policies),
    )

    print(
        "Measured executions:",
        (
            len(rows)
            * len(policies)
            * args.repetitions
        ),
    )

    backend = KVPressBackend(
        model_name=model_cfg["name"],
        device=model_cfg.get(
            "device",
            "cuda:0",
        ),
        dtype=model_cfg.get(
            "dtype",
            "auto",
        ),
        device_map=model_cfg.get(
            "device_map"
        ),
        trust_remote_code=model_cfg.get(
            "trust_remote_code",
            False,
        ),
    )

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------
    # Resume existing measurements if present.
    # -------------------------------------------------
    if output.exists():
        existing = pd.read_csv(output)

        measured = existing.to_dict(
            orient="records"
        )

        completed_reps = (
            existing
            .groupby(
                ["request_id", "policy"]
            )
            .size()
            .to_dict()
        )

        print(
            "Resume rows:",
            len(existing),
        )

        print(
            "Resume request-policy pairs:",
            len(completed_reps),
        )

    else:
        measured = []
        completed_reps = {}

    total_groups = (
        len(rows)
        * len(policies)
    )

    progress = tqdm(
        total=total_groups,
        desc="request-policy pairs",
    )

    for request_index, row in enumerate(
        rows
    ):
        max_new_tokens = int(
            row.get(
                "max_new_tokens",
                model_cfg.get(
                    "max_new_tokens",
                    64,
                ),
            )
        )

        do_sample = bool(
            model_cfg.get(
                "do_sample",
                False,
            )
        )

        temperature = float(
            model_cfg.get(
                "temperature",
                0.0,
            )
        )

        for policy in policies:

            pair_key = (
                row["request_id"],
                policy.name,
            )

            n_existing = int(
                completed_reps.get(
                    pair_key,
                    0,
                )
            )

            if n_existing >= args.repetitions:
                progress.update(1)
                continue

            # -----------------------------------------
            # Warmups — deliberately not recorded.
            # -----------------------------------------
            warmup_failed = False

            for warmup_idx in range(
                args.warmups
            ):
                try:
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                    _ = backend.generate(
                        row["context"],
                        row["question"],
                        policy,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                    )

                    torch.cuda.synchronize()

                except Exception as e:
                    print(
                        "\nWARMUP FAILURE",
                        "request=",
                        row["request_id"],
                        "policy=",
                        policy.name,
                        "warmup=",
                        warmup_idx,
                    )

                    print(
                        type(e).__name__,
                        str(e),
                    )

                    traceback.print_exc()

                    # Clear Python/CUDA state before retry.
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                    warmup_failed = True
                    break

            if warmup_failed:
                print(
                    "Retrying pair with fresh backend..."
                )

                del backend

                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                backend = KVPressBackend(
                    model_name=model_cfg["name"],
                    device=model_cfg.get(
                        "device",
                        "cuda:0",
                    ),
                    dtype=model_cfg.get(
                        "dtype",
                        "auto",
                    ),
                    device_map=model_cfg.get(
                        "device_map"
                    ),
                    trust_remote_code=model_cfg.get(
                        "trust_remote_code",
                        False,
                    ),
                )

                # One clean warmup after backend reset.
                _ = backend.generate(
                    row["context"],
                    row["question"],
                    policy,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                )

                torch.cuda.synchronize()

            # -----------------------------------------
            # Measured repetitions.
            # -----------------------------------------
            for rep in range(
                n_existing,
                args.repetitions,
            ):
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                (
                    gr,
                    telem,
                    n_generated,
                    tokens_per_second,
                ) = run_once(
                    backend=backend,
                    row=row,
                    policy=policy,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    telemetry_interval=(
                        args.telemetry_interval
                    ),
                )

                record = {
                    "request_id": (
                        row["request_id"]
                    ),
                    "task": row["task"],
                    "request_index": (
                        request_index
                    ),
                    "context_tokens": (
                        row[
                            "context_tokens"
                        ]
                    ),

                    "policy": (
                        policy.name
                    ),
                    "family": (
                        policy.family
                    ),
                    "memory_ratio": (
                        policy.memory_ratio
                    ),
                    "compression_ratio": (
                        policy.compression_ratio
                    ),
                    "nbits": policy.nbits,

                    "repetition": rep,

                    "max_new_tokens": (
                        max_new_tokens
                    ),
                    "generated_tokens": (
                        n_generated
                    ),

                    "actual_seconds": (
                        gr.total_seconds
                    ),

                    "generated_tokens_per_second": (
                        tokens_per_second
                    ),

                    "baseline_allocated_gpu_bytes": (
                        gr.baseline_allocated_gpu_bytes
                    ),

                    "baseline_reserved_gpu_bytes": (
                        gr.baseline_reserved_gpu_bytes
                    ),

                    "peak_allocated_gpu_bytes": (
                        gr.peak_gpu_bytes
                    ),

                    "peak_reserved_gpu_bytes": (
                        gr.peak_reserved_gpu_bytes
                    ),

                    "incremental_peak_allocated_gpu_bytes": (
                        gr.incremental_peak_allocated_gpu_bytes
                    ),

                    "incremental_peak_reserved_gpu_bytes": (
                        gr.incremental_peak_reserved_gpu_bytes
                    ),

                    "final_allocated_gpu_bytes": (
                        gr.allocated_gpu_bytes
                    ),

                    "nvml_mean_memory_bytes": (
                        telem.mean_nvml_memory_bytes
                    ),

                    "nvml_peak_memory_bytes": (
                        telem.peak_nvml_memory_bytes
                    ),

                    "gpu_mean_power_w": (
                        telem.mean_power_w
                    ),

                    "gpu_median_power_w": (
                        telem.median_power_w
                    ),

                    "gpu_peak_power_w": (
                        telem.peak_power_w
                    ),

                    "gpu_energy_j": (
                        telem.energy_j
                    ),

                    "gpu_mean_util_pct": (
                        telem.mean_gpu_util_pct
                    ),

                    "gpu_peak_util_pct": (
                        telem.peak_gpu_util_pct
                    ),

                    "telemetry_samples": (
                        telem.n_samples
                    ),

                    "telemetry_window_seconds": (
                        telem.sample_window_seconds
                    ),

                    "answer": gr.answer,
                }

                measured.append(
                    record
                )

                # Persist every run so a timeout does not
                # destroy completed measurements.
                pd.DataFrame(
                    measured
                ).to_csv(
                    output,
                    index=False,
                )

                if args.cooldown > 0:
                    time.sleep(
                        args.cooldown
                    )

            progress.update(1)

    progress.close()

    metadata = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(
            torch.cuda.get_device_capability(0)
        ),
        "requests": len(rows),
        "policies": len(policies),
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "telemetry_interval_s": (
            args.telemetry_interval
        ),
        "raw_rows": len(measured),
    }

    metadata_path = (
        output.parent
        / "hardware_profile_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print(
        "\nRAW ROWS:",
        len(measured),
    )

    expected = (
        len(rows)
        * len(policies)
        * args.repetitions
    )

    assert len(measured) == expected

    print(
        "REPEATED HARDWARE PROFILE: PASS"
    )


if __name__ == "__main__":
    main()
