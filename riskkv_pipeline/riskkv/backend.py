from __future__ import annotations

import time
from dataclasses import dataclass

from .kv_runtime_policies import (
    PolicySpec,
    build_runtime_policy,
)


@dataclass
class GenerationResult:
    answer: str

    total_seconds: float

    peak_gpu_bytes: int
    allocated_gpu_bytes: int

    baseline_allocated_gpu_bytes: int
    baseline_reserved_gpu_bytes: int

    peak_reserved_gpu_bytes: int

    incremental_peak_allocated_gpu_bytes: int
    incremental_peak_reserved_gpu_bytes: int


class KVPressBackend:
    def __init__(
        self,
        model_name: str,
        device="cuda:0",
        dtype="auto",
        device_map=None,
        trust_remote_code=False,
    ):
        import kvpress  # noqa: F401
        from transformers import pipeline

        kwargs = {
            "model": model_name,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
        }

        if device_map:
            kwargs["device_map"] = device_map
        else:
            kwargs["device"] = device

        self.pipe = pipeline(
            "kv-press-text-generation",
            **kwargs,
        )

        self.model = self.pipe.model
        self.tokenizer = self.pipe.tokenizer
        self.device = device

    def generate(
        self,
        context: str,
        question: str,
        policy: PolicySpec,
        max_new_tokens=64,
        do_sample=False,
        temperature=0.0,
    ):
        import torch

        press, cache = build_runtime_policy(
            policy,
            model_config=self.model.config,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            baseline_alloc = int(
                torch.cuda.memory_allocated()
            )

            baseline_reserved = int(
                torch.cuda.memory_reserved()
            )

            torch.cuda.reset_peak_memory_stats()

        else:
            baseline_alloc = 0
            baseline_reserved = 0

        kwargs = {
            "question": question,
            "max_new_tokens": int(
                max_new_tokens
            ),
            "do_sample": bool(
                do_sample
            ),
        }

        if press is not None:
            kwargs["press"] = press

        if cache is not None:
            kwargs["cache"] = cache

        if (
            do_sample
            and temperature > 0
        ):
            kwargs["temperature"] = float(
                temperature
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        out = self.pipe(
            context,
            **kwargs,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        dt = (
            time.perf_counter()
            - t0
        )

        answer = (
            out["answer"]
            if (
                isinstance(out, dict)
                and "answer" in out
            )
            else str(out)
        )

        if torch.cuda.is_available():
            peak_alloc = int(
                torch.cuda.max_memory_allocated()
            )

            current_alloc = int(
                torch.cuda.memory_allocated()
            )

            peak_reserved = int(
                torch.cuda.max_memory_reserved()
            )

            incremental_alloc = max(
                0,
                peak_alloc
                - baseline_alloc,
            )

            incremental_reserved = max(
                0,
                peak_reserved
                - baseline_reserved,
            )

        else:
            peak_alloc = 0
            current_alloc = 0
            peak_reserved = 0
            incremental_alloc = 0
            incremental_reserved = 0

        return GenerationResult(
            answer=answer,

            total_seconds=dt,

            peak_gpu_bytes=peak_alloc,
            allocated_gpu_bytes=current_alloc,

            baseline_allocated_gpu_bytes=(
                baseline_alloc
            ),

            baseline_reserved_gpu_bytes=(
                baseline_reserved
            ),

            peak_reserved_gpu_bytes=(
                peak_reserved
            ),

            incremental_peak_allocated_gpu_bytes=(
                incremental_alloc
            ),

            incremental_peak_reserved_gpu_bytes=(
                incremental_reserved
            ),
        )
