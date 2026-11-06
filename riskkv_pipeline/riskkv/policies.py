from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PolicySpec:
    name: str
    family: str
    compression_ratio: float = 0.0
    memory_ratio: float = 1.0
    nbits: int | None = None
    backend: str | None = None

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d["name"],
            family=d["family"],
            compression_ratio=float(d.get("compression_ratio", 0.0)),
            memory_ratio=float(d.get("memory_ratio", 1.0)),
            nbits=d.get("nbits"),
            backend=d.get("backend"),
        )


def build_runtime_policy(
    spec: PolicySpec,
    *,
    model_config: Any | None = None,
):
    """
    Build the runtime objects for a cache policy.

    Returns
    -------
    (press, cache)
        press:
            KVPress compression press, or None.
        cache:
            Transformers cache object, or None.

    Notes
    -----
    QuantizedCache in Transformers >=5 requires the model configuration.
    """

    if spec.family == "full":
        return None, None

    if spec.family == "snapkv":
        from kvpress import SnapKVPress

        return (
            SnapKVPress(
                compression_ratio=float(spec.compression_ratio)
            ),
            None,
        )

    if spec.family == "knorm":
        from kvpress import KnormPress

        return (
            KnormPress(
                compression_ratio=float(spec.compression_ratio)
            ),
            None,
        )

    if spec.family == "streaming":
        from kvpress import StreamingLLMPress

        return (
            StreamingLLMPress(
                compression_ratio=float(spec.compression_ratio)
            ),
            None,
        )

    if spec.family == "quantized":
        from transformers import QuantizedCache

        if spec.nbits is None:
            raise ValueError(
                f"{spec.name}: nbits is required for quantized cache"
            )

        if model_config is None:
            raise ValueError(
                f"{spec.name}: model_config is required for "
                "Transformers QuantizedCache"
            )

        backend = spec.backend or "quanto"

        return (
            None,
            QuantizedCache(
                backend=backend,
                config=model_config,
                nbits=int(spec.nbits),
            ),
        )

    raise ValueError(
        f"Unknown policy family '{spec.family}'"
    )
