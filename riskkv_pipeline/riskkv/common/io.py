from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Any
import pandas as pd

TASKS = [
    "qasper", "multifieldqa_en", "hotpotqa", "triviaqa", "gov_report",
    "multi_news", "trec", "passage_retrieval_en", "passage_count",
]
POLICIES = ["fullkv", "snapkv", "pyramidkv", "adakv", "l2norm", "dynamickv"]

def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"{path}:{i}: invalid JSON: {e}") from e
    return rows

def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")

def infer_task(parts: Iterable[str]) -> str | None:
    for x in parts:
        if x in TASKS:
            return x
    return None

def infer_budget(parts: Iterable[str]) -> int | None:
    for x in parts:
        if x.startswith("budget_"):
            try:
                return int(x.split("_", 1)[1])
            except Exception:
                pass
    return None

def infer_policy(parts: Iterable[str], filename: str | None = None) -> str | None:
    lowers = [x.lower() for x in parts]
    for p in POLICIES:
        if p in lowers:
            return p
    if filename:
        stem = Path(filename).stem.lower()
        for p in POLICIES:
            if p in stem:
                return p
    return None



def choose_prediction_file(done_dir: Path) -> Path:
    """
    Locate the actual KVCF prediction artifact beneath one completed cell.

    KVCF writes predictions below:
        <save_dir>/<model>_<budget>/<task>/<method>.json

    while DONE/PROVENANCE.txt live directly in <save_dir>.
    Therefore discovery must be recursive.
    """
    ignore_names = {
        "run_meta.json",
        "provenance.json",
        "result.json",
        "summary.json",
        "DONE",
    }

    prediction_keys = {
        "pred",
        "prediction",
        "output",
        "response",
        "generated_text",
    }

    method_names = {
        "fullkv",
        "snapkv",
        "pyramidkv",
        "adakv",
        "l2norm",
        "knorm",
        "quant4",
        "dynamickv",
    }

    candidates = []

    for f in done_dir.rglob("*"):
        if not f.is_file():
            continue

        if f.name in ignore_names:
            continue

        if f.stat().st_size == 0:
            continue

        if f.suffix.lower() not in {".json", ".jsonl"}:
            continue

        candidates.append(f)

    # Strongest rule: filename itself is a known KV method.
    preferred = [
        f for f in candidates
        if f.stem.lower() in method_names
    ]

    if len(preferred) == 1:
        return preferred[0]

    if len(candidates) == 1:
        return candidates[0]

    # Structural fallback: parsed rows contain a generated prediction.
    structural = []

    for f in candidates:
        try:
            rows = read_prediction_rows(f)
        except Exception:
            continue

        if not rows:
            continue

        first = rows[0]

        if isinstance(first, dict) and prediction_keys.intersection(first.keys()):
            structural.append(f)

    if len(structural) == 1:
        return structural[0]

    raise RuntimeError(
        f"{done_dir}: could not uniquely identify prediction artifact. "
        f"candidates={[str(x.relative_to(done_dir)) for x in candidates]}, "
        f"preferred={[str(x.relative_to(done_dir)) for x in preferred]}, "
        f"structural={[str(x.relative_to(done_dir)) for x in structural]}"
    )


def read_prediction_rows(path: str | Path) -> list[dict]:
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise TypeError(f"{p}: JSON root must be list or JSONL")
        return data
    rows = []
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise TypeError(f"{p}:{i}: expected JSON object")
        rows.append(obj)
    return rows
