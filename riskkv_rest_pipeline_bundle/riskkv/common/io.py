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
    candidates = []
    for p in done_dir.glob("*.json"):
        if p.name in {"run_meta.json", "provenance.json", "result.json", "summary.json"}:
            continue
        if p.stat().st_size > 0:
            candidates.append(p)
    if len(candidates) != 1:
        raise RuntimeError(
            f"{done_dir}: expected exactly one non-meta nonempty prediction JSON, "
            f"found {[p.name for p in candidates]}"
        )
    return candidates[0]

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
