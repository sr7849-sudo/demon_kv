from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / "third_party" / "longbench" / "dataset2prompt.json"
METRICS_PATH = ROOT / "third_party" / "longbench" / "metrics.py"


with PROMPT_PATH.open("r", encoding="utf-8") as f:
    DATASET2PROMPT = json.load(f)


def _load_official_metrics():
    spec = importlib.util.spec_from_file_location(
        "longbench_official_metrics",
        METRICS_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load official LongBench metrics.py")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_M = _load_official_metrics()


DATASET2METRIC = {
    "qasper": _M.qa_f1_score,
    "hotpotqa": _M.qa_f1_score,
    "multifieldqa_en": _M.qa_f1_score,
    "trec": _M.classification_score,
    "passage_retrieval_en": _M.retrieval_score,
    "triviaqa": _M.qa_f1_score,
}


FIRST_LINE_DATASETS = {
    "trec",
}


def build_longbench_prompt(
    dataset: str,
    context: str,
    user_input: str,
) -> str:
    if dataset not in DATASET2PROMPT:
        raise KeyError(f"No LongBench prompt template for {dataset!r}")

    return DATASET2PROMPT[dataset].format(
        context=context,
        input=user_input,
    )


def score_longbench(
    dataset: str,
    prediction: str,
    references: Sequence[str],
    all_classes=None,
) -> float:
    if dataset not in DATASET2METRIC:
        raise KeyError(
            f"No LongBench metric for {dataset!r}. "
            f"Supported={sorted(DATASET2METRIC)}"
        )

    pred = prediction

    # Matches official LongBench evaluation behavior.
    if dataset in FIRST_LINE_DATASETS:
        pred = pred.lstrip("\n").split("\n")[0]

    metric = DATASET2METRIC[dataset]

    best = 0.0

    for ref in references:
        kwargs = {}
        if dataset == "trec":
            kwargs["all_classes"] = all_classes

        value = metric(
            pred,
            ref,
            **kwargs,
        )

        best = max(best, float(value))

    return best
