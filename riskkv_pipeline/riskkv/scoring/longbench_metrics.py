from __future__ import annotations
import re, string
from collections import Counter

try:
    from rouge import Rouge
except Exception:
    Rouge = None

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)
    return white_space_fix(remove_articles(remove_punc(str(s).lower())))

def f1_score(prediction_tokens, ground_truth_tokens):
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / max(1, len(prediction_tokens))
    recall = num_same / max(1, len(ground_truth_tokens))
    return 2 * precision * recall / (precision + recall)

def qa_f1_score(prediction, ground_truth, **kwargs):
    return f1_score(
        normalize_answer(prediction).split(),
        normalize_answer(ground_truth).split(),
    )

def rouge_score(prediction, ground_truth, **kwargs):
    if Rouge is None:
        raise RuntimeError("Package `rouge` is required: pip install rouge")
    try:
        return float(Rouge().get_scores([str(prediction)], [str(ground_truth)], avg=True)["rouge-l"]["f"])
    except Exception:
        return 0.0

def classification_score(prediction, ground_truth, all_classes=None, **kwargs):
    prediction = str(prediction)
    all_classes = all_classes or []
    em = [c for c in all_classes if str(c) in prediction]
    em = [c for c in em if not any((str(c) in str(g) and c != g) for g in em)]
    if ground_truth in em:
        return 1.0 / len(em)
    return 0.0

def retrieval_score(prediction, ground_truth, **kwargs):
    matches = re.findall(r"Paragraph (\d+)", str(ground_truth))
    if not matches:
        return 0.0
    gt = matches[0]
    nums = re.findall(r"\d+", str(prediction))
    if not nums:
        return 0.0
    return sum(n == gt for n in nums) / len(nums)

def count_score(prediction, ground_truth, **kwargs):
    nums = re.findall(r"\d+", str(prediction))
    if not nums:
        return 0.0
    return sum(str(n) == str(ground_truth) for n in nums) / len(nums)

DATASET2METRIC = {
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "triviaqa": qa_f1_score,
    "gov_report": rouge_score,
    "multi_news": rouge_score,
    "trec": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
}
FIRST_LINE_TASKS = {"trec", "triviaqa"}

def score_one(task: str, prediction: str, answers, all_classes=None) -> float:
    if task not in DATASET2METRIC:
        raise KeyError(f"No metric for task {task}")
    pred = str(prediction)
    if task in FIRST_LINE_TASKS:
        pred = pred.lstrip("\n").split("\n")[0]
    if not isinstance(answers, list):
        answers = [answers]
    scores = [
        DATASET2METRIC[task](pred, gt, all_classes=all_classes or [])
        for gt in answers
    ]
    return float(max(scores) if scores else 0.0)
