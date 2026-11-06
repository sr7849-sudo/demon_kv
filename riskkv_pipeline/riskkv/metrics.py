from __future__ import annotations
import re, string
from collections import Counter


def normalize_answer(s: str) -> str:
    s=(s or "").lower()
    s="".join(ch for ch in s if ch not in string.punctuation)
    s=re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, ref: str) -> float:
    return float(normalize_answer(pred)==normalize_answer(ref))


def token_f1(pred: str, ref: str) -> float:
    p=normalize_answer(pred).split(); r=normalize_answer(ref).split()
    if not p and not r: return 1.0
    if not p or not r: return 0.0
    common=Counter(p) & Counter(r)
    same=sum(common.values())
    if same==0: return 0.0
    precision=same/len(p); recall=same/len(r)
    return 2*precision*recall/(precision+recall)


def rouge_l(pred: str, ref: str) -> float:
    try:
        from rouge_score import rouge_scorer
        scorer=rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return float(scorer.score(ref or "", pred or "")["rougeL"].fmeasure)
    except ImportError:
        # exact DP fallback, suitable for short references
        a=normalize_answer(pred).split(); b=normalize_answer(ref).split()
        if not a or not b: return float(a==b)
        prev=[0]*(len(b)+1)
        for x in a:
            cur=[0]
            for j,y in enumerate(b,1): cur.append(prev[j-1]+1 if x==y else max(prev[j],cur[-1]))
            prev=cur
        lcs=prev[-1]; pr=lcs/len(a); rc=lcs/len(b)
        return 0.0 if lcs==0 else 2*pr*rc/(pr+rc)

def contains(pred: str, ref: str) -> float:
    """Return 1 when the normalized reference appears inside the normalized prediction."""
    p = normalize_answer(pred)
    r = normalize_answer(ref)
    if not r:
        return float(not p)
    return float(r in p)


METRICS={
    "exact_match": exact_match,
    "f1": token_f1,
    "token_f1": token_f1,
    "rouge_l": rouge_l,
    "contains": contains,
}

def score(pred: str, ref: str, metric: str) -> float:
    if metric not in METRICS: raise KeyError(f"Unsupported metric '{metric}'. Supported={list(METRICS)}")
    return METRICS[metric](pred,ref)
