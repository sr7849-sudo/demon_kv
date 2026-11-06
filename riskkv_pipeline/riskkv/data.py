from __future__ import annotations
from .utils import load_jsonl, stable_id

REQUIRED=("context","question","reference")

def load_requests(path: str):
    rows=load_jsonl(path)
    out=[]
    for i,r in enumerate(rows):
        for k in REQUIRED:
            if k not in r: raise ValueError(f"Row {i} missing '{k}'")
        r=dict(r)
        r.setdefault("request_id", stable_id(r["context"]+"\n"+r["question"]+str(i)))
        r.setdefault("task","unknown")
        r.setdefault("metric","f1")
        out.append(r)
    return out
