from __future__ import annotations
import argparse, re, string, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from riskkv.common.io import TASKS, read_jsonl

def text_features(text: str) -> dict:
    text = str(text or "")
    chars = len(text)
    words = re.findall(r"\S+", text)
    alpha = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)
    punct = sum(c in string.punctuation for c in text)
    lines = text.count("\n") + 1 if text else 0
    return {
        "char_count": chars,
        "word_count": len(words),
        "line_count": lines,
        "digit_fraction": digits / max(1, chars),
        "alpha_fraction": alpha / max(1, chars),
        "punct_fraction": punct / max(1, chars),
        "question_marks": text.count("?"),
        "newline_fraction": text.count("\n") / max(1, chars),
    }

def make_rid(row, task, idx):
    rid = row.get("riskkv_request_id")
    if rid:
        return str(rid)
    raw = f"{task}|{idx}|{row.get('_id','')}|{row.get('input','')}|{row.get('context','')[:128]}"
    return f"{task}__{idx:06d}__{hashlib.sha256(raw.encode()).hexdigest()[:12]}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="experiments/final_scaleup/analysis/oracle_kvcf.csv")
    ap.add_argument("--data-root", default="data/final_common_longbench_filtered")
    ap.add_argument("--out", default="experiments/final_scaleup/analysis/features_kvcf.csv")
    args = ap.parse_args()

    oracle = pd.read_csv(args.oracle)
    reqs = []
    for task in TASKS:
        rows = read_jsonl(Path(args.data_root)/f"{task}.jsonl")
        for idx, r in enumerate(rows):
            rid = make_rid(r, task, idx)
            inp = str(r.get("input", ""))
            ctx = str(r.get("context", ""))
            prompt = str(r.get("prompt", "")) or (ctx + "\n" + inp)
            feat = {}
            feat.update({f"prompt_{k}": v for k,v in text_features(prompt).items()})
            feat.update({f"input_{k}": v for k,v in text_features(inp).items()})
            feat.update({f"context_{k}": v for k,v in text_features(ctx).items()})
            feat.update({
                "request_id": rid,
                "source_length": r.get("length", np.nan),
                "qwen_tokens": r.get("riskkv_qwen_tokens", np.nan),
                "mistral_tokens": r.get("riskkv_mistral_tokens", np.nan),
                "answer_count": len(r.get("answers", []) or []),
            })
            reqs.append(feat)
    rf = pd.DataFrame(reqs)
    if rf.request_id.duplicated().any():
        raise RuntimeError("Duplicate request_id in frozen dataset features")

    df = oracle.merge(rf, on="request_id", how="left", validate="many_to_one")
    if df["prompt_char_count"].isna().any():
        bad = df.loc[df.prompt_char_count.isna(), "request_id"].head().tolist()
        raise RuntimeError(f"Feature merge failed for request ids: {bad}")

    df["budget_num"] = pd.to_numeric(df["budget"], errors="coerce").fillna(df["prompt_tokens"])
    df["log_prompt_tokens"] = np.log1p(df["prompt_tokens"].astype(float))
    df["log_budget"] = np.log1p(df["budget_num"].astype(float))
    df["budget_fraction"] = np.minimum(1.0, df["budget_num"] / df["prompt_tokens"].clip(lower=1))
    df["compressed"] = (df["policy"] != "fullkv").astype(int)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"WROTE {args.out}: {len(df)} rows, {df.request_id.nunique()} requests")

if __name__ == "__main__":
    main()
