from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

from riskkv.common.io import (
    TASKS, read_jsonl, read_prediction_rows, infer_task, infer_budget,
    infer_policy, choose_prediction_file, write_json,
)
from riskkv.scoring.longbench_metrics import score_one

PRED_KEYS = ("pred", "prediction", "output", "response", "generated_text")

def get_pred(row):
    for k in PRED_KEYS:
        if k in row:
            return row[k]
    raise KeyError(f"Prediction row has none of keys {PRED_KEYS}; keys={sorted(row)}")

def request_id(row, task, idx):
    rid = row.get("riskkv_request_id")
    if rid:
        return str(rid)
    raw = f"{task}|{idx}|{row.get('_id','')}|{row.get('input','')}|{row.get('context','')[:128]}"
    return f"{task}__{idx:06d}__{hashlib.sha256(raw.encode()).hexdigest()[:12]}"

def align_rows(pred_rows, frozen_rows, task):
    if len(pred_rows) != len(frozen_rows):
        raise RuntimeError(
            f"{task}: output rows={len(pred_rows)} != frozen rows={len(frozen_rows)}"
        )
    frozen_rids = [r.get("riskkv_request_id") for r in frozen_rows]
    pred_rids = [r.get("riskkv_request_id") for r in pred_rows]
    if all(frozen_rids) and all(pred_rids) and len(set(pred_rids)) == len(pred_rids):
        fm = {str(r["riskkv_request_id"]): r for r in frozen_rows}
        try:
            return [(p, fm[str(p["riskkv_request_id"])]) for p in pred_rows]
        except KeyError as e:
            raise RuntimeError(f"{task}: output riskkv_request_id missing from frozen set: {e}")
    fids = [r.get("_id") for r in frozen_rows]
    pids = [r.get("_id") for r in pred_rows]
    if all(x is not None for x in fids+pids) and len(set(map(str,fids))) == len(fids):
        fm = {str(r["_id"]): r for r in frozen_rows}
        if set(map(str,pids)) == set(map(str,fids)):
            return [(p, fm[str(p["_id"])]) for p in pred_rows]
    pairs = list(zip(pred_rows, frozen_rows))
    mismatch = 0
    for p, f in pairs:
        for key in ("input", "context"):
            if key in p and key in f and str(p[key]) != str(f[key]):
                mismatch += 1
                break
    if mismatch:
        raise RuntimeError(f"{task}: row-order fallback found {mismatch} input/context mismatches")
    return pairs

def discover_done_dirs(results_root: Path):
    return sorted({p.parent for p in results_root.rglob("DONE")})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="experiments/final_scaleup/results/kvcf")
    ap.add_argument("--data-root", default="data/final_common_longbench_filtered")
    ap.add_argument("--out", default="experiments/final_scaleup/analysis/oracle_kvcf.csv")
    ap.add_argument("--summary", default="experiments/final_scaleup/analysis/oracle_kvcf_summary.json")
    ap.add_argument("--expected-configs", type=int, default=189)
    ap.add_argument("--expected-rows", type=int, default=27951)
    ap.add_argument("--model-token-field", default="auto",
                    choices=["auto", "riskkv_mistral_tokens", "riskkv_qwen_tokens"])
    args = ap.parse_args()

    results_root = Path(args.results_root)
    data_root = Path(args.data_root)
    done_dirs = discover_done_dirs(results_root)
    if len(done_dirs) != args.expected_configs:
        raise RuntimeError(f"Expected {args.expected_configs} DONE configs, found {len(done_dirs)}")

    frozen = {t: read_jsonl(data_root / f"{t}.jsonl") for t in TASKS}
    records = []
    config_rows = []

    for d in done_dirs:
        rel = d.relative_to(results_root)
        task = infer_task(rel.parts)
        policy = infer_policy(rel.parts)
        budget = infer_budget(rel.parts)
        pred_file = choose_prediction_file(d)
        if task is None:
            raise RuntimeError(f"Cannot infer task from {rel}")
        if policy is None:
            policy = infer_policy(rel.parts, pred_file.name)
        if policy is None:
            raise RuntimeError(f"Cannot infer policy from {rel}/{pred_file.name}")

        preds = read_prediction_rows(pred_file)
        pairs = align_rows(preds, frozen[task], task)
        config_rows.append(dict(task=task, policy=policy, budget=budget, n=len(pairs), file=str(pred_file)))

        for idx, (p, f) in enumerate(pairs):
            rid = request_id(f, task, idx)
            pred = get_pred(p)
            answers = f.get("answers", p.get("answers", []))
            all_classes = f.get("all_classes", p.get("all_classes", []))
            q = score_one(task, pred, answers, all_classes)

            if args.model_token_field == "auto":
                mtok = f.get("riskkv_mistral_tokens")
                qtok = f.get("riskkv_qwen_tokens")
                prompt_tokens = mtok if mtok is not None else qtok
                token_field = "riskkv_mistral_tokens" if mtok is not None else "riskkv_qwen_tokens"
            else:
                prompt_tokens = f.get(args.model_token_field)
                token_field = args.model_token_field
            if prompt_tokens is None:
                prompt_tokens = f.get("length")
                token_field = "length"
            if prompt_tokens is None:
                raise RuntimeError(f"{task}/{rid}: no usable prompt token count")
            prompt_tokens = int(prompt_tokens)

            if policy == "fullkv" or budget is None:
                retained = prompt_tokens
                nominal_saving = 0.0
            else:
                retained = min(prompt_tokens, int(budget))
                nominal_saving = 1.0 - retained / max(1, prompt_tokens)

            records.append({
                "request_id": rid,
                "task": task,
                "original_index": int(f.get("riskkv_original_index", idx)),
                "policy": policy,
                "budget": np.nan if budget is None else int(budget),
                "prompt_tokens": prompt_tokens,
                "prompt_token_field": token_field,
                "actual_retained_tokens_nominal": retained,
                "retention_ratio_nominal": retained / max(1, prompt_tokens),
                "nominal_saving": nominal_saving,
                "quality": q,
                "prediction": pred,
                "answers_json": json.dumps(answers, ensure_ascii=False),
                "all_classes_json": json.dumps(all_classes, ensure_ascii=False),
                "prediction_file": str(pred_file),
            })

    df = pd.DataFrame(records)
    if len(df) != args.expected_rows:
        raise RuntimeError(f"Expected {args.expected_rows} request-config rows, got {len(df)}")

    full = df[df.policy == "fullkv"][["request_id", "quality"]].copy()
    if len(full) != df.request_id.nunique():
        dup = full.request_id.value_counts()
        raise RuntimeError(
            f"Need exactly one FullKV row/request; full={len(full)} requests={df.request_id.nunique()} "
            f"bad={dup[dup!=1].head().to_dict()}"
        )
    full = full.rename(columns={"quality": "quality_fullkv"})
    df = df.merge(full, on="request_id", how="left", validate="many_to_one")
    df["degradation"] = np.maximum(0.0, df["quality_fullkv"] - df["quality"])
    df["quality_gain"] = np.maximum(0.0, df["quality"] - df["quality_fullkv"])
    df["config"] = df.apply(
        lambda r: r["policy"] if pd.isna(r["budget"]) else f'{r["policy"]}_{int(r["budget"])}',
        axis=1,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    cfg = pd.DataFrame(config_rows).sort_values(["policy","budget","task"], na_position="first")
    summary = {
        "rows": int(len(df)),
        "requests": int(df.request_id.nunique()),
        "tasks": df.task.value_counts().sort_index().to_dict(),
        "configs": int(df.config.nunique()),
        "config_task_cells": int(len(cfg)),
        "policy_counts_rows": df.policy.value_counts().sort_index().to_dict(),
        "fullkv_rows": int((df.policy=="fullkv").sum()),
        "mean_quality_by_config": df.groupby("config").quality.mean().sort_values(ascending=False).to_dict(),
        "mean_nominal_saving_by_config": df.groupby("config").nominal_saving.mean().to_dict(),
        "note": "nominal_saving is retained-token saving, not measured GPU-memory saving",
    }
    write_json(args.summary, summary)
    print(f"WROTE {out}")
    print(json.dumps(summary, indent=2)[:5000])

if __name__ == "__main__":
    main()
