#!/usr/bin/env python3
"""Create one deterministic Qasper example from the final common LongBench set."""
import argparse, json
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument("--input", default="data/final_common_longbench/qasper.jsonl")
ap.add_argument("--output", default="experiments/final_scaleup/smoke/data/qasper_1.jsonl")
a=ap.parse_args()

src=Path(a.input)
if not src.exists():
    raise SystemExit(f"Missing {src}")
rows=[json.loads(x) for x in src.read_text().splitlines() if x.strip()]
if not rows:
    raise SystemExit("No rows")
out=Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rows[0], ensure_ascii=False)+"\n")
print("source_rows", len(rows))
print("request_id", rows[0].get("riskkv_request_id","N/A"))
print("SMOKE DATA: PASS")
