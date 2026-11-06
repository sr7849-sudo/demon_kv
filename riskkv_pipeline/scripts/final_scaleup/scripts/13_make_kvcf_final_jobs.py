#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
TASKS=["qasper","multifieldqa_en","hotpotqa","triviaqa","gov_report","multi_news","trec","passage_retrieval_en","passage_count"]
MODEL="mistralai/Mistral-7B-Instruct-v0.2"
METHODS=["snapkv","pyramidkv","adakv","l2norm"]
BUDGETS=[256,512,1024,2048,4096]
rows=[]; jid=0
for task in TASKS:
    rows.append(dict(job_id=jid,model=MODEL,task=task,method="FullKV",budget_tokens=-1)); jid+=1
for method in METHODS:
    for budget in BUDGETS:
        for task in TASKS:
            rows.append(dict(job_id=jid,model=MODEL,task=task,method=method,budget_tokens=budget)); jid+=1
out=Path("experiments/final_scaleup/manifests/kvcf_final_jobs.csv"); out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out,index=False)
print("jobs",len(rows),"max_job_id",jid-1,"file",out)
