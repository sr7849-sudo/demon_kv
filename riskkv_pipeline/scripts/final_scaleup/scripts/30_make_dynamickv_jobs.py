import csv
from pathlib import Path
TASKS=["qasper","multifieldqa_en","hotpotqa","triviaqa","gov_report","multi_news","trec","passage_retrieval_en","passage_count"]
BUDGETS=[256,512,1024,2048,4096]
out=Path("experiments/final_scaleup/manifests/dynamickv_final_jobs.csv")
out.parent.mkdir(parents=True,exist_ok=True)
rows=[]; jid=0
for budget in BUDGETS:
    for task in TASKS:
        rows.append({"job_id":jid,"task":task,"budget":budget}); jid+=1
with out.open("w",newline="") as f:
    wr=csv.DictWriter(f,fieldnames=["job_id","task","budget"]); wr.writeheader(); wr.writerows(rows)
print(f"WROTE {out}: {len(rows)} cells")
