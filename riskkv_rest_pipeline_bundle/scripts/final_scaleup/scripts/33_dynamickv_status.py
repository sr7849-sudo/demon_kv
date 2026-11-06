import csv
from pathlib import Path
man=Path("experiments/final_scaleup/manifests/dynamickv_final_jobs.csv")
root=Path("experiments/final_scaleup/results/dynamickv")
rows=list(csv.DictReader(man.open()))
miss=[]
for r in rows:
    d=root/f"budget_{r['budget']}"/r["task"]
    if not (d/"DONE").exists(): miss.append(int(r["job_id"]))
print(f"complete {len(rows)-len(miss)} / {len(rows)}")
print("missing", "NONE" if not miss else ",".join(map(str,miss)))
