#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
root=Path("experiments/final_scaleup")
d=pd.read_csv(root/"manifests/kvcf_final_jobs.csv")
done=[]
for _,r in d.iterrows():
    p=root/"results/kvcf"/str(r.method)/f"budget_{int(r.budget_tokens)}"/str(r.task)/"DONE"
    done.append(p.exists())
d["done"]=done
print("complete",sum(done),"/",len(done))
m=d.loc[~d.done,"job_id"].astype(int).tolist()
print("missing",",".join(map(str,m)) if m else "NONE")
