from pathlib import Path
import pandas as pd
from riskkv.common.io import TASKS, read_jsonl, read_prediction_rows
from riskkv.scoring.longbench_metrics import score_one

DATA=Path("data/final_common_longbench_filtered")
ROOT=Path("experiments/final_scaleup/results/dynamickv")
BASE=Path("experiments/final_scaleup/analysis/oracle_kvcf.csv")
OUT=Path("experiments/final_scaleup/analysis/oracle_dynamickv.csv")
base=pd.read_csv(BASE)
full=base[base.policy=="fullkv"][["request_id","task","original_index","quality","prompt_tokens"]].copy()
full=full.rename(columns={"quality":"quality_fullkv"})
records=[]
for budget in [256,512,1024,2048,4096]:
    for task in TASKS:
        d=ROOT/f"budget_{budget}"/task
        if not (d/"DONE").exists(): raise RuntimeError(f"missing {d}/DONE")
        pred=read_prediction_rows(d/"dynamickv.json")
        frozen=read_jsonl(DATA/f"{task}.jsonl")
        if len(pred)!=len(frozen): raise RuntimeError(f"{task}/{budget}: row mismatch")
        taskfull=full[full.task==task].sort_values("original_index")
        for i,(p,f) in enumerate(zip(pred,frozen)):
            rid=f.get("riskkv_request_id")
            if not rid:
                orig=int(f.get("riskkv_original_index",i))
                z=taskfull[taskfull.original_index==orig]
                if len(z)!=1: raise RuntimeError(f"cannot map {task} row {i}")
                rid=z.iloc[0].request_id
            answers=f.get("answers",p.get("answers",[]))
            ac=f.get("all_classes",p.get("all_classes",[]))
            q=score_one(task,p.get("pred",""),answers,ac)
            fr=full[full.request_id==rid]
            if len(fr)!=1: raise RuntimeError(f"full row missing {rid}")
            qf=float(fr.iloc[0].quality_fullkv); pt=int(fr.iloc[0].prompt_tokens)
            retained=min(pt,budget)
            records.append({
                "request_id":rid,"task":task,"policy":"dynamickv","budget":budget,
                "config":f"dynamickv_{budget}","prompt_tokens":pt,
                "actual_retained_tokens_nominal":retained,
                "retention_ratio_nominal":retained/max(1,pt),
                "nominal_saving":1-retained/max(1,pt),
                "quality":q,"quality_fullkv":qf,
                "degradation":max(0,qf-q),
                "prediction":p.get("pred",""),
            })
df=pd.DataFrame(records)
OUT.parent.mkdir(parents=True,exist_ok=True); df.to_csv(OUT,index=False)
print(f"WROTE {OUT}: {len(df)} rows")
