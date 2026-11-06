from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from riskkv.baselines.static import static_table,best_static_under_risk,task_static_selections,oracle_selections

def summary(sel,delta):
    v=(sel.degradation.astype(float)>delta).astype(int)
    return {
        "n":int(len(sel)),
        "violation_rate":float(v.mean()),
        "mean_saving":float(sel.nominal_saving.mean()),
        "mean_quality":float(sel.quality.mean()),
        "fallback_rate":float((sel.policy=="fullkv").mean()),
    }

def boot_ci(sel,delta,B=2000,seed=20261011):
    rng=np.random.default_rng(seed)
    n=len(sel)
    v=(sel.degradation.to_numpy(float)>delta).astype(float)
    s=sel.nominal_saving.to_numpy(float)
    q=sel.quality.to_numpy(float)
    vals=[]
    for _ in range(B):
        ix=rng.integers(0,n,n)
        vals.append((v[ix].mean(),s[ix].mean(),q[ix].mean()))
    a=np.array(vals)
    def ci(j): return [float(np.quantile(a[:,j],.025)),float(np.quantile(a[:,j],.975))]
    return {"violation_rate_ci95":ci(0),"mean_saving_ci95":ci(1),"mean_quality_ci95":ci(2)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--oracle",default="experiments/final_scaleup/analysis/oracle_kvcf.csv")
    ap.add_argument("--riskkv",default="experiments/final_scaleup/analysis/selection_riskkv.csv")
    ap.add_argument("--out",default="experiments/final_scaleup/analysis/controller_report.json")
    ap.add_argument("--delta",type=float,default=.10)
    ap.add_argument("--epsilon",type=float,default=.05)
    ap.add_argument("--bootstrap",type=int,default=2000)
    args=ap.parse_args()
    df=pd.read_csv(args.oracle); rk=pd.read_csv(args.riskkv)
    best=best_static_under_risk(df,args.delta,args.epsilon)
    static=df[df.config==best["config"]].copy()
    task=task_static_selections(df,args.delta,args.epsilon)
    oracle=oracle_selections(df,args.delta)
    out={
        "delta":args.delta,"epsilon":args.epsilon,
        "riskkv":{**summary(rk,args.delta),**boot_ci(rk,args.delta,args.bootstrap)},
        "static":{**summary(static,args.delta),"config":best["config"]},
        "task_static":summary(task,args.delta),
        "oracle":summary(oracle,args.delta),
    }
    denom=max(1e-12,out["oracle"]["mean_saving"])
    out["riskkv"]["oracle_recovery_mean_saving"]=out["riskkv"]["mean_saving"]/denom
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(out,indent=2)+"\n")
    static_table(df,args.delta).to_csv(Path(args.out).with_name("static_config_table.csv"),index=False)
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
