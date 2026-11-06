from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--oracle",default="experiments/final_scaleup/analysis/oracle_kvcf.csv")
    ap.add_argument("--out",default="experiments/final_scaleup/manifests/resource_profile_plan.csv")
    ap.add_argument("--n",type=int,default=100)
    ap.add_argument("--repeats",type=int,default=3)
    ap.add_argument("--seed",type=int,default=20261011)
    args=ap.parse_args()
    df=pd.read_csv(args.oracle)
    comp=df[df.policy!="fullkv"].copy()
    rng=np.random.default_rng(args.seed)
    picks=[]
    groups=list(comp.groupby(["task","config"]))
    per=max(1,args.n//max(1,len(groups)))
    for _,g in groups:
        take=min(per,len(g))
        picks.extend(rng.choice(g.index,take,replace=False).tolist())
    picks=list(dict.fromkeys(picks))
    remaining=np.setdiff1d(comp.index.to_numpy(),np.array(picks,dtype=int))
    if len(picks)<args.n:
        picks.extend(rng.choice(remaining,min(args.n-len(picks),len(remaining)),replace=False).tolist())
    plan=comp.loc[picks[:args.n],["request_id","task","policy","budget","config","prediction_file"]].copy()
    plan["repeats"]=args.repeats
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    plan.to_csv(args.out,index=False)
    print(f"WROTE {args.out}: {len(plan)} cases × {args.repeats} repeats")
if __name__=="__main__":
    main()
