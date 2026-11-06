from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import beta

def cp_upper(k,n,c=.95):
    if n<=0:return 1.0
    if k>=n:return 1.0
    return float(beta.ppf(c,k+1,n-k))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--selection",default="experiments/final_scaleup/analysis/selection_riskkv.csv")
    ap.add_argument("--out",default="experiments/final_scaleup/analysis/quality_audit_sim.csv")
    ap.add_argument("--delta",type=float,default=.10)
    ap.add_argument("--audit-rates",default="0,0.01,0.05,0.10,0.20")
    ap.add_argument("--seeds",type=int,default=20)
    ap.add_argument("--confidence",type=float,default=.95)
    args=ap.parse_args()
    x=pd.read_csv(args.selection).copy()
    x["violation"]=(x.degradation.astype(float)>args.delta).astype(int)
    if "risk_upper" not in x: raise RuntimeError("selection file needs risk_upper")
    rates=[float(z) for z in args.audit_rates.split(",")]
    rows=[]
    for seed in range(args.seeds):
        rng=np.random.default_rng(100000+seed)
        for rate in rates:
            n=int(round(rate*len(x)))
            for strategy in ["random","uncertainty_plus_random_floor"]:
                if n==0:
                    ix=np.array([],dtype=int)
                elif strategy=="random":
                    ix=rng.choice(len(x),n,replace=False)
                else:
                    n_u=int(round(.8*n)); n_r=n-n_u
                    center=float(np.median(x.risk_upper))
                    order=np.argsort(np.abs(x.risk_upper.to_numpy()-center))
                    u=order[:n_u]
                    remain=np.setdiff1d(np.arange(len(x)),u,assume_unique=False)
                    r=rng.choice(remain,n_r,replace=False) if n_r else np.array([],int)
                    ix=np.concatenate([u,r])
                k=int(x.iloc[ix].violation.sum()) if n else 0
                rows.append({
                    "seed":seed,"audit_rate":rate,"audit_n":n,"strategy":strategy,
                    "observed_violations":k,
                    "audit_violation_rate":(k/n if n else np.nan),
                    "audit_violation_upper95":cp_upper(k,n,args.confidence) if n else 1.0,
                    "true_stream_violation_rate":float(x.violation.mean()),
                })
    out=pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(args.out,index=False)
    print(out.groupby(["audit_rate","strategy"]).agg(
        upper_mean=("audit_violation_upper95","mean"),
        upper_std=("audit_violation_upper95","std"),
        n=("audit_n","first")).to_string())

if __name__=="__main__":
    main()
