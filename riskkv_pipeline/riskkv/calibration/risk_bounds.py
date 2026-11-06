from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta

def cp_upper(k,n,confidence=0.95):
    if n <= 0: return 1.0
    if k >= n: return 1.0
    return float(beta.ppf(confidence, k+1, n-k))

def fit_bins(p,y,n_bins=12,confidence=0.95):
    p=np.asarray(p,float); y=np.asarray(y,int)
    order=np.argsort(p)
    chunks=np.array_split(order,n_bins)
    bins=[]; running=0.0
    for idx in chunks:
        if len(idx)==0: continue
        lo=float(np.min(p[idx])); hi=float(np.max(p[idx]))
        k=int(y[idx].sum()); n=int(len(idx))
        ub=cp_upper(k,n,confidence)
        running=max(running,ub)
        bins.append({
            "lo":lo,"hi":hi,"n":n,"violations":k,
            "empirical":k/n,"upper_raw":ub,"upper_monotone":running,
        })
    return bins

def apply_bins(p,bins):
    p=np.asarray(p,float)
    out=np.ones_like(p)
    for i,x in enumerate(p):
        chosen=bins[-1]
        for b in bins:
            if x <= b["hi"]+1e-15:
                chosen=b; break
        out[i]=chosen["upper_monotone"]
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--oof", default="experiments/final_scaleup/analysis/risk_model/oof_predictions.csv")
    ap.add_argument("--out", default="experiments/final_scaleup/analysis/risk_model/calibrator.json")
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--bins", type=int, default=12)
    args=ap.parse_args()

    df=pd.read_csv(args.oof)
    avg=(df.groupby(["source_row","request_id","task","policy","budget","violation"],dropna=False)
           .risk_pred.mean().reset_index())
    bins=fit_bins(avg.risk_pred,avg.violation,args.bins,args.confidence)
    avg["risk_upper"]=apply_bins(avg.risk_pred,bins)
    avg.to_csv(Path(args.out).with_name("oof_calibrated.csv"),index=False)
    obj={"delta":args.delta,"confidence":args.confidence,"n_bins":args.bins,"bins":bins}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(obj,indent=2)+"\n")
    print(json.dumps(obj,indent=2))

if __name__=="__main__":
    main()
