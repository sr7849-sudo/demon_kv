#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np, pandas as pd
ap=argparse.ArgumentParser()
ap.add_argument("--input",required=True); ap.add_argument("--metrics",default="memory_saved,violation,oracle_recovery")
ap.add_argument("--group-cols",default="delta,risk_target,method"); ap.add_argument("--resamples",type=int,default=2000)
ap.add_argument("--seed",type=int,default=2027); ap.add_argument("--output",required=True); a=ap.parse_args()
d=pd.read_csv(a.input); metrics=a.metrics.split(","); groups=a.group_cols.split(","); rng=np.random.default_rng(a.seed); out=[]
it=d.groupby(groups,dropna=False)
for key,g in it:
    if not isinstance(key,tuple): key=(key,)
    if g.request_id.duplicated().any(): raise SystemExit(f"duplicate request_id {key}")
    vals={m:g[m].to_numpy(float) for m in metrics}; n=len(g)
    boot={m:[] for m in metrics}
    for _ in range(a.resamples):
        idx=rng.integers(0,n,n)
        for m in metrics: boot[m].append(vals[m][idx].mean())
    base=dict(zip(groups,key))
    for m in metrics:
        z=np.asarray(boot[m]); out.append({**base,"metric":m,"estimate":vals[m].mean(),
        "ci95_low":np.quantile(z,.025),"ci95_high":np.quantile(z,.975),"n_requests":n,"B":a.resamples})
o=pd.DataFrame(out); Path(a.output).parent.mkdir(parents=True,exist_ok=True); o.to_csv(a.output,index=False); print(o)
