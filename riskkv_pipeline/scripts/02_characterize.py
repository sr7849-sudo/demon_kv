#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

p=argparse.ArgumentParser(); p.add_argument('--input',default='results/oracle.csv'); p.add_argument('--outdir',default='results/characterization'); p.add_argument('--unsafe-threshold',type=float,default=0.05); a=p.parse_args()
df=pd.read_csv(a.input); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
non=df[df.policy!='full'].copy()
summary=non.groupby(['task','policy']).agg(n=('degradation','size'),mean_degradation=('degradation','mean'),median_degradation=('degradation','median'),std_degradation=('degradation','std'),p90_degradation=('degradation',lambda x: np.quantile(x,.9)),unsafe_rate=('degradation',lambda x: float((x>a.unsafe_threshold).mean()))).reset_index()
summary.to_csv(out/'sensitivity_summary.csv',index=False)
# Equal-length-bin control.
non['length_bin']=pd.qcut(non['context_tokens'].rank(method='first'),q=min(6,non.request_id.nunique()),duplicates='drop')
ctrl=non.groupby(['task','policy','length_bin'],observed=True).agg(n=('degradation','size'),std_degradation=('degradation','std'),iqr_degradation=('degradation',lambda x: np.quantile(x,.75)-np.quantile(x,.25))).reset_index()
ctrl.to_csv(out/'length_control.csv',index=False)
# Counterexample pairs: nearly equal length, maximal degradation gap.
pairs=[]
for (task,pol),g in non.groupby(['task','policy']):
    g=g.sort_values('context_tokens').reset_index(drop=True)
    for i in range(len(g)-1):
        for j in range(i+1,min(i+15,len(g))):
            l1,l2=g.loc[i,'context_tokens'],g.loc[j,'context_tokens']
            if abs(l1-l2)/max(l1,l2) <= .05:
                gap=abs(g.loc[i,'degradation']-g.loc[j,'degradation'])
                pairs.append({'task':task,'policy':pol,'request_a':g.loc[i,'request_id'],'request_b':g.loc[j,'request_id'],'length_a':l1,'length_b':l2,'deg_a':g.loc[i,'degradation'],'deg_b':g.loc[j,'degradation'],'gap':gap})
pd.DataFrame(pairs).sort_values('gap',ascending=False).head(100).to_csv(out/'counterexample_pairs.csv',index=False) if pairs else pd.DataFrame().to_csv(out/'counterexample_pairs.csv',index=False)
print(summary.to_string(index=False))
