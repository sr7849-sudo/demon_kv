#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
from riskkv.selector import select_policy
p=argparse.ArgumentParser(); p.add_argument('--input',default='results/heldout_predictions.csv'); p.add_argument('--epsilons',default='0,0.01,0.02,0.05,0.10'); p.add_argument('--output',default='results/quality_sweep.csv'); a=p.parse_args()
df=pd.read_csv(a.input); base=df.groupby('request_id').first().reset_index(); full=base.copy(); full['policy']='full'; full['memory_ratio']=1.; full['predicted_degradation']=0.; full['degradation']=0.; allc=pd.concat([df,full],ignore_index=True,sort=False)
rows=[]
for eps in [float(x) for x in a.epsilons.split(',')]:
    ss=[]
    for _,g in allc.groupby('request_id'): ss.append(select_policy(g,eps))
    s=pd.DataFrame(ss); rows.append({'epsilon':eps,'mean_memory_ratio':s.memory_ratio.mean(),'violation_rate':(s.degradation>eps).mean(),'mean_quality_degradation':s.degradation.mean()})
Path(a.output).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(a.output,index=False); print(pd.DataFrame(rows).to_string(index=False))
