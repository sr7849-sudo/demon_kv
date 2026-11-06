#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from riskkv.selector import select_policy

p=argparse.ArgumentParser(); p.add_argument('--input',default='results/heldout_predictions.csv'); p.add_argument('--epsilon',type=float,default=.02); p.add_argument('--output',default='results/provisioning.csv'); a=p.parse_args()
df=pd.read_csv(a.input)
# Add FullKV candidate for each request. It has known zero degradation.
base=df.groupby('request_id').first().reset_index(); full=base.copy(); full['policy']='full'; full['memory_ratio']=1.0; full['predicted_degradation']=0.0; full['degradation']=0.0
allc=pd.concat([df,full],ignore_index=True,sort=False)
selected=[]
for rid,g in allc.groupby('request_id'):
    row=select_policy(g,a.epsilon); selected.append(row)
sel=pd.DataFrame(selected)
sel['constraint_violation']=sel['degradation']>a.epsilon
# Retrospective oracle on same candidate set.
orows=[]
for rid,g in allc.groupby('request_id'):
    feasible=g[g.degradation<=a.epsilon]
    o=(feasible.sort_values(['memory_ratio','degradation']).iloc[0] if len(feasible) else g[g.policy=='full'].iloc[0])
    orows.append({'request_id':rid,'oracle_policy':o.policy,'oracle_memory_ratio':o.memory_ratio})
sel=sel.merge(pd.DataFrame(orows),on='request_id'); sel['oracle_gap_memory']=sel.memory_ratio-sel.oracle_memory_ratio
Path(a.output).parent.mkdir(parents=True,exist_ok=True); sel.to_csv(a.output,index=False)
print(json.dumps({'n':len(sel),'mean_memory_ratio':float(sel.memory_ratio.mean()),'violation_rate':float(sel.constraint_violation.mean()),'mean_oracle_gap':float(sel.oracle_gap_memory.mean())},indent=2))
