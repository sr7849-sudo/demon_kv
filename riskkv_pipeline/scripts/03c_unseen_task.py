#!/usr/bin/env python3
"""Leave-one-task-out generalization: a required check against benchmark memorization."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from riskkv.predictor import build_pipeline
p=argparse.ArgumentParser(); p.add_argument('--input',default='results/oracle.csv'); p.add_argument('--output',default='results/unseen_task.csv'); a=p.parse_args()
df=pd.read_csv(a.input); df=df[df.policy!='full'].copy(); rows=[]
for task in sorted(df.task.dropna().unique()):
 tr=df[df.task!=task]; te=df[df.task==task]
 if tr.request_id.nunique()<10 or te.request_id.nunique()<5: continue
 m=build_pipeline(tr); m.fit(tr,tr.degradation); pred=np.clip(m.predict(te),0,None)
 rows.append({'heldout_task':task,'n_requests':te.request_id.nunique(),'mae':mean_absolute_error(te.degradation,pred),'r2':r2_score(te.degradation,pred)})
Path(a.output).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(a.output,index=False); print(pd.DataFrame(rows).to_string(index=False))
