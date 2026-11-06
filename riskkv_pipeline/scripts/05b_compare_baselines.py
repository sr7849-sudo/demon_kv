#!/usr/bin/env python3
"""Compare RiskKV with global-static, task-aware, length-aware, FullKV and oracle selectors."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from riskkv.selector import select_policy
p=argparse.ArgumentParser(); p.add_argument('--oracle-data',default='results/oracle.csv'); p.add_argument('--heldout-pred',default='results/heldout_predictions.csv'); p.add_argument('--epsilon',type=float,default=.02); p.add_argument('--output',default='results/selector_baselines.csv'); a=p.parse_args()
all_df=pd.read_csv(a.oracle_data); test=pd.read_csv(a.heldout_pred); test_ids=set(test.request_id.unique()); train=all_df[~all_df.request_id.isin(test_ids)].copy(); eval_df=all_df[all_df.request_id.isin(test_ids)].copy()
# Candidate actual records for evaluation.
full_actual=eval_df[eval_df.policy=='full'].copy()
non_eval=eval_df[eval_df.policy!='full'].copy()

def safe_best(g, eps):
 f=g[g.degradation<=eps]
 if f.empty:
  x=g[g.policy=='full']; return x.iloc[0] if len(x) else g.sort_values('degradation').iloc[0]
 return f.sort_values(['memory_ratio','degradation']).iloc[0]

# Learn global static choice on training by empirical violation <= epsilon, then lowest memory.
def choose_group_policy(g):
 stats=g.groupby('policy').agg(memory_ratio=('memory_ratio','first'),violation=('degradation',lambda x:(x>a.epsilon).mean()),mean_deg=('degradation','mean')).reset_index()
 feasible=stats[stats.violation<=.05]  # safety budget for selector calibration
 if len(feasible): return feasible.sort_values(['memory_ratio','mean_deg']).iloc[0].policy
 return 'full'
global_policy=choose_group_policy(train)
# Task map and length-bin map learned only from training.
task_map={t:choose_group_policy(g) for t,g in train.groupby('task')}
train=train.copy(); train['len_bin']=pd.qcut(train.context_tokens.rank(method='first'),4,labels=False,duplicates='drop')
# derive numeric cutpoints from unique request lengths
qs=np.quantile(train.context_tokens,[.25,.5,.75])
def lbin(x): return int(np.searchsorted(qs,x,side='right'))
len_map={b:choose_group_policy(g) for b,g in train.groupby('len_bin')}
# RiskKV predictions with injected full candidate
base=test.groupby('request_id').first().reset_index(); full_pred=base.copy(); full_pred['policy']='full'; full_pred['memory_ratio']=1.; full_pred['predicted_degradation']=0.; pred_all=pd.concat([test,full_pred],ignore_index=True,sort=False)
rows=[]
for rid,gact in eval_df.groupby('request_id'):
 first=gact.iloc[0]
 choices={
  'full':'full',
  'static':global_policy,
  'task_aware':task_map.get(first.task,global_policy),
  'length_aware':len_map.get(lbin(first.context_tokens),global_policy),
  'oracle':safe_best(gact,a.epsilon).policy,
  'riskkv':select_policy(pred_all[pred_all.request_id==rid],a.epsilon).policy,
 }
 for selector,pol in choices.items():
  hit=gact[gact.policy==pol]
  if hit.empty: continue
  r=hit.iloc[0]; rows.append({'request_id':rid,'selector':selector,'policy':pol,'memory_ratio':r.memory_ratio,'degradation':r.degradation,'violation':r.degradation>a.epsilon,'task':r.task})
res=pd.DataFrame(rows); summ=res.groupby('selector').agg(n=('request_id','size'),mean_memory_ratio=('memory_ratio','mean'),mean_degradation=('degradation','mean'),violation_rate=('violation','mean')).reset_index(); Path(a.output).parent.mkdir(parents=True,exist_ok=True); summ.to_csv(a.output,index=False); res.to_csv(Path(a.output).with_name('selector_baselines_per_request.csv'),index=False); print(summ.sort_values('mean_memory_ratio').to_string(index=False))
