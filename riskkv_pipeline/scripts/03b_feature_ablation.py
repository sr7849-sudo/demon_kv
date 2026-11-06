#!/usr/bin/env python3
"""Feature-group ablation on the same held-out request split."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

p=argparse.ArgumentParser(); p.add_argument('--input',default='results/oracle.csv'); p.add_argument('--output',default='results/feature_ablation.csv'); p.add_argument('--test-size',type=float,default=.2); a=p.parse_args()
df=pd.read_csv(a.input); df=df[df.policy!='full'].copy(); tr,te=next(GroupShuffleSplit(n_splits=1,test_size=a.test_size,random_state=42).split(df,groups=df.request_id)); train,test=df.iloc[tr],df.iloc[te]
groups={
 'length':['context_tokens'],
 'uncertainty':[c for c in df.columns if c.startswith('last_logit_') or c.startswith('prefill_nll') or c.startswith('prefill_perplexity')],
 'kv_stats':[c for c in df.columns if c.startswith('k_') or c.startswith('v_')],
}
groups['all']=sorted(set(sum(groups.values(),[])))
rows=[]
for name,cols in groups.items():
 cols=[c for c in cols if c in df.columns]
 numeric=cols
 pre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),('pol',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['policy'])])
 model=Pipeline([('pre',pre),('reg',HistGradientBoostingRegressor(max_iter=250,learning_rate=.05,random_state=42))]); model.fit(train,train.degradation); pred=np.clip(model.predict(test),0,None)
 rows.append({'feature_group':name,'n_features':len(cols),'mae':mean_absolute_error(test.degradation,pred),'r2':r2_score(test.degradation,pred)})
out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out,index=False); print(pd.DataFrame(rows).to_string(index=False))
