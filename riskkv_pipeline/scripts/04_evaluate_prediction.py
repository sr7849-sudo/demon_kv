#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, mean_squared_error

p=argparse.ArgumentParser(); p.add_argument('--input',default='results/heldout_predictions.csv'); p.add_argument('--unsafe-threshold',type=float,default=.05); p.add_argument('--output',default='results/prediction_metrics.json'); a=p.parse_args()
df=pd.read_csv(a.input); y=(df.degradation>a.unsafe_threshold).astype(int); s=df.predicted_degradation.values
safe_pred=s<=a.unsafe_threshold; false_safe=float(((safe_pred)&(y==1)).sum()/max(1,(y==1).sum()))
res={'mae':float(mean_absolute_error(df.degradation,s)),'rmse':float(mean_squared_error(df.degradation,s)**.5),'unsafe_rate':float(y.mean()),'false_safe_rate_among_unsafe':false_safe}
if y.nunique()>1:
    res['auroc']=float(roc_auc_score(y,s)); res['auprc']=float(average_precision_score(y,s))
Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(res,indent=2)); print(json.dumps(res,indent=2))
