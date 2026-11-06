#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from riskkv.predictor import build_pipeline, save_model

p=argparse.ArgumentParser(); p.add_argument('--input',default='results/oracle.csv'); p.add_argument('--model-out',default='results/risk_model.pkl'); p.add_argument('--pred-out',default='results/heldout_predictions.csv'); p.add_argument('--test-size',type=float,default=.2); a=p.parse_args()
df=pd.read_csv(a.input); df=df[df.policy!='full'].copy()
gss=GroupShuffleSplit(n_splits=1,test_size=a.test_size,random_state=42); tr,te=next(gss.split(df,groups=df.request_id))
train,test=df.iloc[tr].copy(),df.iloc[te].copy(); model=build_pipeline(train); model.fit(train,train.degradation)
pred=np.clip(model.predict(test),0,None); test['predicted_degradation']=pred
metrics={'mae':float(mean_absolute_error(test.degradation,pred)),'rmse':float(mean_squared_error(test.degradation,pred)**0.5),'r2':float(r2_score(test.degradation,pred)),'n_train_requests':int(train.request_id.nunique()),'n_test_requests':int(test.request_id.nunique())}
save_model(a.model_out,model,{'metrics':metrics}); Path(a.pred_out).parent.mkdir(parents=True,exist_ok=True); test.to_csv(a.pred_out,index=False); print(json.dumps(metrics,indent=2))
