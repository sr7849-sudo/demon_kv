from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor

CAT=["task","policy"]
NUM=[
    "prompt_tokens","budget_num","budget_fraction","retention_ratio_nominal","nominal_saving",
    "log_prompt_tokens","log_budget","prompt_char_count","prompt_word_count",
    "prompt_line_count","prompt_digit_fraction","prompt_punct_fraction",
    "input_char_count","input_word_count","context_char_count","context_word_count",
]
def ohe():
    try: return OneHotEncoder(handle_unknown="ignore",sparse_output=False)
    except TypeError: return OneHotEncoder(handle_unknown="ignore",sparse=False)
def model(seed=42):
    prep=ColumnTransformer([
        ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),NUM),
        ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",ohe())]),CAT),
    ])
    reg=HistGradientBoostingRegressor(learning_rate=.05,max_iter=250,max_leaf_nodes=31,
                                      l2_regularization=1.0,min_samples_leaf=25,random_state=seed)
    return Pipeline([("prep",prep),("reg",reg)])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features",default="experiments/final_scaleup/analysis/features_kvcf.csv")
    ap.add_argument("--out",default="experiments/final_scaleup/analysis/selection_metakv_style.csv")
    ap.add_argument("--delta",type=float,default=.10)
    ap.add_argument("--folds",type=int,default=5)
    args=ap.parse_args()
    df=pd.read_csv(args.features).reset_index(drop=False).rename(columns={"index":"source_row"})
    groups=np.array(sorted(df.request_id.unique()))
    rng=np.random.default_rng(260907); rng.shuffle(groups)
    chunks=np.array_split(groups,args.folds)
    pred=np.full(len(df),np.nan)
    for fold,tg in enumerate(chunks):
        te=df.request_id.isin(tg).to_numpy(); tr=~te
        m=model(260907+fold)
        m.fit(df.loc[tr,CAT+NUM],df.loc[tr,"quality"])
        pred[te]=m.predict(df.loc[te,CAT+NUM])
    df["quality_pred_openloop"]=np.clip(pred,0,1)
    fq=(df[df.policy=="fullkv"][["request_id","quality_pred_openloop"]]
        .rename(columns={"quality_pred_openloop":"quality_full_pred"}))
    df=df.merge(fq,on="request_id",how="left",validate="many_to_one")
    df["pred_degradation"]=np.maximum(0,df.quality_full_pred-df.quality_pred_openloop)
    chosen=[]
    for rid,g in df.groupby("request_id",sort=False):
        feasible=g[g.pred_degradation<=args.delta]
        if feasible.empty: feasible=g[g.policy=="fullkv"]
        chosen.append(feasible.sort_values(["nominal_saving","quality_pred_openloop"],
                                           ascending=[False,False]).iloc[0])
    out=pd.DataFrame(chosen)
    out["true_violation"]=(out.degradation>args.delta).astype(int)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(args.out,index=False)
    summary={
        "label":"matched-candidate MetaKV-style open-loop quality predictor",
        "requests":int(len(out)),"delta":args.delta,
        "violation_rate":float(out.true_violation.mean()),
        "mean_nominal_saving":float(out.nominal_saving.mean()),
        "mean_quality":float(out.quality.mean()),
    }
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
