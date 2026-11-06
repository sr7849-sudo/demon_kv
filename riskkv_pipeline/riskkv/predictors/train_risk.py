from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CATEGORICAL = ["task", "policy"]
NUMERIC = [
    "prompt_tokens", "budget_num", "budget_fraction", "retention_ratio_nominal",
    "nominal_saving", "log_prompt_tokens", "log_budget",
    "prompt_char_count", "prompt_word_count", "prompt_line_count",
    "prompt_digit_fraction", "prompt_alpha_fraction", "prompt_punct_fraction",
    "prompt_question_marks", "input_char_count", "input_word_count",
    "context_char_count", "context_word_count", "answer_count",
]

def ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def make_model(seed: int):
    prep = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", ohe()),
        ]), CATEGORICAL),
    ])
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=250, max_leaf_nodes=31,
        l2_regularization=1.0, min_samples_leaf=25,
        random_state=seed,
    )
    return Pipeline([("prep", prep), ("clf", clf)])

def ece(y, p, bins=15):
    y = np.asarray(y, int); p = np.asarray(p, float)
    edges = np.linspace(0,1,bins+1)
    out = 0.0
    for a,b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b if b < 1 else p <= b)
        if m.any():
            out += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(out)

def metrics(y,p):
    y=np.asarray(y,int); p=np.clip(np.asarray(p,float),1e-6,1-1e-6)
    ans = {
        "n": int(len(y)), "positive_rate": float(y.mean()),
        "brier": float(brier_score_loss(y,p)),
        "nll": float(log_loss(y,p,labels=[0,1])),
        "ece15": ece(y,p,15),
    }
    if len(np.unique(y)) == 2:
        ans["auroc"] = float(roc_auc_score(y,p))
        ans["auprc"] = float(average_precision_score(y,p))
    else:
        ans["auroc"] = None; ans["auprc"] = None
    return ans

def group_folds(groups, n_splits, seed):
    unique = np.array(sorted(pd.unique(groups)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    chunks = np.array_split(unique, n_splits)
    g = np.asarray(groups)
    for fold, test_groups in enumerate(chunks):
        te = np.isin(g, test_groups)
        tr = ~te
        yield fold, np.where(tr)[0], np.where(te)[0]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features", default="experiments/final_scaleup/analysis/features_kvcf.csv")
    ap.add_argument("--out-dir", default="experiments/final_scaleup/analysis/risk_model")
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--ensemble", type=int, default=10)
    args=ap.parse_args()

    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.features)
    work=df[df.policy!="fullkv"].copy().reset_index(drop=False).rename(columns={"index":"source_row"})
    work["violation"]=(work.degradation.astype(float)>args.delta).astype(int)
    missing=[c for c in CATEGORICAL+NUMERIC if c not in work]
    if missing: raise RuntimeError(f"Missing feature columns: {missing}")
    X=work[CATEGORICAL+NUMERIC]
    y=work.violation.to_numpy(int)
    groups=work.request_id.astype(str).to_numpy()

    oof=[]; fold_metrics=[]
    for rep in range(args.repeats):
        seed=20261011+rep*1009
        for fold,tr,te in group_folds(groups,args.folds,seed):
            model=make_model(seed+fold)
            model.fit(X.iloc[tr], y[tr])
            p=model.predict_proba(X.iloc[te])[:,1]
            fm=metrics(y[te],p); fm.update({"repeat":rep,"fold":fold})
            fold_metrics.append(fm)
            for pos,prob in zip(te,p):
                oof.append({
                    "source_row": int(work.iloc[pos].source_row),
                    "request_id": work.iloc[pos].request_id,
                    "task": work.iloc[pos].task,
                    "policy": work.iloc[pos].policy,
                    "budget": work.iloc[pos].budget,
                    "violation": int(y[pos]),
                    "risk_pred": float(prob),
                    "repeat": rep, "fold": fold,
                })
    oof_df=pd.DataFrame(oof)
    oof_df.to_csv(out/"oof_predictions.csv",index=False)

    rep_summary=[]
    for rep,g in oof_df.groupby("repeat"):
        m=metrics(g.violation,g.risk_pred); m["repeat"]=int(rep); rep_summary.append(m)
    summary={
        "delta":args.delta, "rows":int(len(work)),
        "requests":int(work.request_id.nunique()),
        "repeats":args.repeats, "folds":args.folds,
        "per_repeat":rep_summary,
    }
    for key in ["auroc","auprc","brier","nll","ece15"]:
        vals=[r[key] for r in rep_summary if r.get(key) is not None]
        summary[f"{key}_mean"]=float(np.mean(vals)) if vals else None
        summary[f"{key}_std"]=float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0
    (out/"cv_summary.json").write_text(json.dumps(summary,indent=2)+"\n")

    for j in range(args.ensemble):
        seed=55000+j*37
        model=make_model(seed)
        model.fit(X,y)
        dump(model,out/f"risk_model_{j:02d}.joblib")
    metadata={"categorical":CATEGORICAL,"numeric":NUMERIC,"delta":args.delta,"ensemble":args.ensemble}
    (out/"model_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
