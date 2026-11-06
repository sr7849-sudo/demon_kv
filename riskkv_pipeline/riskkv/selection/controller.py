from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
from riskkv.calibration.risk_bounds import apply_bins

def load_ensemble(model_dir):
    md=json.loads((Path(model_dir)/"model_metadata.json").read_text())
    models=[load(Path(model_dir)/f"risk_model_{j:02d}.joblib") for j in range(md["ensemble"])]
    cols=md["categorical"]+md["numeric"]
    return md,models,cols

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features", default="experiments/final_scaleup/analysis/features_kvcf.csv")
    ap.add_argument("--model-dir", default="experiments/final_scaleup/analysis/risk_model")
    ap.add_argument("--calibrator", default="experiments/final_scaleup/analysis/risk_model/calibrator.json")
    ap.add_argument("--out", default="experiments/final_scaleup/analysis/selection_riskkv.csv")
    ap.add_argument("--epsilon",type=float,default=0.05)
    ap.add_argument("--delta",type=float,default=0.10)
    ap.add_argument("--latency-max-ms",type=float,default=None)
    ap.add_argument("--memory-max-mb",type=float,default=None)
    args=ap.parse_args()

    df=pd.read_csv(args.features)
    md,models,cols=load_ensemble(args.model_dir)
    cal=json.loads(Path(args.calibrator).read_text())
    if abs(float(md["delta"])-args.delta)>1e-12 or abs(float(cal["delta"])-args.delta)>1e-12:
        raise RuntimeError("delta mismatch among model/calibrator/controller")

    comp=df.policy!="fullkv"
    X=df.loc[comp,cols]
    pred=np.mean([m.predict_proba(X)[:,1] for m in models],axis=0)
    df["risk_pred"]=0.0
    df.loc[comp,"risk_pred"]=pred
    df["risk_upper"]=0.0
    df.loc[comp,"risk_upper"]=apply_bins(pred,cal["bins"])

    feasible=df.risk_upper<=args.epsilon
    if args.latency_max_ms is not None:
        if "latency_ms" not in df: raise RuntimeError("latency_ms absent")
        feasible &= df.latency_ms<=args.latency_max_ms
    if args.memory_max_mb is not None:
        if "peak_memory_mb" not in df: raise RuntimeError("peak_memory_mb absent")
        feasible &= df.peak_memory_mb<=args.memory_max_mb
    df["feasible"]=feasible

    chosen=[]
    for rid,g in df.groupby("request_id",sort=False):
        f=g[g.feasible]
        if f.empty:
            f=g[g.policy=="fullkv"]
        f=f.sort_values(["nominal_saving","risk_upper","risk_pred"], ascending=[False,True,True])
        r=f.iloc[0].copy()
        r["true_violation"]=int(float(r.degradation)>args.delta)
        r["fallback"]=int(r.policy=="fullkv")
        chosen.append(r)
    out=pd.DataFrame(chosen)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(args.out,index=False)
    summary={
        "requests":int(len(out)),
        "delta":args.delta,"epsilon":args.epsilon,
        "violation_rate":float(out.true_violation.mean()),
        "mean_nominal_saving":float(out.nominal_saving.mean()),
        "fallback_rate":float(out.fallback.mean()),
        "mean_quality":float(out.quality.mean()),
        "selection_counts":out.config.value_counts().to_dict(),
    }
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
