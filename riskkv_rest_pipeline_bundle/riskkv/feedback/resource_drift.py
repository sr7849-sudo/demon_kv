from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--selection",default="experiments/final_scaleup/analysis/selection_riskkv.csv")
    ap.add_argument("--resource-csv",required=True)
    ap.add_argument("--out",default="experiments/final_scaleup/analysis/resource_drift.csv")
    ap.add_argument("--latency-sla-ms",type=float,default=None)
    ap.add_argument("--memory-sla-mb",type=float,default=None)
    ap.add_argument("--alpha",type=float,default=.20,help="EWMA adaptation rate")
    ap.add_argument("--seeds",type=int,default=10)
    args=ap.parse_args()
    sel=pd.read_csv(args.selection)
    res=pd.read_csv(args.resource_csv)
    required={"request_id","policy","budget","latency_ms","peak_memory_mb"}
    missing=required-set(res.columns)
    if missing: raise RuntimeError(f"resource CSV missing {sorted(missing)}")
    keys=["request_id","policy","budget"]
    x=sel.merge(res,on=keys,how="left",validate="one_to_one")
    if x.latency_ms.isna().any():
        raise RuntimeError("Resource measurements do not cover every selected row")
    lat_sla=args.latency_sla_ms or float(np.quantile(x.latency_ms,.95)*1.10)
    mem_sla=args.memory_sla_mb or float(np.quantile(x.peak_memory_mb,.95)*1.10)
    phases=[("nominal",1.00),("moderate",1.25),("heavy",1.60),("recovery",1.00)]
    rows=[]
    for seed in range(args.seeds):
        rng=np.random.default_rng(9000+seed)
        idx=np.arange(len(x)); rng.shuffle(idx)
        stream=x.iloc[idx].reset_index(drop=True)
        ewma=1.0
        for phase,mult in phases:
            for _,r in stream.iterrows():
                noise=max(.7,rng.normal(1.0,.05))
                realized_mult=mult*noise
                lat=float(r.latency_ms)*realized_mult
                mem=float(r.peak_memory_mb)*(1+.35*(realized_mult-1))
                open_bad=(lat>lat_sla) or (mem>mem_sla)
                ewma=(1-args.alpha)*ewma+args.alpha*realized_mult
                pred_lat=float(r.latency_ms)*ewma
                pred_mem=float(r.peak_memory_mb)*(1+.35*(ewma-1))
                feedback_bad=(pred_lat>lat_sla) or (pred_mem>mem_sla)
                rows.append({
                    "seed":seed,"phase":phase,"request_id":r.request_id,
                    "realized_multiplier":realized_mult,"ewma_multiplier":ewma,
                    "latency_ms_realized":lat,"memory_mb_realized":mem,
                    "open_loop_sla_violation":int(open_bad),
                    "feedback_detected_pressure":int(feedback_bad),
                })
    out=pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(args.out,index=False)
    summary=(out.groupby("phase")
             .agg(open_loop_violation=("open_loop_sla_violation","mean"),
                  feedback_pressure_flag=("feedback_detected_pressure","mean"),
                  realized_multiplier=("realized_multiplier","mean"))
             .reset_index())
    summary.to_csv(Path(args.out).with_suffix(".summary.csv"),index=False)
    print(summary.to_string(index=False))
    print("NOTE: drift harness only; keep the live contention experiment as the paper result.")

if __name__=="__main__":
    main()
