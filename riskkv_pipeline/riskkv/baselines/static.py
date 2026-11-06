from __future__ import annotations
import pandas as pd

def static_table(df, delta):
    x=df.copy()
    x["violation"]=(x.degradation.astype(float)>delta).astype(int)
    return (x.groupby("config")
             .agg(violation_rate=("violation","mean"),
                  mean_saving=("nominal_saving","mean"),
                  mean_quality=("quality","mean"),
                  n=("request_id","nunique"))
             .reset_index()
             .sort_values(["violation_rate","mean_saving"],ascending=[True,False]))

def best_static_under_risk(df, delta, epsilon):
    tab=static_table(df,delta)
    feasible=tab[tab.violation_rate<=epsilon]
    if feasible.empty:
        return tab[tab.config=="fullkv"].iloc[0]
    return feasible.sort_values("mean_saving",ascending=False).iloc[0]

def task_static_selections(df,delta,epsilon):
    rows=[]
    for task,g in df.groupby("task"):
        best=best_static_under_risk(g,delta,epsilon)
        cfg=best["config"]
        rows.append(g[g.config==cfg])
    return pd.concat(rows,ignore_index=True)

def oracle_selections(df,delta):
    rows=[]
    for rid,g in df.groupby("request_id",sort=False):
        safe=g[g.degradation.astype(float)<=delta]
        if safe.empty: safe=g[g.policy=="fullkv"]
        rows.append(safe.sort_values(["nominal_saving","quality"],ascending=[False,False]).iloc[0])
    return pd.DataFrame(rows)
