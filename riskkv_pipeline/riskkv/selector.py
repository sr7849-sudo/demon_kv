from __future__ import annotations
import pandas as pd

def select_policy(predictions: pd.DataFrame, epsilon: float, cost_col="memory_ratio", full_policy="full"):
    req={"policy","predicted_degradation",cost_col}
    miss=req-set(predictions.columns)
    if miss: raise ValueError(f"Missing selector columns: {miss}")
    feasible=predictions[predictions.predicted_degradation <= epsilon].copy()
    if feasible.empty:
        f=predictions[predictions.policy==full_policy]
        if f.empty: return predictions.sort_values("predicted_degradation").iloc[0]
        return f.iloc[0]
    return feasible.sort_values([cost_col,"predicted_degradation"],ascending=[True,True]).iloc[0]
