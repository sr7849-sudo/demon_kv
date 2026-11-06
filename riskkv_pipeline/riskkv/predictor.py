from __future__ import annotations
import json, pickle
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor

NON_FEATURE={"request_id","task","metric","reference","answer","quality","full_quality","degradation","actual_peak_gpu_bytes","actual_seconds"}

def feature_columns(df: pd.DataFrame):
    return [c for c in df.columns if c not in NON_FEATURE and c not in {"policy"}]

def build_pipeline(df: pd.DataFrame):
    numeric=[c for c in feature_columns(df) if pd.api.types.is_numeric_dtype(df[c])]
    categorical=["policy"]
    pre=ColumnTransformer([
        ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),
        ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),categorical),
    ], remainder="drop")
    reg=HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=42)
    return Pipeline([("pre",pre),("reg",reg)])

def save_model(path, model, metadata):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,"wb") as f: pickle.dump({"model":model,"metadata":metadata},f)

def load_model(path):
    with open(path,"rb") as f: return pickle.load(f)
