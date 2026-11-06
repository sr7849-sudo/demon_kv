from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg=yaml.safe_load(f)
    if not isinstance(cfg, dict): raise ValueError("Config must be a YAML mapping")
    return cfg


def validate_config(cfg):
    required=["model","data","policies","quality","predictor","system"]
    missing=[k for k in required if k not in cfg]
    if missing: raise ValueError(f"Missing config sections: {missing}")
    names=[p["name"] for p in cfg["policies"]]
    if len(names)!=len(set(names)): raise ValueError("Policy names must be unique")
    if "full" not in names: raise ValueError("A policy named 'full' is required as the quality baseline")
    eps=float(cfg["quality"]["epsilon"])
    if eps < 0: raise ValueError("quality.epsilon must be >= 0")
    return cfg
