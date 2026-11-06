#!/usr/bin/env python3
import argparse, json, importlib
from riskkv.config import load_config, validate_config
from riskkv.hardware import gpu_metadata

p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); a=p.parse_args()
cfg=validate_config(load_config(a.config))
mods=['torch','transformers','kvpress','sklearn','pandas','numpy']
status={m:None for m in mods}
for m in mods:
    try: status[m]=getattr(importlib.import_module(m),'__version__','installed')
    except Exception as e: status[m]=f'ERROR: {e}'
print(json.dumps({'dependencies':status,'gpu':gpu_metadata(),'model':cfg['model']['name']},indent=2))
if any(str(v).startswith('ERROR') for v in status.values()): raise SystemExit(2)
