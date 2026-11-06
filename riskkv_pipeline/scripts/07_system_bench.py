#!/usr/bin/env python3
"""Benchmark actual selected policies. Requires GPU/model access and a provisioning file with request_id/policy."""
import argparse, json, time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from riskkv.config import load_config, validate_config
from riskkv.data import load_requests
from riskkv.kv_runtime_policies import PolicySpec
from riskkv.backend import KVPressBackend
from riskkv.metrics import score
from riskkv.hardware import estimated_kv_bytes, gpu_metadata
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--selection',default='results/provisioning.csv'); p.add_argument('--output',default='results/system_bench.csv'); a=p.parse_args()
cfg=validate_config(load_config(a.config)); req={x['request_id']:x for x in load_requests(cfg['data']['input_jsonl'])}; sel=pd.read_csv(a.selection); specs={x['name']:PolicySpec.from_dict(x) for x in cfg['policies']}; m=cfg['model']; backend=KVPressBackend(m['name'],device=m.get('device','cuda:0'),dtype=m.get('dtype','auto'),device_map=m.get('device_map'))
rows=[]
for _,s in tqdm(sel.iterrows(),total=len(sel)):
    r=req[s.request_id]; pol=specs.get(s.policy,PolicySpec('full','full'))
    g=backend.generate(r['context'],r['question'],pol,max_new_tokens=m.get('max_new_tokens',64),do_sample=False)
    q=score(g.answer,r['reference'],r.get('metric','f1')); toks=len(backend.tokenizer(r['context'])['input_ids']); kv_full=estimated_kv_bytes(backend.model.config,toks,2)
    rows.append({'request_id':s.request_id,'policy':pol.name,'quality':q,'seconds':g.total_seconds,'peak_gpu_bytes':g.peak_gpu_bytes,'estimated_full_kv_bytes':kv_full,'estimated_policy_kv_bytes':int(kv_full*pol.memory_ratio),'tokens_per_second':m.get('max_new_tokens',64)/max(g.total_seconds,1e-9)})
Path(a.output).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(a.output,index=False); print(json.dumps(gpu_metadata(),indent=2))
