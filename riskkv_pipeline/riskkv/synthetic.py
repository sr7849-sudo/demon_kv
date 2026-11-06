from __future__ import annotations
import numpy as np, pandas as pd

def make_synthetic(n_requests=500, seed=42):
    rng=np.random.default_rng(seed)
    policies=[("full",1.0,0.0),("snapkv_25",0.75,0.25),("snapkv_50",0.5,0.50),("snapkv_75",0.25,0.75),("quant4",0.25,0.65)]
    rows=[]
    for i in range(n_requests):
        length=int(rng.integers(2048,32768)); entropy=float(rng.normal(4.0,0.8)); kv=float(rng.normal(1.0,0.2)); nll=float(rng.normal(2.8,0.5)); latent=0.04*(entropy-4)+0.03*(nll-2.8)+0.02*(kv-1)+rng.normal(0,0.015)
        fullq=float(np.clip(rng.normal(0.80,0.08),0,1))
        task=["qa","summarization","reasoning"][i%3]
        for name,mem,aggr in policies:
            fam=0.8 if name=="quant4" else 1.0
            deg=max(0.0, aggr*(0.01+fam*max(0,latent))+rng.normal(0,0.004)) if name!="full" else 0.0
            q=max(0,fullq-deg)
            rows.append(dict(request_id=f"r{i:05d}",task=task,metric="f1",policy=name,memory_ratio=mem,context_tokens=length,last_logit_entropy=entropy,k_l2_mean=kv,prefill_nll=nll,quality=q,full_quality=fullq,degradation=deg))
    return pd.DataFrame(rows)
