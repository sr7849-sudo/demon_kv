from __future__ import annotations
import time

def gpu_metadata():
    try:
        import torch
        if not torch.cuda.is_available(): return {"cuda":False}
        p=torch.cuda.get_device_properties(0)
        return {"cuda":True,"gpu_name":p.name,"total_memory_bytes":int(p.total_memory),"capability":f"{p.major}.{p.minor}","torch_cuda":torch.version.cuda}
    except Exception as e: return {"cuda":False,"error":str(e)}


def estimated_kv_bytes(model_config, tokens: int, bytes_per_element=2):
    layers=int(getattr(model_config,"num_hidden_layers"))
    kv_heads=int(getattr(model_config,"num_key_value_heads",getattr(model_config,"num_attention_heads")))
    hidden=int(getattr(model_config,"hidden_size")); heads=int(getattr(model_config,"num_attention_heads"))
    head_dim=hidden//heads
    return int(2*layers*kv_heads*head_dim*tokens*bytes_per_element)
