from __future__ import annotations
import math, time
import numpy as np


def _tensor_stats(x):
    import torch
    x=x.detach().float()
    flat=x.reshape(-1)
    if flat.numel()==0: return {"mean":0.,"std":0.,"maxabs":0.,"l2":0.}
    return {"mean":float(flat.mean()),"std":float(flat.std(unbiased=False)),"maxabs":float(flat.abs().max()),"l2":float(torch.linalg.vector_norm(flat)/math.sqrt(flat.numel()))}


def _extract_cache_layers(pkv):
    # Supports legacy tuple cache and modern Cache objects exposing key_cache/value_cache.
    if pkv is None: return []
    if hasattr(pkv,"key_cache") and hasattr(pkv,"value_cache"):
        return list(zip(pkv.key_cache, pkv.value_cache))
    try:
        return [(layer[0],layer[1]) for layer in pkv]
    except Exception:
        return []


def extract_prefill_features(model, tokenizer, context: str, question: str, max_tokens: int=32768, attention_layers=None, attention_query_window: int=64):
    """One Full-KV prefill. Features intentionally avoid storing the full attention tensor by default.
    Attention entropy is estimated only if the model returns attentions; otherwise NaN columns are emitted.
    """
    import torch
    text=context + ("\n"+question if question else "")
    toks=tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    device=next(model.parameters()).device
    toks={k:v.to(device) for k,v in toks.items()}
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0=time.perf_counter()
    with torch.inference_mode():
        out=model(**toks, use_cache=True, output_hidden_states=False, output_attentions=False, return_dict=True)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    dt=time.perf_counter()-t0
    logits=out.logits.detach().float()
    probs=torch.softmax(logits[:,-1,:],dim=-1)
    logp=torch.log_softmax(logits[:,-1,:],dim=-1)
    entropy=float(-(probs*logp).sum())
    top2=torch.topk(probs,k=min(2,probs.shape[-1]),dim=-1).values
    margin=float((top2[:,0]-top2[:,1]).mean()) if top2.shape[-1]>1 else 0.
    feats={
        "context_tokens":int(toks["input_ids"].shape[1]),
        "prefill_seconds":float(dt),
        "last_logit_entropy":entropy,
        "last_logit_top1":float(top2[:,0].mean()),
        "last_logit_margin":margin,
    }
    layers=_extract_cache_layers(out.past_key_values)
    k_l2=[];v_l2=[];k_std=[];v_std=[];k_max=[];v_max=[]
    for li,(k,v) in enumerate(layers):
        ks=_tensor_stats(k); vs=_tensor_stats(v)
        k_l2.append(ks["l2"]);v_l2.append(vs["l2"]);k_std.append(ks["std"]);v_std.append(vs["std"]);k_max.append(ks["maxabs"]);v_max.append(vs["maxabs"])
    for name,arr in [("k_l2",k_l2),("v_l2",v_l2),("k_std",k_std),("v_std",v_std),("k_maxabs",k_max),("v_maxabs",v_max)]:
        a=np.asarray(arr,dtype=float)
        feats[f"{name}_mean"]=float(np.nanmean(a)) if len(a) else np.nan
        feats[f"{name}_std_layers"]=float(np.nanstd(a)) if len(a) else np.nan
        feats[f"{name}_max"]=float(np.nanmax(a)) if len(a) else np.nan
    # Cheap token-level NLL over prefill logits (excluding first token).
    ids=toks["input_ids"]
    if ids.shape[1]>1:
        shift_logits=logits[:,:-1,:]
        shift_labels=ids[:,1:]
        nll=torch.nn.functional.cross_entropy(shift_logits.reshape(-1,shift_logits.shape[-1]), shift_labels.reshape(-1), reduction="mean")
        feats["prefill_nll"]=float(nll); feats["prefill_perplexity_capped"]=float(torch.exp(torch.clamp(nll,max=20)))
    else:
        feats["prefill_nll"]=np.nan; feats["prefill_perplexity_capped"]=np.nan
    return feats
