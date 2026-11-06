#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt
p=argparse.ArgumentParser(); p.add_argument('--oracle',default='results/oracle.csv'); p.add_argument('--pred',default='results/heldout_predictions.csv'); p.add_argument('--sweep',default='results/quality_sweep.csv'); p.add_argument('--outdir',default='figures'); a=p.parse_args(); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
if Path(a.oracle).exists():
 d=pd.read_csv(a.oracle); d=d[d.policy!='full']; policies=list(d.policy.unique()); vals=[d.loc[d.policy==x,'degradation'].values for x in policies]; plt.figure(figsize=(9,5)); plt.boxplot(vals,tick_labels=policies,showfliers=False); plt.ylabel('Per-request quality degradation'); plt.xticks(rotation=30,ha='right'); plt.tight_layout(); plt.savefig(out/'sensitivity_boxplot.png',dpi=220); plt.close()
if Path(a.pred).exists():
 d=pd.read_csv(a.pred); plt.figure(figsize=(6,6)); plt.scatter(d.degradation,d.predicted_degradation,s=10,alpha=.4); mx=max(d.degradation.max(),d.predicted_degradation.max(),.01); plt.plot([0,mx],[0,mx]); plt.xlabel('Actual degradation'); plt.ylabel('Predicted degradation'); plt.tight_layout(); plt.savefig(out/'prediction_calibration.png',dpi=220); plt.close()
if Path(a.sweep).exists():
 d=pd.read_csv(a.sweep); plt.figure(figsize=(7,5)); plt.plot(d.epsilon,d.mean_memory_ratio,marker='o'); plt.xlabel('Permitted quality degradation ε'); plt.ylabel('Mean KV memory ratio'); plt.tight_layout(); plt.savefig(out/'quality_memory_tradeoff.png',dpi=220); plt.close()
print(f'Figures written to {out}')
