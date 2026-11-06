#!/usr/bin/env python3
from pathlib import Path
import pandas as pd, subprocess, sys
from riskkv.synthetic import make_synthetic
Path('results').mkdir(exist_ok=True); make_synthetic(300).to_csv('results/oracle.csv',index=False)
cmds=[
 [sys.executable,'scripts/02_characterize.py','--input','results/oracle.csv'],
 [sys.executable,'scripts/03_train_risk.py','--input','results/oracle.csv'],
 [sys.executable,'scripts/03b_feature_ablation.py'],
 [sys.executable,'scripts/03c_unseen_task.py'],
 [sys.executable,'scripts/04_evaluate_prediction.py'],
 [sys.executable,'scripts/05_provision.py','--epsilon','0.02'],
 [sys.executable,'scripts/05b_compare_baselines.py','--epsilon','0.02'],
 [sys.executable,'scripts/06_quality_sweep.py'],
 [sys.executable,'scripts/08_make_figures.py'],
]
for c in cmds:
 print('+',' '.join(c)); subprocess.run(c,check=True)
print('SYNTHETIC END-TO-END SMOKE TEST PASSED')
