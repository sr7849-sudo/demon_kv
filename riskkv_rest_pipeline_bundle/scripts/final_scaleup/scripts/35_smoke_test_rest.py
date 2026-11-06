import numpy as np
from riskkv.scoring.longbench_metrics import score_one
from riskkv.calibration.risk_bounds import fit_bins,apply_bins

assert score_one("qasper","The cat",["cat"],[]) > 0
assert score_one("passage_retrieval_en","Paragraph 7",["Paragraph 7"],[]) == 1
assert score_one("passage_count","12",["12"],[]) == 1
p=np.array([.01,.05,.2,.8]); y=np.array([0,0,1,1])
b=fit_bins(p,y,2,.95)
u=apply_bins(p,b)
assert np.all(np.diff(u)>=-1e-12)
print("REST PIPELINE SMOKE: PASS")
