import pandas as pd
from riskkv.selector import select_policy

def test_selects_lowest_memory_feasible():
 d=pd.DataFrame([{'policy':'full','predicted_degradation':0,'memory_ratio':1.0},{'policy':'a','predicted_degradation':.01,'memory_ratio':.5},{'policy':'b','predicted_degradation':.03,'memory_ratio':.25}])
 assert select_policy(d,.02).policy=='a'

def test_fallback_full():
 d=pd.DataFrame([{'policy':'full','predicted_degradation':0,'memory_ratio':1.0},{'policy':'a','predicted_degradation':.1,'memory_ratio':.5}])
 assert select_policy(d,0).policy=='full'
