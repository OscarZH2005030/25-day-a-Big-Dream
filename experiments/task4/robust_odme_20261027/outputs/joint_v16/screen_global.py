"""Calibration-only feasibility of globally coupled, family-safe choices."""
import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
from pathlib import Path
import json,warnings
import pandas as pd,numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint,OptimizeWarning
O=Path(__file__).resolve().parent;t=pd.read_csv(O.parent/'joint_v15/calibration_candidates.csv');panels=sorted(t.panel.unique());names=sorted(t.candidate.unique());variables=[(p,n) for p in panels for n in names];families=sorted(t.scenario.unique());metrics=['S_od','S_link','S_dev','S_attr','synthetic_S_ODME'];lookup=t.set_index(['panel','scenario','candidate']);deltas={}
for p,n in variables:
 deltas[p,n]={s:{k:float(lookup.loc[p,s,n][k]-lookup.loc[p,s,'v11'][k]) for k in metrics} for s in families}
M=[];lb=[];ub=[]
for p in panels:M.append([float(pp==p) for pp,n in variables]);lb.append(1.);ub.append(1.)
# Each error family guards every component, not merely the grand average.
for s in families:
 for k in metrics:
  row=np.array([deltas[v][s][k]/len(panels) for v in variables]);row[abs(row)<1e-12]=0.;M.append(row);lb.append(-1e-10);ub.append(np.inf)
objective=np.array([-np.mean([deltas[v][s]['S_dev'] for s in families])/len(panels) for v in variables]);objective[abs(objective)<1e-12]=0.
with warnings.catch_warnings():
 warnings.simplefilter('ignore',OptimizeWarning);warnings.filterwarnings('ignore',message='Unrecognized options detected',category=RuntimeWarning)
 r=milp(objective,integrality=np.ones(len(variables)),bounds=Bounds(0,1),constraints=LinearConstraint(np.array(M),lb,ub),options={'time_limit':60.,'mip_rel_gap':1e-9,'threads':4})
assert r.success,r.message
policy={p:n for (p,n),x in zip(variables,r.x) if x>.5};assert len(policy)==len(panels);rows=[]
for s in families:
 rows.append({'scenario':s,**{'delta_'+k:sum(deltas[p,policy[p]][s][k] for p in panels)/len(panels) for k in metrics}})
pd.DataFrame(rows).to_csv(O/'calibration_global_family_deltas.csv',index=False);gain=float(np.mean([r['delta_S_dev'] for r in rows]));result={'policy':policy,'calibration_Sdev_gain':gain,'raw_MIP_objective_gain':float(-r.fun),'constraints':'for each of 16 error families, all four components plus total no worse on calibration; select one candidate per panel','calibration_only_screen':True,'count_CV_not_yet_checked':True,'hidden_truth_available':False,'MIP_optimal':True,'MIP_gap':float(r.mip_gap),'scope':'only this 26-candidate discrete grid, subject to numerical solver tolerances; not impossibility for all models','promoted':False};(O/'screen.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
