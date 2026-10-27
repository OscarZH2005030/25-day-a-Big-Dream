# 中文说明：记录真实公开计数可行域中的S_dev条件性界限，而不是生成隐藏标签。
from pathlib import Path
import sys,time,json
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O));from feasible_bounds import REL,load,load_operator,bounds
sys.path.insert(0,str(O.parent/'scale_shape_v8'));from run_scale_shape import predict
import numpy as np,pandas as pd
start=time.time();rows=[]
for p in sorted((REL/'task4').iterdir()):
 ids,ls,D,paths=load_operator(REL/'corridors'/p.name/'network');A,c,b,t,_=load(p.name,'train',ids,ls,D);sols,_=predict(A,c,b)
 for tol in [0.,.01,.05]:
  L,U,maxf=bounds(A,c,b,tol)
  for name,x in sols.items():
   dev=float(abs(x-b).sum());bound=float(np.exp(-max(abs(dev/L-1),abs(dev/U-1)))) if L>1e-9 else 0.
   rows.append({'panel':p.name,'count_tolerance':tol,'candidate':name,'D_lower':L,'D_safe_upper':U,'Dhat':dev,'conditional_Sdev_lower_bound':bound,'minimax_target_D':2*L*U/(L+U) if L+U else 0,'max_total_path_flow':maxf})
 pd.DataFrame(rows).to_csv(O/'actual_count_feasible_bounds.csv',index=False);print('bounded',p.name,flush=True)
pd.DataFrame(rows).groupby(['count_tolerance','candidate']).conditional_Sdev_lower_bound.mean().to_csv(O/'feasible_bound_summary.csv');(O/'bounds_manifest.json').write_text(json.dumps({'seconds':time.time()-start,'panels':10,'conditions':'true nonnegative paths must load within given per-link count tolerance; not known to hold for hidden truth','bound_kind':'Dlower exact LP; Dupper sum(b)+LP max totalflow (conservative)','CPU_threads':4,'GPU_used':False},indent=2),encoding='utf-8');print('DONE bounds',time.time()-start)
