# 中文说明：保留S_dev优先与综合分优先两种选择；复核真实train留出计数但不称为路径真值。
from pathlib import Path
import sys,json
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O));from run_scale_shape import REL,load,load_operator,predict
sys.path.insert(0,str(O.parent/'improvement_v2'));from run_v2 import folds
import pandas as pd,numpy as np
sel=json.loads((O/'selection.json').read_text());cal=pd.DataFrame(sel['calibration']).T;name=str(cal[cal.S_dev>=cal.loc['old_amp125','S_dev']].synthetic_S_ODME.idxmax());base=pd.read_csv(O.parent/'task4_path_flow.csv',dtype=str);final=base.copy();real=pd.read_csv(O.parent/'fivemin_v4/real_daily_PM_flows.csv');rows=[];deploy=[]
for panel in sorted(real.panel.unique()):
 ids,links,D,paths=load_operator(REL/'corridors'/panel/'network');A,c,b,t,_=load(panel,'train',ids,links,D);fd=folds(A,20261102)
 for k in range(3):
  sols,s=predict(A[fd!=k],c[fd!=k],b)
  for cand in ['official','old_amp125',name]:rows.append({'panel':panel,'fold':k,'candidate':cand,'count_NMAE':float(abs(A[fd==k]@sols[cand]-c[fd==k]).sum()/c[fd==k].sum())})
 for split in ['validation','private']:
  M,cc,bb,t,_=load(panel,split,ids,links,D);sols,s=predict(M,cc,bb);mask=(final.panel==panel)&(final.departure_time==t.departure_time.iloc[0]);final.loc[mask,'path_flow']=sols[name].astype(str);deploy.append({'panel':panel,'split':split,'candidate':name,'scale':s,'prior_relative_L1':float(abs(sols[name]-bb).sum()/bb.sum())})
keys=['panel','departure_time','path_id','origin_zone','destination_zone'];v=final.path_flow.to_numpy(float);assert final[keys].equals(base[keys]) and np.isfinite(v).all() and (v>=0).all() and not final.duplicated(keys[:3]).any();final.to_csv(O/'task4_balanced_scale_candidate.csv',index=False)
pd.DataFrame(rows).to_csv(O/'real_train_count_holdout.csv',index=False);pd.DataFrame(deploy).to_csv(O/'balanced_deployment.csv',index=False);(O/'balanced_selection.json').write_text(json.dumps({'candidate':name,'criterion':'calibration synthetic SODME with Sdev>=old_amp125','truth':'artificial only','final_rows':len(final),'keys_match':True,'finite_nonnegative':True,'baseline_preserved':True},indent=2),encoding='utf-8');print('balanced',name);print(pd.DataFrame(rows).groupby('candidate').count_NMAE.mean())
