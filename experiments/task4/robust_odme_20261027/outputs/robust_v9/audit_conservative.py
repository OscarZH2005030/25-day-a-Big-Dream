from pathlib import Path
import sys,json
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O));from run_multi_error import REL,load,load_operator,predict,folds
import pandas as pd,numpy as np
policy=json.loads((O/'conservative_policy.json').read_text())['policy'];rows=[];final=pd.read_csv(O.parent/'task4_path_flow.csv',dtype=str);base=final.copy();deploy=[]
for panel,name in policy.items():
 ids,ls,D,paths=load_operator(REL/'corridors'/panel/'network');A,c,b,t,_=load(panel,'train',ids,ls,D)
 for seed in [20261106,20261107]:
  fd=folds(A,seed)
  for k in range(3):
   sols=predict(A[fd!=k],c[fd!=k],b)
   for method,n in [('official','official'),('conservative',name)]:rows.append({'panel':panel,'seed':seed,'fold':k,'method':method,'NMAE':float(abs(A[fd==k]@sols[n]-c[fd==k]).sum()/c[fd==k].sum())})
 if name!='official':
  for split in ['validation','private']:
   M,cc,bb,tt,_=load(panel,split,ids,ls,D);sols=predict(M,cc,bb);mask=(final.panel==panel)&(final.departure_time==tt.departure_time.iloc[0]);assert final.loc[mask,'path_id'].tolist()==ids;final.loc[mask,'path_flow']=sols[name].astype(str);deploy.append({'panel':panel,'split':split,'candidate':name,'S_link_diagnostic':max(0,1-float(abs(M@sols[name]-cc).sum()/cc.sum()))})
keys=['panel','departure_time','path_id','origin_zone','destination_zone'];v=final.path_flow.to_numpy(float);assert final[keys].equals(base[keys]) and np.isfinite(v).all() and (v>=0).all() and not final.duplicated(keys[:3]).any();final.to_csv(O/'task4_conservative_candidate.csv',index=False);pd.DataFrame(rows).to_csv(O/'fresh_seed_actual_cv.csv',index=False);summary=pd.DataFrame(rows).groupby(['seed','method']).NMAE.mean();summary.to_csv(O/'fresh_seed_actual_cv_overall.csv');pd.DataFrame(deploy).to_csv(O/'conservative_deployment.csv',index=False)
# 在未参与选择的两个合成形状阶段复核固定方向策略，不据结果反向改策略。
tr=pd.read_csv(O/'multi_error_trials.csv');parts=[]
for p,n in policy.items():
 q=tr[(tr.panel==p)&(tr.candidate==n)&(tr.phase!='calibration')].copy();q['method']='conservative';parts.append(q)
q=tr[(tr.candidate=='official')&(tr.phase!='calibration')].copy();q['method']='official';parts.append(q);t=pd.concat(parts,ignore_index=True);t.to_csv(O/'conservative_proxy_trials.csv',index=False);t.groupby(['phase','method'])[['S_od','S_dev','S_attr','S_link','synthetic_S_ODME']].mean().to_csv(O/'conservative_proxy_overall.csv')
(O/'conservative_acceptance.json').write_text(json.dumps({'rows':len(final),'keys_match':True,'finite_nonnegative':True,'duplicates':0,'updated_panels':[p for p,n in policy.items() if n!='official'],'baseline_preserved':True,'synthetic_scores_are_competition_scores':False},indent=2),encoding='utf-8');print(summary.to_string());print(t.groupby(['phase','method'])[['S_od','S_dev','S_attr','synthetic_S_ODME']].mean().to_string())
