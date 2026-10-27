# 中文说明：固定新一轮扩展情景，细化调整幅度；只在calibration选参。
import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import sys,json,time
from pathlib import Path
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O.parent/'dev_v5'));from run_dev import REL,load,load_operator,metric,ridge,kl
import numpy as np,pandas as pd
ALPHAS=[1.0,1.20,1.25,1.275,1.30,1.325,1.35,1.375,1.40,1.45,1.50]
def main():
 start=time.time();real=pd.read_csv(O.parent/'fivemin_v4/real_daily_PM_flows.csv');rows=[];cache={}
 for panel,d in real.groupby('panel'):
  ids,links,D,paths=load_operator(REL/'corridors'/panel/'network');A,c,b,t,_=load(panel,'train',ids,links,D);cache[panel]=(ids,links,D,paths)
  table=d.pivot(index='day_index',columns='station_id',values='PM_mean_flow_per_lane');ratio=np.clip(table.to_numpy()/table.iloc[:15].mean().to_numpy(),.25,4);oi=pd.factorize(paths.origin_zone)[0];di=pd.factorize(paths.destination_zone)[0]
  for phase,days in [('calibration',[0,4,8,12]),('evaluation',[15,20,25,30])]:
   for day in days:
    for scenario in ['destination','origin_destination','additive','path_multiplicative','mixed']:
     rng=np.random.default_rng(61000+day);gain=1. if day%2==0 else 2.;z=np.clip(ratio[day]**gain,.1,8);dest=rng.choice(z,di.max()+1)[di];orig=rng.choice(z,oi.max()+1)[oi];ind=rng.choice(z,len(b))
     if scenario=='destination':f=b*dest
     elif scenario=='origin_destination':f=b*np.sqrt(dest*orig)
     elif scenario=='path_multiplicative':f=b*ind
     elif scenario=='mixed':f=np.maximum(0,b*np.sqrt(dest*orig)+.5*np.mean(b)*(ind-1))
     else:f=np.maximum(0,b+np.mean(b)*(ind-1))
     cc=A@f;y,st=kl(A,cc,b,.05);assert st['success']
     for alpha in ALPHAS:
      x,st=ridge(A,cc,np.maximum(b+alpha*(y-b),0),'ridge',.05);assert st['success'];rows.append({'panel':panel,'phase':phase,'day':day,'scenario':scenario,'alpha':alpha,**metric(x,f,b,A,cc,paths.destination_zone.to_numpy())})
  pd.DataFrame(rows).to_csv(O/'fine_amplitude_trials.csv',index=False);print('fine grid',panel,flush=True)
 df=pd.DataFrame(rows);cal=df[df.phase=='calibration'].groupby('alpha')[['S_od','S_dev','S_attr','synthetic_S_ODME']].mean();ref=cal.loc[1.25]
 # 相对原方案最多损失0.002路径分、0.0005吸引力分；所有情景的综合不能跌超0.001。
 scenario_cal=df[df.phase=='calibration'].groupby(['alpha','scenario']).synthetic_S_ODME.mean().unstack();worst=(scenario_cal-scenario_cal.loc[1.25]).min(axis=1)
 eligible=cal[(cal.S_od>=ref.S_od-.002)&(cal.S_attr>=ref.S_attr-.0005)&(cal.synthetic_S_ODME>=ref.synthetic_S_ODME)&(worst>=-.001)]
 best=float(eligible.S_dev.idxmax());df.groupby(['phase','alpha'])[['S_od','S_dev','S_attr','S_link','synthetic_S_ODME']].mean().to_csv(O/'fine_amplitude_overall.csv');df.groupby(['phase','scenario','alpha'])[['S_od','S_dev','S_attr','S_link','synthetic_S_ODME']].mean().to_csv(O/'fine_amplitude_by_scenario.csv')
 (O/'fine_selection.json').write_text(json.dumps({'alpha':best,'criterion':'calibration Sdev with path/attraction/each-scenario guards','path_truth':'artificial only; earlier evaluations already viewed','calibration_scores':cal.to_dict(orient='index')},indent=2),encoding='utf-8')
 base=pd.read_csv(O.parent/'task4_path_flow.csv',dtype=str);final=base.copy();diagnostics=[]
 for panel,(ids,links,D,paths) in cache.items():
  for split in ['validation','private']:
   A,c,b,t,_=load(panel,split,ids,links,D);y,st=kl(A,c,b,.05);assert st['success'];x,st=ridge(A,c,np.maximum(b+best*(y-b),0),'ridge',.05);assert st['success'];mask=(final.panel==panel)&(final.departure_time==t.departure_time.iloc[0]);final.loc[mask,'path_flow']=x.astype(str);diagnostics.append({'panel':panel,'split':split,'alpha':best,'S_link':max(0,1-float(abs(A@x-c).sum()/c.sum())),'prior_relative_L1':float(abs(x-b).sum()/b.sum())})
 keys=['panel','departure_time','path_id','origin_zone','destination_zone'];v=final.path_flow.to_numpy(float);assert final[keys].equals(base[keys]) and np.isfinite(v).all() and (v>=0).all() and not final.duplicated(keys[:3]).any();final.to_csv(O/'task4_fine_amplitude_candidate.csv',index=False);pd.DataFrame(diagnostics).to_csv(O/'fine_deployment_diagnostics.csv',index=False)
 (O/'fine_manifest.json').write_text(json.dumps({'seconds':time.time()-start,'alpha':best,'rows':len(final),'keys_match':True,'finite_nonnegative':True,'duplicates':0,'GPU_used':False,'CPU_threads':4,'CPU_temperature':None,'baseline_preserved':True},indent=2),encoding='utf-8');print('DONE alpha',best);print(df[(df.phase=='evaluation')&df.alpha.isin([1.25,best])].groupby('alpha')[['S_od','S_dev','S_attr','synthetic_S_ODME']].mean().to_string())
if __name__=='__main__':main()
