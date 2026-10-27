# 中文说明：本轮选参目标包含人工已知路径的S_od/S_dev/S_attr，不以S_link选参。
import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import sys,json,time
from pathlib import Path
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O.parent));sys.path.insert(0,str(O.parent/'improvement_v2'))
from run_task4 import REL,load,load_operator,solve as ridge
from run_v2 import kl,metric
import numpy as np,pandas as pd
WEIGHTS=[0,.25,.5,.75,1.]
def main():
 start=time.time();real=pd.read_csv(O/'real_daily_PM_flows.csv');rows=[];cache={}
 for panel,d in real.groupby('panel'):
  ids,links,D,paths=load_operator(REL/'corridors'/panel/'network');A,c,b,t,_=load(panel,'train',ids,links,D);cache[panel]=(ids,links,D,paths)
  table=d.pivot(index='day_index',columns='station_id',values='PM_mean_flow_per_lane');ref=table.iloc[:15].mean().to_numpy();ratio=np.clip(table.to_numpy()/ref,.25,4);oi=pd.factorize(paths.origin_zone)[0];di=pd.factorize(paths.destination_zone)[0]
  for phase,days in [('calibration',[0,3,6,9,12]),('heldout_days',[15,18,21,24,27,30])]:
   for day in days:
    for scenario in ['destination_shock','origin_destination_shock','additive_clipped']:
     rng=np.random.default_rng(9100+day);z=ratio[day]
     if scenario=='destination_shock':f=b*rng.choice(z,di.max()+1)[di]
     elif scenario=='origin_destination_shock':f=b*np.sqrt(rng.choice(z,oi.max()+1)[oi]*rng.choice(z,di.max()+1)[di])
     else:f=np.maximum(0,b+np.mean(b)*(rng.choice(z,len(b))-1))
     cc=A@f;x,st=ridge(A,cc,b,'ridge',.05);assert st['success'];y,st=kl(A,cc,b,.05);assert st['success']
     for w in WEIGHTS:
      xx=(1-w)*x+w*y;m=metric(xx,f,b,A,cc,paths.destination_zone.to_numpy());m['path_components']=(.45*m['S_od']+.15*m['S_dev']+.15*m['S_attr'])/.75
      rows.append({'panel':panel,'phase':phase,'real_day_index':day,'scenario':scenario,'kl_weight':w,**m})
  pd.DataFrame(rows).to_csv(O/'path_component_trials.csv',index=False);print('path experiments',panel,flush=True)
 df=pd.DataFrame(rows);scores=df.query("phase=='calibration'").groupby(['kl_weight','scenario','panel']).path_components.mean().groupby('kl_weight').mean().sort_values(ascending=False);best=float(scores.index[0]);summary=df.groupby(['phase','scenario','kl_weight'])[['S_od','S_dev','S_attr','S_link','path_components','synthetic_S_ODME']].mean();summary.to_csv(O/'path_components_by_scenario.csv');overall=df.groupby(['phase','kl_weight'])[['S_od','S_dev','S_attr','S_link','path_components','synthetic_S_ODME']].mean();overall.to_csv(O/'path_components_overall.csv')
 (O/'selection.json').write_text(json.dumps({'KL_weight':best,'criterion':'calibration artificial known path: (.45 Sod+.15 Sdev+.15 Sattr)/.75; excludes Slink','scores':{str(k):v for k,v in scores.items()},'flow_source_panels':sorted(cache),'real_calibration_days':'March1-15 2018','real_evaluation_days':'March16-31 2018','path_truth':'artificial; never competition truth','previous_days_seen':'all PM values were audited before this experiment'},indent=2),encoding='utf-8')
 # 只对有同方向实测流量证据的四方向生成候选，其余六方向保留原基准。
 base=pd.read_csv(O.parent/'task4_path_flow.csv',dtype=str);final=base.copy();fits=[]
 for panel,(ids,links,D,paths) in cache.items():
  for split in ['validation','private']:
   A,c,b,t,_=load(panel,split,ids,links,D);x,st=ridge(A,c,b,'ridge',.05);y,st=kl(A,c,b,.05);assert st['success'];pred=(1-best)*x+best*y;mask=(final.panel==panel)&(final.departure_time==t.departure_time.iloc[0]);assert final.loc[mask,'path_id'].tolist()==ids;final.loc[mask,'path_flow']=pred.astype(str)
   dest=paths.destination_zone.to_numpy();ah=pd.Series(pred).groupby(dest).sum();ab=pd.Series(b).groupby(dest).sum()
   fits.append({'panel':panel,'split':split,'S_link':max(0,1-float(abs(A@pred-c).sum()/c.sum())),'prior_relative_L1':float(abs(pred-b).sum()/b.sum()),'destination_prior_TV':float(abs(ah/ah.sum()-ab/ab.sum()).sum()/2)})
 keys=['panel','departure_time','path_id','origin_zone','destination_zone'];v=final.path_flow.to_numpy(float);assert final[keys].equals(base[keys]) and np.isfinite(v).all() and (v>=0).all();final.to_csv(O/'task4_path_objective_candidate.csv',index=False);pd.DataFrame(fits).to_csv(O/'candidate_fit_diagnostics.csv',index=False)
 (O/'manifest.json').write_text(json.dumps({'seconds':time.time()-start,'CPU_threads':4,'GPU_used':False,'CPU_temperature':None,'final_rows':len(final),'all_template_keys_match_baseline':True,'finite_nonnegative':True,'baseline_preserved':True},indent=2),encoding='utf-8');print('DONE weight',best);print(overall.to_string())
if __name__=='__main__':main()
