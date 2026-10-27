# 中文说明：针对S_dev测试结构先验和KL调整幅度，所有真值仅来自明确标注的人工场景。
import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import sys,json,time
from pathlib import Path
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O.parent));sys.path.insert(0,str(O.parent/'improvement_v2'))
from run_task4 import REL,load,load_operator,solve as ridge
from run_v2 import kl,metric
import numpy as np,pandas as pd
from scipy.optimize import minimize
NAMES=['official','kl','kl_amp125','kl_amp150','kl_amp200','dest_factor','od_factor','kl_dest50','kl_od50']
def factor(A,c,b,paths,both=False):
 di=pd.factorize(paths.destination_zone)[0];oi=pd.factorize(paths.origin_zone)[0];nd=di.max()+1;no=oi.max()+1 if both else 0;scale=max(np.median(c),1.)
 def fun(theta):
  z=theta[di]+(theta[nd+oi] if both else 0);f=b*np.exp(z);e=(A@f-c)/scale
  g=np.asarray(A.T@e).ravel()*f/scale;grad=np.bincount(di,weights=g,minlength=nd)
  if both:grad=np.r_[grad,np.bincount(oi,weights=g,minlength=no)]
  return .5*(e@e+.001*(theta@theta)),grad+.001*theta
 res=minimize(fun,np.zeros(nd+no),jac=True,method='L-BFGS-B',bounds=[(-2,2)]*(nd+no),options={'maxiter':2000,'ftol':1e-12,'gtol':1e-7})
 assert res.success,res.message
 return b*np.exp(res.x[di]+(res.x[nd+oi] if both else 0))
def candidates(A,c,b,paths):
 x,st=ridge(A,c,b,'ridge',.05);assert st['success'];y,st=kl(A,c,b,.05);assert st['success'];out={'official':x,'kl':y}
 for a in [1.25,1.5,2.]:
  # 沿已有KL调整方向外推后用原计数重新约束，避免直接缩放破坏计数。
  bb=np.maximum(b+a*(y-b),0);z,st=ridge(A,c,bb,'ridge',.05);assert st['success'];out[{1.25:'kl_amp125',1.5:'kl_amp150',2.:'kl_amp200'}[a]]=z
 dest=factor(A,c,b,paths);od=factor(A,c,b,paths,True)
 out.update(dest_factor=dest,od_factor=od,kl_dest50=.5*(y+dest),kl_od50=.5*(y+od))
 return out

def main():
 start=time.time();real=pd.read_csv(O.parent/'fivemin_v4/real_daily_PM_flows.csv');rows=[];cache={}
 for panel,d in real.groupby('panel'):
  ids,links,D,paths=load_operator(REL/'corridors'/panel/'network');A,c,b,t,_=load(panel,'train',ids,links,D);cache[panel]=(ids,links,D,paths)
  table=d.pivot(index='day_index',columns='station_id',values='PM_mean_flow_per_lane');ref=table.iloc[:15].mean().to_numpy();ratio=np.clip(table.to_numpy()/ref,.25,4);oi=pd.factorize(paths.origin_zone)[0];di=pd.factorize(paths.destination_zone)[0]
  # 前轮所有日数据及结果已查看；本轮改变随机映射，属于重复数据上的敏感性复核。
  for phase,days in [('calibration',[1,4,7,10,13]),('evaluation',[16,19,22,25,28])]:
   for day in days:
    for scenario in ['destination_shock','origin_destination_shock','additive_clipped']:
     rng=np.random.default_rng(19100+day);z=ratio[day]
     if scenario=='destination_shock':f=b*rng.choice(z,di.max()+1)[di]
     elif scenario=='origin_destination_shock':f=b*np.sqrt(rng.choice(z,oi.max()+1)[oi]*rng.choice(z,di.max()+1)[di])
     else:f=np.maximum(0,b+np.mean(b)*(rng.choice(z,len(b))-1))
     cc=A@f;sols=candidates(A,cc,b,paths)
     for name,x in sols.items():
      m=metric(x,f,b,A,cc,paths.destination_zone.to_numpy());m['deviation_ratio']=float(abs(x-b).sum()/max(abs(f-b).sum(),1e-12));rows.append({'panel':panel,'phase':phase,'real_day_index':day,'scenario':scenario,'candidate':name,**m})
  pd.DataFrame(rows).to_csv(O/'all_trials.csv',index=False);print('completed',panel,flush=True)
 df=pd.DataFrame(rows);cal=df[df.phase=='calibration'].groupby('candidate')[['S_od','S_dev','S_attr','synthetic_S_ODME']].mean();ref=cal.loc['kl'];eligible=cal[(cal.S_od>=ref.S_od-.003)&(cal.S_attr>=ref.S_attr-.001)&(cal.synthetic_S_ODME>=ref.synthetic_S_ODME)]
 chosen=str(eligible.S_dev.idxmax()) if len(eligible) else 'kl'
 (O/'selection.json').write_text(json.dumps({'chosen':chosen,'criterion':'maximize synthetic calibration Sdev; Sod>=KL-.003,Sattr>=KL-.001,SODME>=KL','path_truth':'artificial only','previous_data_seen':True,'calibration':cal.to_dict(orient='index')},indent=2),encoding='utf-8')
 df.groupby(['phase','candidate'])[['S_od','S_dev','S_attr','S_link','synthetic_S_ODME','deviation_ratio']].mean().to_csv(O/'overall.csv');df.groupby(['phase','scenario','candidate'])[['S_od','S_dev','S_attr','S_link','synthetic_S_ODME','deviation_ratio']].mean().to_csv(O/'by_scenario.csv')
 base=pd.read_csv(O.parent/'task4_path_flow.csv',dtype=str);final=base.copy()
 for panel,(ids,links,D,paths) in cache.items():
  for split in ['validation','private']:
   A,c,b,t,_=load(panel,split,ids,links,D);sols=candidates(A,c,b,paths);mask=(final.panel==panel)&(final.departure_time==t.departure_time.iloc[0]);assert final.loc[mask,'path_id'].tolist()==ids;final.loc[mask,'path_flow']=sols[chosen].astype(str)
 keys=['panel','departure_time','path_id','origin_zone','destination_zone'];v=final.path_flow.to_numpy(float);assert final[keys].equals(base[keys]) and np.isfinite(v).all() and (v>=0).all();final.to_csv(O/'task4_dev_candidate.csv',index=False)
 (O/'manifest.json').write_text(json.dumps({'seconds':time.time()-start,'candidate':chosen,'rows':len(final),'all_template_keys_match':True,'finite_nonnegative':True,'CPU_threads':4,'GPU_used':False,'CPU_temperature':None,'baseline_preserved':True},indent=2),encoding='utf-8');print('DONE',chosen);print(df[df.phase=='evaluation'].groupby('candidate')[['S_od','S_dev','S_attr','synthetic_S_ODME']].mean().to_string())
if __name__=='__main__':main()
