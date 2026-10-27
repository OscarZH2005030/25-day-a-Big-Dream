# 中文说明：区分可观测整体计数尺度与不可识别路径分配；人工真值明确标注。
import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import sys,json,time
from pathlib import Path
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O.parent/'dev_v5'));from run_dev import REL,load,load_operator,metric,ridge,kl
import numpy as np,pandas as pd
NAMES=['official','old_amp125','scaled_ridge','scaled_kl','blend25','blend50','blend75']
def predict(A,c,b):
 x,st=ridge(A,c,b,'ridge',.05);assert st['success'];y,st=kl(A,c,b,.05);assert st['success'];old,st=ridge(A,c,np.maximum(b+1.25*(y-b),0),'ridge',.05);assert st['success']
 scale=float(c.sum()/max(float((A@b).sum()),1e-12));z,st=ridge(A,c,b*scale,'ridge',.05);assert st['success'];q,st=kl(A,c,b*scale,.05);assert st['success']
 return {'official':x,'old_amp125':old,'scaled_ridge':z,'scaled_kl':q,'blend25':.75*old+.25*z,'blend50':.5*(old+z),'blend75':.25*old+.75*z},scale

def main():
 start=time.time();real=pd.read_csv(O.parent/'fivemin_v4/real_daily_PM_flows.csv');rows=[];cache={};audits=[]
 for panel,d in real.groupby('panel'):
  ids,links,D,paths=load_operator(REL/'corridors'/panel/'network');A,c,b,t,_=load(panel,'train',ids,links,D);cache[panel]=(ids,links,D,paths);targetscale=float(c.sum()/(A@b).sum())
  table=d.pivot(index='day_index',columns='station_id',values='PM_mean_flow_per_lane');ratio=np.clip(table.to_numpy()/table.iloc[:15].mean().to_numpy(),.25,4);oi=pd.factorize(paths.origin_zone)[0];di=pd.factorize(paths.destination_zone)[0]
  for phase,days in [('calibration',[0,4,8,12]),('evaluation',[15,20,25,30])]:
   for day in days:
    for scenario in ['destination','origin_destination','additive','path_multiplicative','high_dispersion']:
     rng=np.random.default_rng(81000+day);z=np.clip(ratio[day]**2,.1,8);dest=rng.choice(z,di.max()+1)[di];orig=rng.choice(z,oi.max()+1)[oi];ind=rng.choice(z,len(b))
     if scenario=='destination':f=b*dest
     elif scenario=='origin_destination':f=b*np.sqrt(dest*orig)
     elif scenario=='path_multiplicative':f=b*ind
     elif scenario=='high_dispersion':f=b*rng.lognormal(-.5,1.,len(b))
     else:f=np.maximum(0,b+np.mean(b)*(ind-1))
     # 以单一正比例使合成计数/先验加载计数总量匹配train观测比例，精确验收，不逼迫波动指数无限增大。
     f=f*(targetscale*float((A@b).sum())/float((A@f).sum()));cc=A@f;assert abs(cc.sum()/(A@b).sum()-targetscale)<1e-10
     sols,sc=predict(A,cc,b)
     audits.append({'panel':panel,'phase':phase,'day':day,'scenario':scenario,'target_scale':targetscale,'achieved_scale':sc,'actual_shape_NMAE_vs_prior':float(abs(cc-A@b).sum()/cc.sum())})
     for n,x in sols.items():rows.append({'panel':panel,'phase':phase,'day':day,'scenario':scenario,'candidate':n,**metric(x,f,b,A,cc,paths.destination_zone.to_numpy())})
  pd.DataFrame(rows).to_csv(O/'trials.csv',index=False);pd.DataFrame(audits).to_csv(O/'scale_calibration_audit.csv',index=False);print('completed',panel,flush=True)
 df=pd.DataFrame(rows);cal=df[df.phase=='calibration'].groupby('candidate')[['S_od','S_dev','S_attr','synthetic_S_ODME']].mean();ref=cal.loc['old_amp125'];eligible=cal[(cal.S_od>=ref.S_od-.002)&(cal.S_attr>=ref.S_attr-.0005)&(cal.synthetic_S_ODME>=ref.synthetic_S_ODME)]
 chosen=str(eligible.S_dev.idxmax());(O/'selection.json').write_text(json.dumps({'chosen':chosen,'criterion':'calibration Sdev with path,attraction,total guards','truth':'artificial only; global count-scale calibrated exactly to train','calibration':cal.to_dict(orient='index')},indent=2),encoding='utf-8');df.groupby(['phase','candidate'])[['S_od','S_dev','S_attr','S_link','synthetic_S_ODME']].mean().to_csv(O/'overall.csv');df.groupby(['phase','scenario','candidate'])[['S_od','S_dev','S_attr','S_link','synthetic_S_ODME']].mean().to_csv(O/'by_scenario.csv')
 base=pd.read_csv(O.parent/'task4_path_flow.csv',dtype=str);final=base.copy();deploy=[]
 for panel,(ids,links,D,paths) in cache.items():
  for split in ['validation','private']:
   A,c,b,t,_=load(panel,split,ids,links,D);sols,sc=predict(A,c,b);x=sols[chosen];mask=(final.panel==panel)&(final.departure_time==t.departure_time.iloc[0]);assert final.loc[mask,'path_id'].tolist()==ids;final.loc[mask,'path_flow']=x.astype(str);deploy.append({'panel':panel,'split':split,'scale':sc,'candidate':chosen,'S_link':max(0,1-float(abs(A@x-c).sum()/c.sum())),'prior_relative_L1':float(abs(x-b).sum()/b.sum())})
 keys=['panel','departure_time','path_id','origin_zone','destination_zone'];v=final.path_flow.to_numpy(float);assert final[keys].equals(base[keys]) and np.isfinite(v).all() and (v>=0).all() and not final.duplicated(keys[:3]).any();final.to_csv(O/'task4_scale_shape_candidate.csv',index=False);pd.DataFrame(deploy).to_csv(O/'deployment_diagnostics.csv',index=False)
 (O/'manifest.json').write_text(json.dumps({'seconds':time.time()-start,'candidate':chosen,'rows':len(final),'keys_match':True,'finite_nonnegative':True,'duplicates':0,'CPU_threads':4,'GPU_used':False,'CPU_temperature':None,'baseline_preserved':True,'calibration_scope':'global scale only; not full spatial count-vector match'},indent=2),encoding='utf-8');print('DONE',chosen);print(df[df.phase=='evaluation'].groupby('candidate')[['S_od','S_dev','S_attr','synthetic_S_ODME']].mean().to_string())
if __name__=='__main__':main()
