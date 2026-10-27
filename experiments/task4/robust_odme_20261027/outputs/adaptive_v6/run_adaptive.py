# 中文说明：控制器只读取计数、先验和候选解特征；人工f*仅用于校准目标与离方向评估。
import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import sys,json,time,pickle,hashlib
from pathlib import Path
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O.parent/'dev_v5'));from run_dev import REL,load,load_operator,metric,candidates
import numpy as np,pandas as pd
from sklearn.ensemble import RandomForestRegressor
CANDS=['kl','kl_amp125','kl_amp150','dest_factor','kl_dest50','kl_od50']
REF=CANDS.index('kl_amp125')
def features(A,c,b,sols):
 loadb=A@b;r=(c-loadb)/np.maximum(c,1);out=[np.mean(r),np.std(r),np.mean(abs(r)),np.max(abs(r)),np.std(b)/np.mean(b),np.std(c)/np.mean(c),np.mean(np.log1p(b)),len(c)/len(b)]
 for n in CANDS:
  x=sols[n];delta=x-b;rel=delta/np.maximum(b,np.median(b)*.1)
  out +=[abs(delta).sum()/b.sum(),np.std(rel),np.mean(rel),np.mean(x<1e-8),abs(A@x-c).sum()/c.sum(),x.sum()/b.sum()]
 return np.array(out,float)
def controller(X,Y):
 m=RandomForestRegressor(n_estimators=160,max_depth=5,min_samples_leaf=5,max_features=.85,n_jobs=4,random_state=6127);m.fit(X,Y.reshape(len(Y),-1));return m
def choose(pred):
 pred=pred.reshape(len(CANDS),4);ref=pred[REF]
 ok=(pred[:,0]>=ref[0]-.002)&(pred[:,2]>=ref[2]-.0005)&(pred[:,3]>=ref[3])
 return int(np.argmax(np.where(ok,pred[:,1],-np.inf))) if ok.any() else REF

def main():
 start=time.time();real=pd.read_csv(O.parent/'fivemin_v4/real_daily_PM_flows.csv');records=[];X=[];Y=[];panels=[];phases=[];meta=[];cache={}
 for panel,d in real.groupby('panel'):
  ids,links,D,paths=load_operator(REL/'corridors'/panel/'network');A,c,b,t,_=load(panel,'train',ids,links,D);cache[panel]=(ids,links,D,paths)
  table=d.pivot(index='day_index',columns='station_id',values='PM_mean_flow_per_lane');ratio=np.clip(table.to_numpy()/table.iloc[:15].mean().to_numpy(),.25,4);oi=pd.factorize(paths.origin_zone)[0];di=pd.factorize(paths.destination_zone)[0]
  for phase,days in [('calibration',[0,4,8,12]),('evaluation',[15,20,25,30])]:
   for day in days:
    for scenario in ['destination','origin_destination','additive','path_multiplicative','mixed']:
     rng=np.random.default_rng(61000+day);gain=1. if day%2==0 else 2.;z=np.clip(ratio[day]**gain,.1,8)
     dest=rng.choice(z,di.max()+1)[di];orig=rng.choice(z,oi.max()+1)[oi];ind=rng.choice(z,len(b))
     if scenario=='destination':f=b*dest
     elif scenario=='origin_destination':f=b*np.sqrt(dest*orig)
     elif scenario=='path_multiplicative':f=b*ind
     elif scenario=='mixed':f=np.maximum(0,b*np.sqrt(dest*orig)+.5*np.mean(b)*(ind-1))
     else:f=np.maximum(0,b+np.mean(b)*(ind-1))
     cc=A@f;sols=candidates(A,cc,b,paths);feat=features(A,cc,b,sols);values=[]
     for n in CANDS:
      m=metric(sols[n],f,b,A,cc,paths.destination_zone.to_numpy());values.append([m[k] for k in ['S_od','S_dev','S_attr','synthetic_S_ODME']]);records.append({'panel':panel,'phase':phase,'day':day,'scenario':scenario,'gain':gain,'candidate':n,**m})
     X.append(feat);Y.append(values);panels.append(panel);phases.append(phase);meta.append({'panel':panel,'phase':phase,'day':day,'scenario':scenario,'gain':gain})
  pd.DataFrame(records).to_csv(O/'candidate_trials.csv',index=False);print('generated',panel,flush=True)
 X=np.array(X);Y=np.array(Y);panels=np.array(panels);phases=np.array(phases);np.savez_compressed(O/'controller_training.npz',X=X,Y=Y,panels=panels,phases=phases)
 # 控制器评估时完全排除当前方向的人工标签，只在另外三方向前半月校准。
 selected=[]
 for panel in sorted(set(panels)):
  train=(panels!=panel)&(phases=='calibration');test=(panels==panel)&(phases=='evaluation');model=controller(X[train],Y[train]);pred=model.predict(X[test])
  for idx,pr in zip(np.flatnonzero(test),pred):
   k=choose(pr)
   for name,j in [('fixed_amp125',REF),('adaptive',k)]:
    selected.append({**meta[idx],'method':name,'chosen_candidate':CANDS[j],**dict(zip(['S_od','S_dev','S_attr','synthetic_S_ODME'],Y[idx,j]))})
 pd.DataFrame(selected).to_csv(O/'leave_direction_out_results.csv',index=False);s=pd.DataFrame(selected);s.groupby('method')[['S_od','S_dev','S_attr','synthetic_S_ODME']].mean().to_csv(O/'overall.csv');s.groupby(['scenario','method'])[['S_od','S_dev','S_attr','synthetic_S_ODME']].mean().to_csv(O/'by_scenario.csv');s.groupby(['panel','method'])[['S_od','S_dev','S_attr','synthetic_S_ODME']].mean().to_csv(O/'by_panel.csv')
 # 所有校准方向训练最终控制器，但只有评估均值同时满足约束才输出为推荐候选。
 finalmodel=controller(X[phases=='calibration'],Y[phases=='calibration']);pickle.dump(finalmodel,open(O/'controller.pkl','wb'))
 stats=s.groupby('method')[['S_od','S_dev','S_attr','synthetic_S_ODME']].mean();r=stats.loc['fixed_amp125'];a=stats.loc['adaptive'];recommend=bool(a.S_dev>r.S_dev and a.S_od>=r.S_od-.002 and a.S_attr>=r.S_attr-.0005 and a.synthetic_S_ODME>=r.synthetic_S_ODME)
 base=pd.read_csv(O.parent/'task4_path_flow.csv',dtype=str);final=base.copy();decisions=[]
 for panel,(ids,links,D,paths) in cache.items():
  for split in ['validation','private']:
   A,c,b,t,_=load(panel,split,ids,links,D);sols=candidates(A,c,b,paths);pr=finalmodel.predict(features(A,c,b,sols)[None,:])[0];j=choose(pr);name=CANDS[j];mask=(final.panel==panel)&(final.departure_time==t.departure_time.iloc[0]);assert final.loc[mask,'path_id'].tolist()==ids;final.loc[mask,'path_flow']=sols[name].astype(str);decisions.append({'panel':panel,'split':split,'candidate':name,'evidence':'synthetic controller; no hidden truth','S_link_diagnostic':max(0,1-float(abs(A@sols[name]-c).sum()/c.sum()))})
 keys=['panel','departure_time','path_id','origin_zone','destination_zone'];v=final.path_flow.to_numpy(float);assert final[keys].equals(base[keys]) and np.isfinite(v).all() and (v>=0).all();final.to_csv(O/'task4_adaptive_candidate.csv',index=False);pd.DataFrame(decisions).to_csv(O/'deployment_decisions.csv',index=False)
 (O/'manifest.json').write_text(json.dumps({'seconds':time.time()-start,'synthetic_recommendation':recommend,'scope':'four source-supported D7 directions; six others baseline','rows':len(final),'finite_nonnegative_keys_match':True,'CPU_threads':4,'GPU_used':False,'CPU_temperature':None,'path_truth_available':False,'baseline_preserved':True},indent=2),encoding='utf-8');print('DONE',recommend);print(stats.to_string())
if __name__=='__main__':main()
