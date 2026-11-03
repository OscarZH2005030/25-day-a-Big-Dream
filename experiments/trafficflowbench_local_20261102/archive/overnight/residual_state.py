"""Supervised residual correction, trained/selected in January only.

February was exposed by prior development. Report as repeated development, not
blind test or official physics/leaderboard improvement. No network operations.
"""
import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import json,hashlib,msvcrt,traceback
from pathlib import Path
import numpy as np,pandas as pd,lightgbm as lgb
import state_experiment as e
W=e.WORK;O=e.OUT/'residual_state';O.mkdir(parents=True,exist_ok=True)
TRAIN=['2031-01-05','2031-01-08','2031-01-11','2031-01-14','2031-01-17','2031-01-20']
ALPHAS=[0.,.1,.25,.5,1.]
RECIPES={'small':dict(num_leaves=7,min_data_in_leaf=1000),'medium':dict(num_leaves=15,min_data_in_leaf=500)}

def features(panel,date,p):
 assert not any(c.startswith('true') for c in p.columns),'Truth not accepted by feature function'
 folder=W/'residual_features'/panel;folder.mkdir(parents=True,exist_ok=True);cache=folder/(date+'.npz')
 signature=hashlib.sha256(p[e.KEYS].to_csv(index=False).encode()).hexdigest()
 if cache.exists():
  z=np.load(cache);assert str(z['key_sha256'])==signature;return z['X']
 path=next((e.DATA/'corridors'/panel/'train/mainline_states_masked').rglob('synthetic_mainline_'+date.replace('-','_')+'.parquet'))
 raw=pd.read_parquet(path,columns=['timestamp','station_id','link_id','speed_kmh','flow_vph']);raw.station_id=raw.station_id.astype(str);raw.link_id=raw.link_id.astype(str)
 fd=pd.read_csv(e.DATA/'corridors'/panel/'network/fd_parameters.csv',dtype={'link_id':str}).groupby('link_id').mean(numeric_only=True);raw['q_lane']=raw.flow_vph/raw.link_id.map(fd.lanes)
 ts=pd.to_datetime(raw.timestamp,utc=True);raw['tick']=ts.dt.hour*12+ts.dt.minute//5;stations=sorted(raw.station_id.unique());si={s:i for i,s in enumerate(stations)}
 def grid(col):return raw.pivot(index='tick',columns='station_id',values=col).reindex(index=range(288),columns=stations).to_numpy()
 vg=grid('speed_kmh');qg=grid('q_lane');t=pd.to_datetime(p.timestamp,utc=True);ti=(t.dt.hour*12+t.dt.minute//5).to_numpy();ii=p.station_id.map(si).to_numpy();idx=(ti,ii)
 assert not np.isfinite(vg[idx]).any() and not np.isfinite(qg[idx]).any()
 vv=p.speed_kmh.to_numpy();qq=p.flow_vph.to_numpy()/p.lanes.to_numpy();parts=[vv/25,qq/600,np.sin(ti*2*np.pi/288),np.cos(ti*2*np.pi/288),t.dt.dayofweek.to_numpy()/6]
 for g,base,scale in [(vg,vv,25),(qg,qq,600)]:
  f=pd.DataFrame(g);prev=f.ffill().to_numpy();nxt=f.bfill().to_numpy();ticks=np.broadcast_to(np.arange(288)[:,None],g.shape).astype(float);observed=np.where(np.isfinite(g),ticks,np.nan);before=pd.DataFrame(observed).ffill().to_numpy();after=pd.DataFrame(observed).bfill().to_numpy();gl=ticks-before;gr=after-ticks
  interp=(prev*gr+nxt*gl)/(gl+gr+1e-12);parts.extend([(prev[idx]-base)/scale,(nxt[idx]-base)/scale,(interp[idx]-base)/scale,np.log1p(gl[idx]),np.log1p(gr[idx])])
  for lag in [-3,-1,1,3]:
   shift=np.full_like(g,np.nan)
   if lag>0:shift[:-lag]=g[lag:]
   else:shift[-lag:]=g[:lag]
   parts.append((shift[idx]-base)/scale)
  count=np.isfinite(g).sum(1);mean=np.nansum(g,axis=1)/np.maximum(count,1);mean[count==0]=np.nan;parts.append((mean[ti]-base)/scale)
 pf=p.link_id.map(fd.free_speed_kmh).to_numpy();cap=p.link_id.map(fd.capacity_vph).to_numpy()/p.lanes.to_numpy();kc=cap/np.maximum(pf,1);kj=p.link_id.map(fd.k_jam).to_numpy()/p.lanes.to_numpy();density=qq/np.maximum(vv,1);qfd=np.where(density<=kc,pf*density,cap/np.maximum(kj-kc,1e-6)*np.maximum(kj-density,0));parts.extend([vv/np.maximum(pf,1),qq/np.maximum(cap,1),(qfd-qq)/600])
 X=np.column_stack(parts).astype('float32');X[~np.isfinite(X)]=np.nan;np.savez_compressed(cache,X=X,key_sha256=np.array(signature));return X
def build(items,panel):
 return [features(panel,it['date'],it['frame'][e.KEYS+['speed_kmh','flow_vph','lanes']].copy()) for it in items]
def corrected(item,r,alpha):
 f=item['frame'];p=f[['speed_kmh','flow_vph']].to_numpy().copy();p[:,0]+=alpha*r[:,0]*25;p[:,1]+=alpha*r[:,1]*600*f.lanes.to_numpy();return np.maximum(p,0)
def models_fit(panel,recipe,train,dev,Xtrain,Xdev):
 folder=O/'models'/panel/recipe;folder.mkdir(parents=True,exist_ok=True)
 xt=np.concatenate(Xtrain);xv=np.concatenate(Xdev);yt=np.concatenate([(v['frame'][['true_v','true_q_lane']].to_numpy()-v['frame'][['speed_kmh','q_lane']].to_numpy())/[25,600] for v in train]);yv=np.concatenate([(v['frame'][['true_v','true_q_lane']].to_numpy()-v['frame'][['speed_kmh','q_lane']].to_numpy())/[25,600] for v in dev]);models=[]
 for c in range(2):
  path=folder/('speed.txt' if c==0 else 'flow_per_lane.txt')
  if path.exists():m=lgb.Booster(model_file=str(path))
  else:
   params=dict(objective='regression',metric='rmse',learning_rate=.03,lambda_l2=20.,feature_fraction=.9,verbosity=-1,num_threads=4,seed=2026110202,deterministic=True,force_col_wise=True,**RECIPES[recipe]);m=lgb.train(params,lgb.Dataset(xt,label=yt[:,c]),num_boost_round=200,valid_sets=[lgb.Dataset(xv,label=yv[:,c])],callbacks=[lgb.early_stopping(30,verbose=False)]);m.save_model(str(path))
  models.append(m)
 return models
def predict(models,X):return [np.column_stack([m.predict(x,num_threads=4) for m in models]) for x in X]
def main():
 routes=e.predictor.load_routes();protocol=dict(scope='Repeated-development January-trained residual learner; February never used in this fit/selection, but already exposed by prior research. No official Task3 or leaderboard claims.',train_dates=TRAIN,dev_dates='last available January date per regime, >=Jan21',recipes=RECIPES,alpha_grid=ALPHAS,selection='maximize January dev S_state subject to station target deltaN RMSE <=1.01*baseline; retain zero correction as fallback',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seed=2026110202,no_kaggle_upload=True)
 if (O/'protocol.json').exists():assert json.loads((O/'protocol.json').read_text())==protocol
 else:e.save(O/'protocol.json',protocol)
 policy={};devrows=[];templates={}
 for panel,route in routes.items():
  e.state(status='running',stage='residual_january_training',panel=panel,next='Inspect residual_state.log; no duplicate GPU process or policy retuning on February')
  keys=pd.read_csv(e.DATA/'task1'/panel/'train/sample_submission_state.csv',dtype=str,usecols=e.KEYS);templates[panel]=keys;dates=keys.assign(date=keys.timestamp.str[:10]);devdates=sorted(dates[(dates.date>='2031-01-21')&(dates.date<='2031-01-31')].groupby('mask_regime').date.max().tolist());assert set(devdates).isdisjoint(TRAIN)
  train=e.prepare(panel,TRAIN,route,keys);dev=e.prepare(panel,devdates,route,keys);xt=build(train,panel);xv=build(dev,panel);baseline=e.metrics(dev,[v['frame'][['speed_kmh','flow_vph']].to_numpy() for v in dev]);best=dict(recipe='baseline',alpha=0.,metrics=baseline)
  for name in RECIPES:
   models=models_fit(panel,name,train,dev,xt,xv);res=predict(models,xv)
   for alpha in ALPHAS:
    score=e.metrics(dev,[corrected(v,r,alpha) for v,r in zip(dev,res)]);eligible=score['station_target_deltaN_RMSE']<=baseline['station_target_deltaN_RMSE']*1.01;devrows.append(dict(panel=panel,recipe=name,alpha=alpha,eligible=eligible,**score))
    if eligible and score['S_state']>best['metrics']['S_state']:best=dict(recipe=name,alpha=alpha,metrics=score)
  policy[panel]=best;e.save(O/'policy_partial.json',policy);pd.DataFrame(devrows).to_csv(O/'january_development.csv',index=False);print('FROZEN RESIDUAL',panel,best,flush=True)
 e.save(O/'frozen_policy.json',dict(policy=policy,utc=e.datetime.now(e.timezone.utc).isoformat()))
 rows=[]
 for panel,route in routes.items():
  e.state(status='running',stage='residual_february_evaluation',panel=panel,next='Evaluate already frozen residual policy; no Kaggle uploads')
  done=O/(panel+'_evaluation.json')
  if done.exists():rows.extend(json.loads(done.read_text()));continue
  k=templates[panel];dates=sorted(set(k.timestamp.str[:10]));dates=[d for d in dates if '2031-02-01'<=d<='2031-02-28'];items=e.prepare(panel,dates,route,k);base=e.metrics(items,[it['frame'][['speed_kmh','flow_vph']].to_numpy() for it in items]);p=policy[panel]
  if p['recipe']=='baseline':score=base
  else:
   folder=O/'models'/panel/p['recipe'];models=[lgb.Booster(model_file=str(folder/n)) for n in ['speed.txt','flow_per_lane.txt']];res=predict(models,build(items,panel));score=e.metrics(items,[corrected(it,r,p['alpha']) for it,r in zip(items,res)])
  rr=[dict(panel=panel,variant='baseline',**base),dict(panel=panel,variant='frozen_residual',recipe=p['recipe'],alpha=p['alpha'],**score)];rows.extend(rr);e.save(done,rr);pd.DataFrame(rows).to_csv(O/'february_by_panel.csv',index=False);print('FEB RESIDUAL',panel,rr,flush=True)
 summary=pd.DataFrame(rows).groupby('variant')[['S_state','speed_RMSE','flow_per_lane_RMSE','station_target_deltaN_RMSE']].mean();summary.to_csv(O/'february_summary.csv');print(summary.to_string(),flush=True);e.state(status='stage_complete',stage='residual_state_complete',next='Assess repeated-development evidence and preserve official baseline; continue justified research until deadline.')
if __name__=='__main__':
 lock=(W/'research.lock').open('r+b');lock.seek(0)
 try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 except OSError:raise SystemExit('Research lock held')
 try:main()
 except Exception as err:e.state(status='failed_or_cutoff',stage='residual_state',error=repr(err),next='Fix/resume residual experiment without changing frozen protocol or examining February for tuning');traceback.print_exc();raise
 finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
