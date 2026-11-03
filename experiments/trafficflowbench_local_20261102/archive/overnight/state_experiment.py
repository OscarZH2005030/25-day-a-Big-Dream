"""Frozen, local-only calibration study. No Kaggle/network operations."""
import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import sys,json,hashlib,time,traceback,msvcrt
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'work/overnight';OUT=ROOT/'outputs/overnight'
OUT.mkdir(parents=True,exist_ok=True)
DATA=Path('C:/Users/Lenovo/Downloads/2026-ieee-big-data-traffic-flow-bench/kaggle_public')
BUNDLE=ROOT/'work/task1_bundle/TrafficFlowBench-Best-20261026'
sys.path.insert(0,str(BUNDLE))
import predict_selected as predictor
torch.set_num_threads(4)
DEADLINE=datetime(2026,11,2,16,tzinfo=timezone.utc)
PENALTIES=[.001,.01,.1,1.,10.]
CANDIDATES=['baseline','affine','density_010','density_025','qk_010','qk_025','affine_density_010']
KEYS=['panel','timestamp','station_id','link_id','mask_regime']

def save(path,x):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf8');tmp.replace(path)
def state(**kw):
 save(WORK/'STATE.json',dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),kaggle_submission_authorized=False,remaining_kaggle_opportunities_user_reported=2,automation_id='trafficflowbench',**kw))
def check_deadline():
 if datetime.now(timezone.utc)>=DEADLINE:raise TimeoutError('Research cutoff reached; caches and partial results retained')
def affine_fit(items,penalty):
 z=np.concatenate([x['frame'][['speed_kmh','q_lane']].to_numpy()/[25,600] for x in items]);y=np.concatenate([x['frame'][['true_v','true_q_lane']].to_numpy()/[25,600] for x in items]);result=[]
 for c in range(2):
  X=np.column_stack([z[:,c],np.ones(len(z))]);error=y[:,c]-z[:,c];coef=np.linalg.solve(X.T@X+len(z)*penalty*np.eye(2),X.T@error);result.append(coef)
 return np.array(result)
def apply(item,name,coef=None):
 f=item['frame'];pred=f[['speed_kmh','flow_vph']].to_numpy().copy()
 if name.startswith('affine'):
  z=pred/[25,600]/np.column_stack([np.ones(len(f)),f.lanes])
  for c in range(2):z[:,c]+=coef[c,0]*z[:,c]+coef[c,1]
  pred=np.maximum(z,0)*[25,600]*np.column_stack([np.ones(len(f)),f.lanes])
 if 'density' in name or name.startswith('qk_'):
  alpha=.25 if name.endswith('025') else .1
  full=item['obs'].copy();full=full.merge(f[['timestamp','station_id']].assign(pv=pred[:,0],pq=pred[:,1]),on=['timestamp','station_id'],how='left',validate='one_to_one')
  mask=full.pv.notna();full.loc[mask,'speed_kmh']=full.loc[mask,'pv'];full.loc[mask,'flow_vph']=full.loc[mask,'pq']
  full=full.sort_values(['station_id','timestamp']).reset_index(drop=True);full['k']=full.flow_vph/full.speed_kmh.clip(lower=1)
  for col in ['k','flow_vph']:
   g=full.groupby('station_id',sort=False);before=g[col].shift(1);after=g[col].shift(-1)
   ts=pd.to_datetime(full.timestamp,utc=True);dt0=(ts-g.timestamp.shift(1).pipe(pd.to_datetime,utc=True)).dt.total_seconds();dt1=(g.timestamp.shift(-1).pipe(pd.to_datetime,utc=True)-ts).dt.total_seconds()
   before=before.where(dt0==300);after=after.where(dt1==300)
   avg=pd.concat([before,full[col],after],axis=1).mean(axis=1)
   full['smooth_'+col]=(1-alpha)*full[col]+alpha*avg
  changed=full[full.pv.notna()].copy();q=changed.smooth_flow_vph if name.startswith('qk_') else changed.flow_vph
  changed['new_q']=q;changed['new_v']=q/changed.smooth_k.clip(lower=1e-9)
  mapped=f[['timestamp','station_id']].merge(changed[['timestamp','station_id','new_v','new_q']],on=['timestamp','station_id'],validate='one_to_one')
  proposed=mapped[['new_v','new_q']].to_numpy();good=np.isfinite(proposed).all(1);pred[good]=proposed[good]
 assert np.isfinite(pred).all() and (pred>=0).all()
 return pred
def metrics(items,outputs):
 rows=[]
 for item,p in zip(items,outputs):
  f=item['frame'];reg=f.mask_regime.iloc[0];err=p-f[['true_v','true_q']].to_numpy();err[:,1]/=f.lanes.to_numpy()
  v=f[['station_id','link_id','timestamp','length_km','true_v','true_q']].copy();v['predN']=p[:,1]/np.maximum(p[:,0],1)*v.length_km;v['trueN']=v.true_q/v.true_v.clip(lower=1)*v.length_km
  v=v.sort_values(['station_id','link_id','timestamp']);g=v.groupby(['station_id','link_id']);dt=(pd.to_datetime(g.timestamp.shift(-1),utc=True)-pd.to_datetime(v.timestamp,utc=True)).dt.total_seconds();dn=(g.predN.shift(-1)-v.predN)-(g.trueN.shift(-1)-v.trueN);good=(dt==300)&dn.notna()
  rows.append(dict(regime=reg,n=len(f),v_sse=float((err[:,0]**2).sum()),q_sse=float((err[:,1]**2).sum()),dn_n=int(good.sum()),dn_sse=float((dn[good]**2).sum())))
 r=pd.DataFrame(rows).groupby('regime').sum(numeric_only=True);vr=np.sqrt(r.v_sse/r.n);qr=np.sqrt(r.q_sse/r.n);dn=np.sqrt(r.dn_sse/r.dn_n.replace(0,np.nan))
 return dict(S_state=float((.54*(1-vr/25).clip(lower=0)+.46*(1-qr/600).clip(lower=0)).mean()),speed_RMSE=float(vr.mean()),flow_per_lane_RMSE=float(qr.mean()),station_target_deltaN_RMSE=float(dn.mean()),target_rows=int(r.n.sum()),adjacent_target_pairs=int(r.dn_n.sum()))
def prepare(panel,dates,route,keys):
 items=[];keysby={date:keys[keys.timestamp.str[:10]==date][KEYS].copy().reset_index(drop=True) for date in dates}
 predictor.keys_for=lambda data,road,split,date:keysby[date].copy()
 net=DATA/'corridors'/panel/'network';fd=pd.read_csv(net/'fd_parameters.csv',dtype={'link_id':str});lanes=fd.groupby('link_id').lanes.mean();length=pd.read_csv(net/'links.csv',dtype={'link_id':str}).drop_duplicates('link_id').set_index('link_id').length_km
 for date in dates:
  check_deadline();folder=WORK/'baseline_predictions';p=folder/(panel+'_'+date+'.parquet')
  if p.exists():pred=pd.read_parquet(p)
  else:pred=predictor.predict(panel,date,route,DATA,folder,'cuda')
  # Supervised scoring/calibration reads public train truth only AFTER predictor completes.
  truthpath=next((DATA/'corridors'/panel/'train/mainline_states').rglob('synthetic_mainline_'+date.replace('-','_')+'.parquet'));truth=pd.read_parquet(truthpath,columns=['timestamp','station_id','link_id','speed_kmh','flow_vph'])
  for c in ['timestamp','station_id','link_id']:truth[c]=truth[c].astype(str)
  truth=truth.rename(columns={'speed_kmh':'true_v','flow_vph':'true_q'});f=pred.merge(truth,on=['timestamp','station_id','link_id'],validate='one_to_one');f['lanes']=f.link_id.map(lanes);f['length_km']=f.link_id.map(length);f['q_lane']=f.flow_vph/f.lanes;f['true_q_lane']=f.true_q/f.lanes
  assert f[['true_v','true_q','lanes','length_km']].notna().all().all()
  masked=next((DATA/'corridors'/panel/'train/mainline_states_masked').rglob('synthetic_mainline_'+date.replace('-','_')+'.parquet'));obs=pd.read_parquet(masked,columns=['timestamp','station_id','speed_kmh','flow_vph'])
  for c in ['timestamp','station_id']:obs[c]=obs[c].astype(str)
  items.append(dict(date=date,frame=f,obs=obs))
  print('prepared',panel,date,len(f),flush=True)
 return items
def main():
 state(status='running',stage='register_dates',next='Resume state_experiment.py if no process holds lock')
 routes=predictor.load_routes();plan={};templates={}
 for panel in routes:
  keys=pd.read_csv(DATA/'task1'/panel/'train/sample_submission_state.csv',usecols=KEYS,dtype=str);templates[panel]=keys
  ds=keys[['timestamp','mask_regime']].assign(date=keys.timestamp.str[:10])[['date','mask_regime']].drop_duplicates();plan[panel]={}
  for phase,lo,hi in [('jan','2031-01-05','2031-01-31'),('feb','2031-02-01','2031-02-28')]:
   selected=[]
   for reg,g in ds[(ds.date>=lo)&(ds.date<=hi)].groupby('mask_regime'):
    dates=sorted(g.date);assert len(dates)>=2;selected.extend([dates[0],dates[-1]])
   assert len(selected)==6;plan[panel][phase]=sorted(selected)
 save(OUT/'frozen_plan.json',dict(dates=plan,candidates=CANDIDATES,penalties=PENALTIES,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='historically exposed development holdouts, NOT blind test',selection='January leave-one-date-out affine penalty; January CV gate accumulation error, maximize S_state; February evaluation only'))
 janrecords=[];policy={}
 for panel,route in routes.items():
  state(status='running',stage='january_cv',panel=panel,next='Let current process finish; caches support resume')
  items=prepare(panel,plan[panel]['jan'],route,templates[panel]);baseout=[apply(x,'baseline') for x in items];base=metrics(items,baseout)
  penalty_results={};cvpreds={}
  for penalty in PENALTIES:
   preds=[apply(item,'affine',affine_fit([v for v in items if v['date']!=item['date']],penalty)) for item in items];cvpreds[penalty]=preds;penalty_results[penalty]=metrics(items,preds)
  penalty=max(PENALTIES,key=lambda p:penalty_results[p]['S_state']);coef=affine_fit(items,penalty);scores={}
  for name in CANDIDATES:
   if name=='affine':preds=cvpreds[penalty]
   elif name=='affine_density_010':preds=[apply(item,name,affine_fit([v for v in items if v['date']!=item['date']],penalty)) for item in items]
   else:preds=[apply(item,name) for item in items]
   scores[name]=metrics(items,preds);janrecords.append(dict(panel=panel,candidate=name,phase='january_cv',penalty=penalty,**scores[name]))
  eligible=[n for n,s in scores.items() if s['station_target_deltaN_RMSE']<=base['station_target_deltaN_RMSE']+1e-12 and s['S_state']>=base['S_state']]
  winner=max(eligible,key=lambda n:scores[n]['S_state'])
  policy[panel]=dict(candidate=winner,penalty=penalty,affine_coefficients=coef.tolist(),january_cv_scores=scores,penalty_cv_scores=penalty_results)
  pd.DataFrame(janrecords).to_csv(OUT/'january_cv.csv',index=False);save(OUT/'policy_partial.json',policy);print('FROZEN ROAD',panel,winner,scores[winner],flush=True)
 save(OUT/'frozen_policy.json',dict(policy=policy,scope='Frozen before any February candidate scoring',utc=datetime.now(timezone.utc).isoformat()))
 febrecords=[]
 for panel,route in routes.items():
  state(status='running',stage='february_evaluation',panel=panel,next='Policy frozen; do not tune on February')
  items=prepare(panel,plan[panel]['feb'],route,templates[panel]);p=policy[panel]
  for name,label in [('baseline','baseline'),(p['candidate'],'frozen_selected')]:
   score=metrics(items,[apply(x,name,np.array(p['affine_coefficients'])) for x in items]);febrecords.append(dict(panel=panel,variant=label,candidate=name,phase='february',**score));print('FEB',panel,label,score,flush=True)
  pd.DataFrame(febrecords).to_csv(OUT/'february_by_panel.csv',index=False)
 summary=pd.DataFrame(febrecords).groupby('variant')[['S_state','speed_RMSE','flow_per_lane_RMSE','station_target_deltaN_RMSE']].mean();summary.to_csv(OUT/'february_summary.csv');print(summary.to_string(),flush=True)
 state(status='stage_complete',stage='state_calibration_complete',next='Read February results honestly; plan distinct new experiment, never call it blind test. Do not upload Kaggle.')

if __name__=='__main__':
 lock=(WORK/'research.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
 try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 except OSError:raise SystemExit('Research lock held; not launching duplicate')
 try:main()
 except Exception as e:
  state(status='failed_or_cutoff',error=repr(e),next='Inspect state_experiment.log and fix/resume existing script; preserve caches');traceback.print_exc();raise
 finally:
  lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
