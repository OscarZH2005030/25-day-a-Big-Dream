"""January-only global selection of temporal regularization of corrections."""
import common as d
import augmentation_fit as a
from pseudo_data import generate
from scipy.linalg import solve_banded
import numpy as np,pandas as pd,json,msvcrt,traceback
from datetime import datetime,timezone

O=d.OUT/'temporal_regularization';LAMBDAS=[0.,.25,1.,4.]

def smooth(frame,corrected,lam):
 # Only keys, baseline values and lane count participate; labels are never read.
 p=frame[['station_id','link_id','timestamp','speed_kmh','flow_vph','lanes']].copy()
 if lam==0:return corrected.copy()
 times=pd.to_datetime(p.timestamp,utc=True).astype('int64').to_numpy();station=p.station_id.astype(str).to_numpy();link=p.link_id.astype(str).to_numpy();idx=np.lexsort((times,link,station));t=times[idx];s=station[idx];l=link[idx]
 edge=(t[1:]-t[:-1]==300000000000)&(s[1:]==s[:-1])&(l[1:]==l[:-1]);w=lam*edge;diag=np.ones(len(p));diag[:-1]+=w;diag[1:]+=w;ab=np.zeros((3,len(p)));ab[1]=diag;ab[0,1:]=-w;ab[2,:-1]=-w
 base=p[['speed_kmh','flow_vph']].to_numpy();scale=np.column_stack([np.ones(len(p)),p.lanes.to_numpy()]);r=(corrected-base)/scale;solution=solve_banded((1,1),ab,r[idx],check_finite=True);out=r.copy();out[idx]=solution
 return np.maximum(base+out*scale,0)

def checks():
 p=pd.DataFrame(dict(station_id=['s']*4,link_id=['l']*4,timestamp=pd.date_range('2031-01-01',periods=4,freq='5min',tz='UTC').astype(str),speed_kmh=[30.]*4,flow_vph=[100.]*4,lanes=[2.]*4));base=p[['speed_kmh','flow_vph']].to_numpy();constant=base+[2.,4.];assert np.allclose(smooth(p,constant,4),constant)
 v=base+np.array([[0,0],[4,8],[0,0],[4,8]]);order=np.array([3,0,2,1]);assert np.allclose(smooth(p.iloc[order],v[order],1),smooth(p,v,1)[order]);assert np.allclose(smooth(p,v,0),v)
 isolated=p.copy();isolated.timestamp=pd.date_range('2031-01-01',periods=4,freq='10min',tz='UTC').astype(str);assert np.allclose(smooth(isolated,v,4),v)
 d.save(O/'runtime_checks.json',dict(constant_correction_preserved=True,row_order_invariant=True,zero_lambda_identity=True,no_edges_across_gaps=True))

def original_phase(phase,dates,lam_values):
 rows=[];daily=[]
 for panel in d.night.predictor.load_routes():
  d.check_deadline();d.state(status='running',stage='temporal_'+phase,panel=panel,next='Global January selection only; preserve state/accumulation tradeoffs, no February retuning')
  rp,rm=a.load_reference(panel);items=[];vv={lam:[] for lam in lam_values}
  for date in dates:
   d.check_deadline();it,X=a.original(panel,date);items.append(it);corr=a.correct(it['frame'],X,rm,rp['alpha'])
   for lam in lam_values:
    value=smooth(it['frame'],corr,lam);vv[lam].append(value);daily.append(dict(panel=panel,date=date,lambda_value=lam,**d.night.metrics([it],[value])))
  rows.extend(dict(panel=panel,lambda_value=lam,**d.night.metrics(items,values)) for lam,values in vv.items());pd.DataFrame(rows).to_csv(O/(phase+'_by_panel.csv'),index=False);pd.DataFrame(daily).to_csv(O/(phase+'_by_date.csv'),index=False)
  print('TEMPORAL',phase,panel,flush=True)
 return pd.DataFrame(rows)

def main():
 checks();protocol=dict(registered_utc=datetime.now(timezone.utc).isoformat(),hypothesis='Temporal jitter in context residual corrections can worsen accumulation-change errors; regularize only corrections on adjacent same-station/link target cells.',method='Solve (I+lambda*graph_laplacian) residual_smooth=residual, independently for speed and per-lane flow. Only five-minute adjacent target pairs, no interpolation through observed gaps. No truth at inference.',lambdas=LAMBDAS,january_dates=[f'2031-01-{i:02}' for i in range(21,32)],february_dates=[f'2031-02-{i:02}' for i in range(1,29)],selection='One global lambda minimizing mean January accumulation-change RMSE, with mean S_state >= context-0.00002, every road S_state >= context-0.0002 and every road accumulation RMSE <= context*1.02. Zero fallback.',auxiliary_seed=2026110232,auxiliary_dates='Median date per regime in February03-27, selected from keys only',source_sha256=d.sha(__file__),context_model_sha256={str(p.relative_to(d.ROOT)):d.sha(p) for p in (d.ROOT/'outputs/overnight/context_candidate/models').rglob('*.txt')},scope='All public train exposed; no official Task3/LWR claim, no blind-system claim; Task4 fixed, no Kaggle upload.')
 d.save(O/'protocol.json',protocol)
 jan=original_phase('january',protocol['january_dates'],LAMBDAS);ref=jan[jan.lambda_value==0].set_index('panel');chosen=0.;best=ref.station_target_deltaN_RMSE.mean();decisions=[]
 for lam in LAMBDAS:
  rr=jan[jan.lambda_value==lam].set_index('panel').loc[ref.index];eligible=bool(rr.S_state.mean()>=ref.S_state.mean()-.00002 and (rr.S_state>=ref.S_state-.0002).all() and (rr.station_target_deltaN_RMSE<=ref.station_target_deltaN_RMSE*1.02).all());objective=float(rr.station_target_deltaN_RMSE.mean());decisions.append(dict(lambda_value=lam,eligible=eligible,objective=objective,S_state=float(rr.S_state.mean())))
  if eligible and objective<best:best=objective;chosen=lam
 pd.DataFrame(decisions).to_csv(O/'january_selection.csv',index=False);d.save(O/'frozen_policy.json',dict(lambda_value=chosen,freeze_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=d.sha(O/'protocol.json')))
 if chosen==0:
  d.save(O/'evaluation_skipped.json',dict(reason='No nonzero lambda passed January criteria; selected operation is exactly existing context. Skip duplicate February/GPU evaluation.',new_february_score=None))
  d.state(status='stage_complete',stage='temporal_regularization_complete',next='Negative January result: no nonzero correction smoother selected; retain failed lambda results, no duplicate GPU evaluations.')
  return
 feb=original_phase('february',protocol['february_dates'],sorted(set([0.,chosen])));feb.groupby('lambda_value')[['S_state','speed_RMSE','flow_per_lane_RMSE','station_target_deltaN_RMSE']].mean().to_csv(O/'february_summary.csv')
 rows=[]
 for panel,route in d.night.predictor.load_routes().items():
  rp,rm=a.load_reference(panel);k=pd.read_csv(d.DATA/'task1'/panel/'train/sample_submission_state.csv',usecols=d.KEYS,dtype=str);v=k.assign(date=k.timestamp.str[:10]);v=v[(v.date>='2031-02-03')&(v.date<='2031-02-27')];dates=sorted(v.groupby('mask_regime').date.apply(lambda z:sorted(set(z))[len(set(z))//2]).tolist())
  for condition in ['point10','block10']:
   items=[];values={n:[] for n in ['baseline','context','regularized']}
   for date in dates:
    d.check_deadline();d.state(status='running',stage='temporal_auxiliary_evaluation',panel=panel,condition=condition,date=date,next='Fixed global lambda; evaluate fresh pseudo-mask seed, never change policy by February results')
    def corr(p,X):return a.correct(p,X,rm,rp['alpha'])
    callbacks=dict(context=corr,regularized=lambda p,X:smooth(p,corr(p,X),chosen));item,_,pred=generate(panel,date,condition,protocol['auxiliary_seed'],d.WORK/'temporal_auxiliary'/panel/condition/date,route,k,callbacks);items.append(item)
    for name in values:values[name].append(pred[name])
   rows.extend(dict(panel=panel,condition=condition,variant=name,lambda_value=chosen,**d.night.metrics(items,pp)) for name,pp in values.items());pd.DataFrame(rows).to_csv(O/'auxiliary_by_panel.csv',index=False)
  print('TEMPORAL AUX',panel,flush=True)
 pd.DataFrame(rows).groupby(['condition','variant'])[['S_state','speed_RMSE','flow_per_lane_RMSE','station_target_deltaN_RMSE']].mean().to_csv(O/'auxiliary_summary.csv')
 d.state(status='stage_complete',stage='temporal_regularization_complete',next='Inspect fixed original and fresh-mask accumulation/state tradeoffs; retain failures and do not claim official improvement.')

if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True);lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:main()
 except Exception as e:d.state(status='failed_or_cutoff',stage='temporal_regularization',error=repr(e),next='Repair execution preserving selection protocol and immutable context');traceback.print_exc();raise
 finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
