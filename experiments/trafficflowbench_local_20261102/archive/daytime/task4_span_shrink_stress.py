"""Frozen conservative span-prior smoothing: artificial assumption stress only."""
import common as d
import task4_prior_structure as structure
import numpy as np,pandas as pd,json,msvcrt,traceback,hashlib
from scipy.linalg import solve
from datetime import datetime,timezone
O=d.OUT/'task4_span_shrink_stress';METHODS={'baseline':0.,'span005':.05,'span010':.1,'span025':.25};TRUTHS=['flat','short','long','origin_destination','sparse','piecewise_span'];NOISES=['additive_clipped','multiplicative','origin_destination'];SEED=2026110306

def load(panel,split):
 net=d.DATA/'corridors'/panel/'network';paths=pd.read_csv(net/'path_set.csv',dtype=str);inc=pd.read_csv(net/'path_link_incidence.csv',dtype=str);folder=d.DATA/'task4'/panel/split;cf=pd.read_csv(folder/'synthetic_link_counts.csv',dtype={'link_id':str});b=pd.read_csv(folder/'synthetic_weak_prior.csv',dtype={'path_id':str}).set_index('path_id').reindex(paths.path_id).path_flow.to_numpy(float);pi={p:i for i,p in enumerate(paths.path_id)};li={p:i for i,p in enumerate(cf.link_id)};sel=inc[inc.link_id.isin(li)];Q=np.zeros((len(cf),len(paths)));Q[sel.link_id.map(li),sel.path_id.map(pi)]=1;hashes={str(p.relative_to(d.DATA)):d.sha(p) for p in [net/'path_set.csv',net/'path_link_incidence.csv',folder/'synthetic_link_counts.csv',folder/'synthetic_weak_prior.csv']};return paths,Q,cf['count'].to_numpy(float),b,hashes

def ridge(Q,c,b):
 lam=.05;y=np.zeros(len(c))
 def fg(y):
  x=np.maximum(b-Q.T@y/lam,0);return .5*lam*(x@x)+c@y+.5*(y@y),y+c-Q@x,x
 for iteration in range(150):
  value,g,x=fg(y)
  if np.abs(g).max()<1e-7:break
  H=(Q*(x>0)/lam)@Q.T+np.eye(len(c));step=solve(H,-g,assume_a='pos');rate=1.;slope=g@step
  for attempt in range(60):
   if fg(y+rate*step)[0]<=value+1e-4*rate*slope+1e-8:break
   rate*=.5
  y+=rate*step
 _,g,x=fg(y);return x,dict(success=bool(np.abs(g).max()<1e-5),dual_gradient_inf=float(np.abs(g).max()),iterations=iteration+1)

def metrics(f,truth,b,Q,c,dest):
 D=float(np.abs(f-b).sum());Dstar=max(float(np.abs(truth-b).sum()),1e-12);ah=pd.Series(f).groupby(dest).sum();at=pd.Series(truth).groupby(dest).sum();od=max(0,1-float(np.abs(f-truth).sum()/truth.sum()));link=max(0,1-float(np.abs(Q@f-c).sum()/max(c.sum(),1e-12)));dev=float(np.exp(-abs(D/Dstar-1)));attr=max(0,1-float(.5*np.abs(ah/ah.sum()-at/at.sum()).sum()));return dict(synthetic_S_od=od,synthetic_S_link=link,synthetic_S_dev=dev,synthetic_S_attr=attr,synthetic_S_ODME=.45*od+.25*link+.15*dev+.15*attr)

def truth_draw(paths,Q,level,name,rng):
 oi=paths.origin_zone.str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy();di=paths.destination_zone.str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy();span=(di-oi)/max(di);f=rng.gamma(2.,.5,len(paths))
 if name=='short':f*=np.exp(-6*span)
 elif name=='long':f*=.1+span
 elif name=='origin_destination':f*=np.exp(rng.normal(0,.6,oi.max()+1)[oi]+rng.normal(0,.6,di.max()+1)[di])
 elif name=='sparse':f*=rng.random(len(paths))<.08
 elif name=='piecewise_span':f*=np.where(span<.15,3.,np.where(span<.5,.3,2.))
 loaded=Q@f;f*=level/max(float(np.median(loaded[loaded>0])),1e-9);return f,oi,di

def main():
 covariance=d.OUT/'task4_covariance_audit/full_train_frozen_models.json';model=json.loads(covariance.read_text())['models']['path_additive'];protocol=dict(registered_utc=datetime.now(timezone.utc).isoformat(),hypothesis='Small span-smoothing of noisy priors may improve path estimates under some error structures; quantify harms under non-span/sparse/correlated counterexamples before considering any real candidate.',seed=SEED,truths=TRUTHS,noises=NOISES,levels=['low','high'],replicates=2,case_count=720,methods=METHODS,algorithm='Fit fixed ridge1 span spline to supplied noisy prior; blend original prior with smoothed prior by fixed0/.05/.1/.25; each solves the same nonnegative official lambda.05 objective.',calibration='Truth draws independent of released b, scaled to released train median link load. Additive noise scale uses train-only count-covariance path variance estimate; low/high multipliers .5/2. Multiplicative logsigma .35/.8; origin/destination correlated log factors .25/.6.',selection='None. All methods and harms reported, no best synthetic model promoted; original official Task4 baseline unchanged.',source_sha256=d.sha(__file__),feature_source_sha256=d.sha(d.WORK/'task4_prior_structure.py'),covariance_model_sha256=d.sha(covariance),scope='Artificial sensitivity, not hidden path validation. Actual priors have positive continuous values with no observed zero atom; clipped Gaussian stress is explicitly a counter-assumption, not claimed true generation. No Kaggle upload.')
 file=O/'protocol.json'
 if file.exists():old=json.loads(file.read_text());protocol['registered_utc']=old['registered_utc'];assert old==protocol
 else:d.save(file,protocol)
 # Verify implementation against the immutable published baseline before stress.
 basefile=d.ROOT/'work/review-repo/experiments/task4/robust_odme_20261027/outputs/task4_path_flow.csv';base=pd.read_csv(basefile,dtype={'panel':str,'departure_time':str,'path_id':str});p,Q,c,b,_=load('D12_I405_N','validation');expected=base[(base.panel=='D12_I405_N')&(base.departure_time=='SYN_PM_VALIDATION')].set_index('path_id').reindex(p.path_id).path_flow.to_numpy(float);actual,status=ridge(Q,c,b);delta=float(np.abs(actual-expected).max());assert status['success'] and delta<1e-7;d.save(O/'solver_crosscheck.json',dict(max_absolute_delta=delta,baseline_sha256=d.sha(basefile),success=True));allrows=[];hashes={}
 for panel in d.night.predictor.load_routes():
  paths,Q,realc,realb,inputhash=load(panel,'train');hashes.update(inputhash);X=structure.features(paths)['span'];penalty=np.eye(X.shape[1]);penalty[0,0]=0;gram=X.T@X+penalty;loaded=Q@realb;countscale=float(np.median(loaded[loaded>0]));norm=float(np.mean(np.sum(Q*Q,axis=1)));sigma=np.sqrt(model['variance'])*countscale/np.sqrt(norm);ratio=sigma/max(float(realb.mean()),1e-9);record=O/(panel+'_cases.csv');rows=pd.read_csv(record).to_dict('records') if record.exists() else [];done={r['case'] for r in rows}
  for truth_name in TRUTHS:
   for noise in NOISES:
    for level in ['low','high']:
     for rep in range(2):
      key=f'{panel}/{truth_name}/{noise}/{level}/{rep}'
      if key in done:continue
      d.check_deadline();d.state(status='running',stage='task4_span_shrink_stress',panel=panel,truth_shape=truth_name,noise=noise,level=level,next='Fixed independent artificial cases, compare full synthetic components and every harm; never replace official baseline from these scores')
      seed=int.from_bytes(hashlib.sha256((str(SEED)+'/'+key).encode()).digest()[:8],'little');rng=np.random.default_rng(seed);truth,oi,di=truth_draw(paths,Q,float(np.median(realc)),truth_name,rng);c=Q@truth
      if noise=='additive_clipped':b=np.maximum(truth+rng.normal(0,ratio*truth.mean()*(.5 if level=='low' else 2.),len(truth)),1e-6)
      elif noise=='multiplicative':s=.35 if level=='low' else .8;b=np.maximum(truth*np.exp(rng.normal(-s*s/2,s,len(truth))),1e-6)
      else:s=.25 if level=='low' else .6;b=np.maximum(truth*np.exp(rng.normal(0,s,oi.max()+1)[oi]+rng.normal(0,s,di.max()+1)[di]-s*s),1e-6)
      scale=max(float(np.median(b[b>0])),1e-9);coef=solve(gram,X.T@(b/scale),assume_a='pos');smooth=np.maximum(X@coef*scale,0);saved=dict(artificial_truth=truth,artificial_prior=b,artificial_counts=c,span_coefficients=coef)
      for name,alpha in METHODS.items():
       pred,st=ridge(Q,c,(1-alpha)*b+alpha*smooth);row=dict(panel=panel,case=key,seed=seed,truth_shape=truth_name,noise=noise,level=level,replicate=rep,method=name,alpha=alpha,calibrated_additive_sigma_ratio=ratio,**st)
       if st['success'] and np.isfinite(pred).all() and (pred>=0).all():row.update(metrics(pred,truth,b,Q,c,paths.destination_zone));saved[name]=pred
       rows.append(row)
      casefile=d.WORK/'task4_span_shrink_cases'/panel/(hashlib.sha256(key.encode()).hexdigest()[:16]+'.npz');casefile.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(casefile,**saved)
      for row in rows[-len(METHODS):]:row['artificial_case_sha256']=d.sha(casefile)
      pd.DataFrame(rows).to_csv(record,index=False)
  allrows.extend(rows);d.save(O/'input_sha256.json',hashes);print('SPAN SHRINK STRESS',panel,len(rows),flush=True)
 df=pd.DataFrame(allrows);df.to_csv(O/'all_cases.csv',index=False);scores=['synthetic_S_od','synthetic_S_link','synthetic_S_dev','synthetic_S_attr','synthetic_S_ODME'];df.groupby('method')[scores].mean().to_csv(O/'summary.csv');df.groupby(['truth_shape','noise','level','method'])[scores].mean().to_csv(O/'by_assumption.csv');ref=df[df.method=='baseline'].set_index('case');risk=[]
 for name in METHODS:
  z=df[df.method==name].set_index('case');valid=z.success&ref.success;delta=z.loc[valid,'synthetic_S_ODME']-ref.loc[valid,'synthetic_S_ODME'];oddelta=z.loc[valid,'synthetic_S_od']-ref.loc[valid,'synthetic_S_od'];risk.append(dict(method=name,total_cases=len(z),successful_cases=int(z.success.sum()),paired_cases=len(delta),mean_synthetic_score_delta=float(delta.mean()),worst_synthetic_score_delta=float(delta.min()),harmed_cases=int((delta<0).sum()),mean_synthetic_OD_delta=float(oddelta.mean()),OD_harmed_cases=int((oddelta<0).sum())))
 pd.DataFrame(risk).to_csv(O/'paired_risk.csv',index=False);d.state(status='stage_complete',stage='task4_span_shrink_stress_complete',next='Inspect conditional full-component gains and worst-case harms, retain all failures. Artificial gains never prove hidden OD improvement; preserve official Task4 baseline and publish only verified findings.')

if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True);lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:main()
 except Exception as e:d.state(status='failed_or_cutoff',stage='task4_span_shrink_stress',error=repr(e),next='Repair fixed stress execution preserving all shape/noise/strength assumptions and negative cases; no Kaggle upload');traceback.print_exc();raise
 finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
