"""Fresh stress of an observable-only conservative span gate; no hidden labels."""
import common as d
import task4_span_shrink_stress as base
import task4_prior_structure as structure
import numpy as np,pandas as pd,json,msvcrt,traceback,hashlib
from scipy.linalg import solve
from datetime import datetime,timezone
O=d.OUT/'task4_observable_gate_stress';SEED=2026110307
TRUTHS=base.TRUTHS+['sparse_short','sparse_long','localized_OD_block']
NOISES=base.NOISES+['positive_background']

def signal(b,X,seed):
 order=np.random.default_rng(seed).permutation(len(b));fold=np.empty(len(b),int);fold[order]=np.arange(len(b))%2;sse=0.;ref_sse=0.
 for k in range(2):
  train=fold!=k;test=~train;scale=max(float(np.median(b[train&(b>0)])),1e-9);penalty=np.eye(X.shape[1]);penalty[0,0]=0;coef=solve(X[train].T@X[train]+penalty,X[train].T@(b[train]/scale),assume_a='pos');pred=np.maximum(X[test]@coef*scale,0);sse+=float(np.sum((pred-b[test])**2));ref_sse+=float(np.sum((float(b[train].mean())-b[test])**2))
 r2=1-sse/max(ref_sse,1e-12);atom=float((b<=b.min()+1e-12*max(float(np.median(b)),1)).mean());return dict(prior_entry_CV_R2=float(r2),minimum_atom_fraction=atom,gate_enabled=bool(r2>=.2 and atom<=.01))

def key_seed(key):return int.from_bytes(hashlib.sha256((str(SEED)+'/'+key).encode()).digest()[:8],'little')

def draw(paths,Q,level,name,rng):
 if name in base.TRUTHS:return base.truth_draw(paths,Q,level,name,rng)
 f,oi,di=base.truth_draw(paths,Q,level,'flat',rng);span=(di-oi)/max(di)
 if name=='sparse_short':f*=(rng.random(len(f))<.08)*np.exp(-6*span)
 elif name=='sparse_long':f*=(rng.random(len(f))<.08)*(.1+span)
 else:f*=np.where((oi/max(di)<.4)&(di/max(di)>.5)&(di/max(di)<.75),8.,.05)
 loaded=Q@f;assert (loaded>0).any();f*=level/max(float(np.median(loaded[loaded>0])),1e-9);return f,oi,di

def main():
 covariance=d.OUT/'task4_covariance_audit/full_train_frozen_models.json';noise_model=json.loads(covariance.read_text())['models']['path_additive'];protocol=dict(registered_utc=datetime.now(timezone.utc).isoformat(),hypothesis='Enable a fixed5% span prior blend only when supplied prior has reproducible span signal and no large floor atom; challenge with sparse-but-span-predictable and positive-background counterexamples.',seed=SEED,development_evidence='Prior span holdouts and earlier720 artificial cases informed this hypothesis. This fresh stress is not whole-system blind.',gate='Twofold weak-prior-entry CV_R2>=0.2 and fraction within1e-12*max(median,1) of minimum<=0.01; inputs are supplied prior and geometry only. Fixed alpha.05 if enabled,0 otherwise.',methods=['baseline','fixed005','gated005'],truths=TRUTHS,noises=NOISES,levels=['low','high'],replicates=2,case_count=1440,noise_calibration='Use EXACT QR-selected count rows and additive component normalizer from registered train covariance audit; old full-row convention is not pooled. Other noise amplitudes unchanged; positive-background noise adds0.2/0.8 times truth mean with lognormal positive background.',selection='No test tuning of thresholds, alpha or families. Report all failures, every harm, original and three new counterexample shapes. Baseline remains frozen.',source_sha256=d.sha(__file__),solver_source_sha256=d.sha(d.WORK/'task4_span_shrink_stress.py'),feature_source_sha256=d.sha(d.WORK/'task4_prior_structure.py'),noise_model_sha256=d.sha(covariance),scope='Artificial sensitivity and public-input gate diagnostics only. No real hidden path labels, no official S_od/S_ODME claim, no Kaggle uploads.')
 file=O/'protocol.json'
 if file.exists():old=json.loads(file.read_text());protocol['registered_utc']=old['registered_utc'];assert old==protocol
 else:d.save(file,protocol)
 rows_all=[];hashes={};real_rows=[];calibration=[]
 for panel in d.night.predictor.load_routes():
  paths,Q,realc,realb,inputhash=base.load(panel,'train');hashes.update(inputhash);X=structure.features(paths)['span'];penalty=np.eye(X.shape[1]);penalty[0,0]=0;gram=X.T@X+penalty;cache=d.WORK/'task4_covariance_inputs'/panel/'train.npz';z=np.load(cache);idx=z['selected_count_rows'];loaded=(Q@realb)[idx];norm=float(z['normalizers'][1]);sigma=np.sqrt(noise_model['variance'])*float(np.median(loaded[loaded>0]))/np.sqrt(norm);ratio=sigma/max(float(realb.mean()),1e-9);calibration.append(dict(panel=panel,noise_ratio=ratio,independent_count_rows=len(idx),additive_component_normalizer=norm,source_cache_sha256=d.sha(cache)))
  for split in ['train','validation','private']:
   realfile=d.DATA/'task4'/panel/split/'synthetic_weak_prior.csv';prior=pd.read_csv(realfile,dtype={'path_id':str}).set_index('path_id').reindex(paths.path_id).path_flow.to_numpy(float);real_rows.append(dict(panel=panel,split=split,**signal(prior,X,key_seed('real/'+panel+'/'+split)),input_sha256=d.sha(realfile)))
  record=O/(panel+'_cases.csv');rows=pd.read_csv(record).to_dict('records') if record.exists() else [];done={r['case'] for r in rows}
  for truth_name in TRUTHS:
   for noise in NOISES:
    for level in ['low','high']:
     for rep in range(2):
      key=f'{panel}/{truth_name}/{noise}/{level}/{rep}'
      if key in done:continue
      d.check_deadline();d.state(status='running',stage='task4_observable_gate_stress',panel=panel,truth_shape=truth_name,noise=noise,level=level,next='Fresh fixed-threshold counterexamples; sparse span and positive-background cases must remain in test, no tuning or official claim')
      rng=np.random.default_rng(key_seed(key));truth,oi,di=draw(paths,Q,float(np.median(realc)),truth_name,rng);c=Q@truth
      if noise=='additive_clipped':b=np.maximum(truth+rng.normal(0,ratio*truth.mean()*(.5 if level=='low' else 2.),len(truth)),1e-6)
      elif noise=='multiplicative':s=.35 if level=='low' else .8;b=np.maximum(truth*np.exp(rng.normal(-s*s/2,s,len(truth))),1e-6)
      elif noise=='origin_destination':s=.25 if level=='low' else .6;b=np.maximum(truth*np.exp(rng.normal(0,s,oi.max()+1)[oi]+rng.normal(0,s,di.max()+1)[di]-s*s),1e-6)
      else:b=truth*np.exp(rng.normal(-.25**2/2,.25,len(truth)))+(.2 if level=='low' else .8)*truth.mean()*rng.lognormal(-.7**2/2,.7,len(truth))
      observed=signal(b,X,key_seed(key+'/gate'));scale=max(float(np.median(b[b>0])),1e-9);coef=solve(gram,X.T@(b/scale),assume_a='pos');smooth=np.maximum(X@coef*scale,0);baseline,st0=base.ridge(Q,c,b);fixed,st1=base.ridge(Q,c,.95*b+.05*smooth);gated,stgate=(fixed,st1) if observed['gate_enabled'] else (baseline,st0);saved=dict(artificial_truth=truth,artificial_prior=b,artificial_counts=c,baseline=baseline,fixed005=fixed,gated005=gated,span_coefficients=coef)
      casefile=d.WORK/'task4_observable_gate_cases'/panel/(hashlib.sha256(key.encode()).hexdigest()[:16]+'.npz');casefile.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(casefile,**saved);casehash=d.sha(casefile)
      for name,pred,st in [('baseline',baseline,st0),('fixed005',fixed,st1),('gated005',gated,stgate)]:
       row=dict(panel=panel,case=key,seed=key_seed(key),truth_shape=truth_name,noise=noise,level=level,replicate=rep,method=name,**observed,**st,artificial_case_sha256=casehash)
       if st['success'] and np.isfinite(pred).all() and (pred>=0).all():row.update(base.metrics(pred,truth,b,Q,c,paths.destination_zone))
       rows.append(row)
      pd.DataFrame(rows).to_csv(record,index=False)
  rows_all.extend(rows);d.save(O/'input_sha256.json',hashes);pd.DataFrame(real_rows).to_csv(O/'public_prior_gate_diagnostics.csv',index=False);pd.DataFrame(calibration).to_csv(O/'noise_calibration.csv',index=False);print('OBSERVABLE GATE',panel,len(rows),flush=True)
 df=pd.DataFrame(rows_all);df.to_csv(O/'all_cases.csv',index=False);scores=['synthetic_S_od','synthetic_S_link','synthetic_S_dev','synthetic_S_attr','synthetic_S_ODME'];df.groupby('method')[scores].mean().to_csv(O/'summary.csv');df.groupby(['truth_shape','noise','level','method'])[scores].mean().to_csv(O/'by_assumption.csv');reference=df[df.method=='baseline'].set_index('case');risk=[]
 for name in ['baseline','fixed005','gated005']:
  r=df[df.method==name].set_index('case');good=r.success&reference.success;delta=r.loc[good,'synthetic_S_ODME']-reference.loc[good,'synthetic_S_ODME'];od=r.loc[good,'synthetic_S_od']-reference.loc[good,'synthetic_S_od'];risk.append(dict(method=name,total_cases=len(r),successful_cases=int(r.success.sum()),paired_cases=len(delta),mean_synthetic_score_delta=float(delta.mean()),worst_synthetic_score_delta=float(delta.min()),harmed_cases=int((delta<0).sum()),mean_synthetic_OD_delta=float(od.mean()),OD_harmed_cases=int((od<0).sum()),gate_enabled_cases=int(r.gate_enabled.sum())))
 pd.DataFrame(risk).to_csv(O/'paired_risk.csv',index=False);d.state(status='stage_complete',stage='task4_observable_gate_stress_complete',next='Assess fresh sparse-span/positive-background harms and public-prior gates. Do not equate synthetic safety with hidden-score improvement; preserve official baseline and publish verified negative cases.')

if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True);lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:main()
 except Exception as e:d.state(status='failed_or_cutoff',stage='task4_observable_gate_stress',error=repr(e),next='Repair fresh stress execution without changing gate threshold, source normalization, seed or hard counterexamples');traceback.print_exc();raise
 finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
