"""Can public train residuals support subtracting a positive prior background?"""
import common as d
import task4_span_shrink_stress as base
import numpy as np,pandas as pd,json,msvcrt,traceback
from scipy.linalg import solve_triangular
from scipy.optimize import lsq_linear
from datetime import datetime,timezone
O=d.OUT/'task4_background_mean_audit'
MODELS={'zero':[],'scale':[0],'background':[1],'scale_background':[0,1]}

def record(panel,split):
 paths,Q,c,b,hashes=base.load(panel,split);cache=d.WORK/'task4_covariance_inputs'/panel/(split+'.npz');z=np.load(cache);idx=z['selected_count_rows'];loaded=(Q@b)[idx];scale=float(np.median(loaded[loaded>0]));r=(c[idx]-loaded)/scale;assert np.allclose(r,z['r'],atol=1e-10,rtol=0);X=np.column_stack([loaded/scale,Q[idx].sum(axis=1)*b.mean()/scale]);C=z['Cs'][1]+np.eye(len(r))*1e-6;L=np.linalg.cholesky(C);k=len(r);Xw=solve_triangular(L,X,lower=True)/np.sqrt(k);yw=solve_triangular(L,r,lower=True)/np.sqrt(k);return dict(panel=panel,family=panel.rsplit('_',1)[0],split=split,Xw=Xw,yw=yw,logdet_per_dimension=float(2*np.log(np.diag(L)).sum()/k),b=b,rank=k,hashes=hashes,cache_sha256=d.sha(cache))

def fit(records,name):
 columns=MODELS[name];X=np.concatenate([r['Xw'][:,columns] for r in records]);y=np.concatenate([r['yw'] for r in records]);coef=np.zeros(2);success=True;message='Zero mean fixed'
 if columns:
  bounds=np.array([[-.9,2.],[-3.,3.]])[columns];result=lsq_linear(X,y,bounds=(bounds[:,0],bounds[:,1]),lsq_solver='exact',tol=1e-12);coef[columns]=result.x;success=bool(result.success);message=str(result.message)
 variance=max(float(np.mean([np.sum((r['yw']-r['Xw']@coef)**2) for r in records])),1e-10);loss=.5*(np.log(2*np.pi*variance)+float(np.mean([r['logdet_per_dimension'] for r in records]))+1);design=np.concatenate([r['Xw'] for r in records]);return dict(model=name,scale_beta=float(coef[0]),path_background_gamma=float(coef[1]),variance=variance,train_NLL_per_dimension=float(loss),success=success,message=message,two_component_design_condition=float(np.linalg.cond(design.T@design)))

def evaluate(rec,model):
 coef=np.array([model['scale_beta'],model['path_background_gamma']]);q=float(np.sum((rec['yw']-rec['Xw']@coef)**2))/model['variance'];nll=.5*(np.log(2*np.pi*model['variance'])+rec['logdet_per_dimension']+q);mu=(1+coef[0])*rec['b']+coef[1]*rec['b'].mean();return dict(NLL_per_dimension=float(nll),whitened_quadratic_per_dimension=q,linear_path_mean_negative_fraction=float((mu<0).mean()),linear_path_mean_total_relative_to_prior=float(mu.sum()/rec['b'].sum()))

def main():
 panels=list(d.night.predictor.load_routes());families=sorted(set(p.rsplit('_',1)[0] for p in panels));protocol=dict(registered_utc=datetime.now(timezone.utc).isoformat(),hypothesis='Wrong positive-background prior assumptions harmed converged Gaussian decisions in artificial checks; test whether actual public train count/prior residuals support a separate path-independent background component beyond global prior scaling.',mean='(c-Ab)/s = beta*(Ab)/s + gamma*(A1)*mean(b)/s + error; s=median positive prior-loaded independent counts. gamma<0 would support subtracting background, but is not imposed.',bounds=dict(beta=[-.9,2.],gamma=[-3.,3.]),covariance='Fixed normalized additive-path covariance from prior audit, no retuning of covariance components; variance profiled on training roads.',models=MODELS,validation='Five leave-one-corridor-family folds on train only; all four models and coefficients retained. Full-train models frozen before supplied validation/private inputs are evaluated.',selection='No path-flow candidate or hidden score is produced. Linear corrected means may be negative; report this as evidence against direct use, not silently clip to improve likelihood.',source_sha256=d.sha(__file__),prior_covariance_protocol_sha256=d.sha(d.OUT/'task4_covariance_audit/protocol.json'),scope='Tests an observable-space mean assumption, not identification of nullspace path errors or official OD gain. Frozen baseline unchanged, no Kaggle upload.')
 d.save(O/'protocol.json',protocol);records=[];hashes={}
 for panel in panels:
  d.state(status='running',stage='task4_background_mean_inputs',panel=panel,next='Train-only evidence for background correction, no hidden path labels or real candidate promotion');r=record(panel,'train');records.append(r);hashes.update(r['hashes'])
 d.save(O/'train_input_sha256.json',hashes);models={};rows=[]
 for family in families:
  train=[r for r in records if r['family']!=family];test=[r for r in records if r['family']==family]
  for name in MODELS:
   d.check_deadline();d.state(status='running',stage='task4_background_mean_cv',family=family,model=name,next='Held-out corridor likelihood, retain coefficient instability and negative corrected-mean fractions');model=fit(train,name);models[family+'/'+name]=model
   for r in test:rows.append(dict(panel=r['panel'],family=family,model=name,**evaluate(r,model)))
  pd.DataFrame(rows).to_csv(O/'family_cv_by_panel.csv',index=False);d.save(O/'family_cv_models.json',models)
 pd.DataFrame(rows).groupby('model')[['NLL_per_dimension','whitened_quadratic_per_dimension','linear_path_mean_negative_fraction']].mean().to_csv(O/'family_cv_summary.csv');frozen={name:fit(records,name) for name in MODELS};d.save(O/'full_train_models.json',dict(models=frozen,frozen_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=d.sha(O/'protocol.json')));rows=[];hashes={}
 for panel in panels:
  for split in ['validation','private']:
   d.state(status='running',stage='task4_background_mean_transfer',panel=panel,split=split,next='All fixed models on public input residuals; not hidden OD scoring');r=record(panel,split);hashes.update(r['hashes'])
   for name,model in frozen.items():rows.append(dict(panel=panel,split=split,model=name,**evaluate(r,model)))
 pd.DataFrame(rows).to_csv(O/'transfer_by_panel.csv',index=False);pd.DataFrame(rows).groupby(['split','model'])[['NLL_per_dimension','whitened_quadratic_per_dimension','linear_path_mean_negative_fraction']].mean().to_csv(O/'transfer_summary.csv');d.save(O/'transfer_input_sha256.json',hashes);d.state(status='stage_complete',stage='task4_background_mean_audit_complete',next='Check cross-family support and sign stability for background correction, reject unsupported extrapolation. No hidden improvement claimed, official baseline untouched.')

if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True);lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:main()
 except Exception as e:d.state(status='failed_or_cutoff',stage='task4_background_mean_audit',error=repr(e),next='Repair deterministic mean audit without changing held-out families, coefficient bounds or covariance selection.');traceback.print_exc();raise
 finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
