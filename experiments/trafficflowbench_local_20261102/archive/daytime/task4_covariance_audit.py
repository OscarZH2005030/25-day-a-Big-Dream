"""Train-only prior-error covariance hypotheses; leave-one-family predictive check."""
import common as d
import numpy as np,pandas as pd,json,msvcrt,traceback
from scipy.linalg import qr,cho_factor,cho_solve
from scipy.optimize import minimize
from datetime import datetime,timezone
O=d.OUT/'task4_covariance_audit'
COMPONENTS=['link_independent','path_additive','path_multiplicative','origin_correlated','destination_correlated']
FIXED={'link_iid':[1,0,0,0,0],'path_additive':[0,1,0,0,0],'path_multiplicative':[0,0,1,0,0],'origin_destination':[0,0,0,.5,.5]}

def record(panel,split):
 net=d.DATA/'corridors'/panel/'network';folder=d.DATA/'task4'/panel/split;paths=pd.read_csv(net/'path_set.csv',dtype=str);inc=pd.read_csv(net/'path_link_incidence.csv',dtype=str);cf=pd.read_csv(folder/'synthetic_link_counts.csv',dtype={'link_id':str});b=pd.read_csv(folder/'synthetic_weak_prior.csv',dtype={'path_id':str}).set_index('path_id').reindex(paths.path_id).path_flow.to_numpy(float);pi={v:i for i,v in enumerate(paths.path_id)};li={v:i for i,v in enumerate(cf.link_id)};sel=inc[inc.link_id.isin(li)];A=np.zeros((len(cf),len(paths)));A[sel.link_id.map(li).to_numpy(),sel.path_id.map(pi).to_numpy()]=1
 _,R,piv=qr(A.T,mode='economic',pivoting=True);rank=int((np.abs(np.diag(R))>max(abs(R[0,0]),1)*1e-10).sum());idx=piv[:rank];A=A[idx];c=cf['count'].to_numpy(float)[idx];loaded=A@b;scale=max(float(np.median(loaded[loaded>0])),1e-9);bs=b/max(float(np.median(b[b>0])),1e-9);weighted=A*bs;oi=pd.factorize(paths.origin_zone,sort=True)[0];di=pd.factorize(paths.destination_zone,sort=True)[0];U=weighted@np.eye(oi.max()+1)[oi];V=weighted@np.eye(di.max()+1)[di];raw=[np.eye(rank),A@A.T,weighted@weighted.T,U@U.T,V@V.T];normalizers=[float(np.trace(C)/rank) for C in raw];Cs=np.stack([C/max(z,1e-12) for C,z in zip(raw,normalizers)]);r=(c-loaded)/scale;l=loaded/scale;correlations=[]
 for i in range(len(Cs)):
  for j in range(i+1,len(Cs)):correlations.append(dict(panel=panel,split=split,component1=COMPONENTS[i],component2=COMPONENTS[j],Frobenius_cosine=float(np.sum(Cs[i]*Cs[j])/np.linalg.norm(Cs[i])/np.linalg.norm(Cs[j]))))
 hashes={str(p.relative_to(d.DATA)):d.sha(p) for p in [net/'path_set.csv',net/'path_link_incidence.csv',folder/'synthetic_link_counts.csv',folder/'synthetic_weak_prior.csv']};cache=d.WORK/'task4_covariance_inputs'/panel/(split+'.npz');cache.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(cache,r=r,l=l,Cs=Cs,selected_count_rows=idx,normalizers=normalizers);return dict(panel=panel,family=panel.rsplit('_',1)[0],split=split,r=r,l=l,Cs=Cs,rank=rank,hashes=hashes,correlations=correlations)

def factors(rec,w):
 C=np.tensordot(w,rec['Cs'],axes=1)+np.eye(rec['rank'])*1e-6;factor=cho_factor(C,lower=True,check_finite=False);logdet=float(2*np.log(np.diag(factor[0])).sum());return factor,logdet

def profile(records,w):
 work=[];num=0.;den=0.
 for rec in records:
  factor,logdet=factors(rec,w);sol=cho_solve(factor,np.column_stack([rec['r'],rec['l']]),check_finite=False);k=rec['rank'];num+=float(rec['l']@sol[:,0])/k;den+=float(rec['l']@sol[:,1])/k;work.append((rec,factor,logdet))
 beta=float(np.clip(num/max(den,1e-12),-.9,2.));quad=[];logs=[]
 for rec,factor,logdet in work:
  e=rec['r']-beta*rec['l'];quad.append(float(e@cho_solve(factor,e,check_finite=False))/rec['rank']);logs.append(logdet/rec['rank'])
 variance=max(float(np.mean(quad)),1e-10);loss=.5*(np.log(2*np.pi*variance)+float(np.mean(logs))+float(np.mean(quad))/variance);return float(loss),beta,variance

def mixture(theta):
 v=np.r_[theta,0.];v=np.exp(v-v.max());v/=v.sum();return .0001+.9995*v

def fit(records,name):
 trials=[]
 if name in FIXED:w=np.array(FIXED[name],float)
 else:
  starts=[np.zeros(4),np.array([-2.,2.,-2.,-2.]),np.array([-2.,-2.,2.,-2.])];results=[]
  for initial in starts:
   result=minimize(lambda x:profile(records,mixture(x))[0],initial,method='L-BFGS-B',bounds=[(-10,10)]*4,options=dict(maxiter=80,ftol=1e-8));trials.append(dict(success=bool(result.success),message=str(result.message),objective=float(result.fun),iterations=int(result.nit)));results.append(result)
  best=min(results,key=lambda x:x.fun);w=mixture(best.x)
 loss,beta,variance=profile(records,w);return dict(name=name,weights={k:float(v) for k,v in zip(COMPONENTS,w)},beta=beta,variance=variance,train_NLL_per_dimension=loss,optimizer_trials=trials)

def evaluate(rec,model):
 w=np.array([model['weights'][k] for k in COMPONENTS]);factor,logdet=factors(rec,w);e=rec['r']-model['beta']*rec['l'];q=float(e@cho_solve(factor,e,check_finite=False))/rec['rank']/model['variance'];nll=.5*(np.log(2*np.pi*model['variance'])+logdet/rec['rank']+q);return dict(NLL_per_dimension=float(nll),whitened_quadratic_per_dimension=q,normalized_count_residual_RMSE=float(np.sqrt(np.mean(e**2))))

def main():
 panels=list(d.night.predictor.load_routes());families=sorted(set(p.rsplit('_',1)[0] for p in panels));protocol=dict(registered_utc=datetime.now(timezone.utc).isoformat(),hypothesis='Public train count/prior residuals may distinguish independent path errors from origin/destination-correlated errors; transferable covariance assumptions can guide demand inference, while nullspace extrapolation remains an assumption.',panels=panels,train_split='train',cross_validation='Leave one corridor family (both directions) out, five folds; no evaluation count inputs used for fitting.',models=list(FIXED)+['learned_mixture'],components=COMPONENTS,normalization='Prior-loaded median count scale only; independent QR-selected count equations; each covariance component normalized to mean diagonal1; each road contributes equally per dimension.',mean='One shared prior-loaded global scale correction beta, profiled GLS and clipped[-.9,2]',variance='Shared residual variance profiled on training roads only',mixture='Five nonnegative normalized covariance components, minimum weight0.0001 each; deterministic three initializations; L-BFGS-B maxiter80; training objective selects optimizer start.',evaluation='After all train CV and full-train fits freeze, report predictive count/prior-residual log density on supplied validation/private inputs for every model; no path truth, no hidden score, no model promotion.',source_sha256=d.sha(__file__),seed='No randomness; deterministic linear algebra and optimizer initializations',caveat='Observed count covariance cannot identify arbitrary nullspace error. Counts/prior are public and previously exposed. Predictive likelihood is evidence about error assumptions only, not S_od/S_ODME. Official frozen baseline retained, no uploads.')
 d.save(O/'protocol.json',protocol);records=[]
 for panel in panels:
  d.check_deadline();d.state(status='running',stage='task4_covariance_train_inputs',panel=panel,next='Public train prior/count error covariance; five family-held-out checks, no hidden path labels');records.append(record(panel,'train'))
 d.save(O/'train_input_sha256.json',{k:v for r in records for k,v in r['hashes'].items()});pd.DataFrame([z for r in records for z in r['correlations']]).to_csv(O/'train_component_similarity.csv',index=False);rows=[];models={}
 for family in families:
  train=[r for r in records if r['family']!=family];test=[r for r in records if r['family']==family]
  for name in protocol['models']:
   d.check_deadline();d.state(status='running',stage='task4_covariance_family_cv',family=family,model=name,next='Fit only other four families; evaluate held-out prior/count residual distribution, not hidden path score');model=fit(train,name);models[family+'/'+name]=model
   for rec in test:rows.append(dict(family=family,panel=rec['panel'],model=name,rank=rec['rank'],**evaluate(rec,model)))
   pd.DataFrame(rows).to_csv(O/'family_cv_by_panel.csv',index=False);d.save(O/'family_cv_models.json',models);print('COVARIANCE CV',family,name,flush=True)
 pd.DataFrame(rows).groupby('model')[['NLL_per_dimension','whitened_quadratic_per_dimension','normalized_count_residual_RMSE']].mean().to_csv(O/'family_cv_summary.csv');frozen={name:fit(records,name) for name in protocol['models']};d.save(O/'full_train_frozen_models.json',dict(models=frozen,frozen_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=d.sha(O/'protocol.json'),scope='No validation/private data in fitting; no path-flow model is submitted.'));rows=[];hashes={}
 for panel in panels:
  for split in ['validation','private']:
   d.check_deadline();d.state(status='running',stage='task4_covariance_transfer',panel=panel,split=split,next='Evaluate every fixed covariance model; no model/road choice by evaluation result');rec=record(panel,split);hashes.update(rec['hashes'])
   for name,model in frozen.items():rows.append(dict(panel=panel,split=split,model=name,rank=rec['rank'],**evaluate(rec,model)))
   pd.DataFrame(rows).to_csv(O/'transfer_by_panel.csv',index=False);d.save(O/'transfer_input_sha256.json',hashes)
 pd.DataFrame(rows).groupby(['split','model'])[['NLL_per_dimension','whitened_quadratic_per_dimension','normalized_count_residual_RMSE']].mean().to_csv(O/'transfer_summary.csv');d.state(status='stage_complete',stage='task4_covariance_audit_complete',next='Assess held-out-family predictive support and covariance collinearity before structured demand inference; do not interpret likelihood as hidden path improvement or replace frozen baseline.')

if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True);lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:main()
 except Exception as e:d.state(status='failed_or_cutoff',stage='task4_covariance_audit',error=repr(e),next='Repair pre-registered covariance audit without using hidden truth or selecting on evaluation counts');traceback.print_exc();raise
 finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
