"""Does weak prior contain reproducible OD structure? No path truth is used."""
import common as d
import numpy as np,pandas as pd,json,msvcrt,traceback
from scipy.interpolate import BSpline
from scipy.linalg import solve
from datetime import datetime,timezone
O=d.OUT/'task4_prior_structure';SEEDS=[2026110302,2026110303]

def features(paths):
 oi=paths.origin_zone.str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy();di=paths.destination_zone.str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy();assert (di>oi).all();n=max(oi.max(),di.max());span=(di-oi)/n;knots=np.r_[np.zeros(4),[.05,.1,.2,.4,.6,.8],np.ones(4)];spline=BSpline.design_matrix(span,knots,3,extrapolate=False).toarray();origin=np.eye(n+1)[oi];dest=np.eye(n+1)[di];one=np.ones((len(paths),1));return dict(global_mean=one,span=np.column_stack([one,spline]),origin_destination=np.column_stack([one,origin,dest]),span_origin_destination=np.column_stack([one,spline,origin,dest]))

def main():
 panels=list(d.night.predictor.load_routes());protocol=dict(registered_utc=datetime.now(timezone.utc).isoformat(),hypothesis='Independent additive prior noise may hide reusable span and origin/destination signal; evaluate predicting withheld weak-prior entries without their values as features.',split='train only',folds=5,seeds=SEEDS,assignment='Fixed shuffled OD entries, one path per OD; all seeds and folds reported, no best-seed selection',features='Geometry only: normalized positive zone span cubic spline with fixed interior knots .05/.1/.2/.4/.6/.8, origin/destination one-hot, intercept. No prior target or loaded-count-derived feature.',models='Global training mean; span; origin/destination; span+origin/destination. Each nonconstant model reports all ridge penalties1/10/100, no tuning on evaluation entries.',target_scale='Median positive weak-prior value in each training fold only',score='MAE/RMSE on held-out weak prior, normalized by training scale; NOT hidden path accuracy or S_od. Clipped-zero predictions and clipping rate reported.',source_sha256=d.sha(__file__),scope='Tests predictable weak-prior structure only. Independent/clipped noise assumptions can fail; this does not validate denoising real OD, and repeated public inputs are not blind. Frozen Task4 baseline retained, no Kaggle upload.')
 d.save(O/'protocol.json',protocol);rows=[];hashes={};modelhashes={};foldrows=[]
 for pn,panel in enumerate(panels):
  d.check_deadline();d.state(status='running',stage='task4_prior_structure_cv',panel=panel,next='Fixed two-seed five-fold weak-prior prediction, no hidden labels; report all geometric models/penalties and negative results')
  net=d.DATA/'corridors'/panel/'network';pathfile=net/'path_set.csv';priorfile=d.DATA/'task4'/panel/'train/synthetic_weak_prior.csv';paths=pd.read_csv(pathfile,dtype=str);b=pd.read_csv(priorfile,dtype={'path_id':str}).set_index('path_id').reindex(paths.path_id).path_flow.to_numpy(float);assert np.isfinite(b).all() and (b>=0).all();designs=features(paths);hashes[str(pathfile.relative_to(d.DATA))]=d.sha(pathfile);hashes[str(priorfile.relative_to(d.DATA))]=d.sha(priorfile)
  for seed in SEEDS:
   order=np.random.default_rng(seed+pn*100).permutation(len(b));fold=np.empty(len(b),int);fold[order]=np.arange(len(b))%5
   foldrows.extend(dict(panel=panel,path_id=path,seed=seed,fold=int(k)) for path,k in zip(paths.path_id,fold))
   for k in range(5):
    train=fold!=k;test=~train;scale=max(float(np.median(b[train&(b>0)])),1e-9);y=b/scale;reference=float(np.mean(y[train]));baseline_mse=float(np.mean((reference-y[test])**2))
    for kind,X in designs.items():
     for lam in ([0.] if kind=='global_mean' else [1.,10.,100.]):
      d.check_deadline();Xt=X[train];gram=Xt.T@Xt;penalty=np.eye(X.shape[1])*lam;penalty[0,0]=0;coef=solve(gram+penalty,Xt.T@y[train],assume_a='pos');raw=X[test]@coef;pred=np.maximum(raw,0);error=pred-y[test];mse=float(np.mean(error**2));model=kind if kind=='global_mean' else kind+'_ridge'+str(int(lam));folder=d.WORK/'task4_prior_structure_models'/panel/model;folder.mkdir(parents=True,exist_ok=True);weightfile=folder/(str(seed)+'_fold'+str(k)+'.npz');np.savez_compressed(weightfile,coef=coef,scale=scale,seed=seed,fold=k,penalty=lam);modelhashes[str(weightfile.relative_to(d.ROOT))]=d.sha(weightfile);rows.append(dict(panel=panel,seed=seed,fold=k,model=model,heldout_paths=int(test.sum()),normalized_weak_prior_RMSE=float(np.sqrt(mse)),normalized_weak_prior_MAE=float(np.abs(error).mean()),relative_MSE_reduction_vs_global=1-mse/max(baseline_mse,1e-12),prediction_clipped_fraction=float((raw<0).mean())))
   pd.DataFrame(rows).to_csv(O/'by_panel_seed_fold.csv',index=False);d.save(O/'model_sha256.json',modelhashes)
  print('PRIOR STRUCTURE',panel,flush=True)
 metrics=['normalized_weak_prior_RMSE','normalized_weak_prior_MAE','relative_MSE_reduction_vs_global','prediction_clipped_fraction'];df=pd.DataFrame(rows);df.groupby(['panel','model'])[metrics].mean().to_csv(O/'by_panel.csv');df.groupby(['seed','model'])[metrics].mean().to_csv(O/'by_seed.csv');df.groupby('model')[metrics].mean().to_csv(O/'summary.csv');pd.DataFrame(foldrows).to_csv(O/'fold_assignment.csv',index=False);d.save(O/'input_sha256.json',hashes);d.state(status='stage_complete',stage='task4_prior_structure_complete',next='Assess weak-prior geometric signal and all failure models; predictive weak-prior error is not hidden OD score. Only design denoising/constraints after examining assumptions and uncertainty; verified substantive results may be published to authorized GitHub.')

if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True);lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:main()
 except Exception as e:d.state(status='failed_or_cutoff',stage='task4_prior_structure',error=repr(e),next='Repair registered prior-entry holdout without changing folds, seeds, targets or penalty grid');traceback.print_exc();raise
 finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
