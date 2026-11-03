"""Frozen span/OD ridge recipe on future public-prior entry holdouts."""
import common as d
import task4_prior_structure as source
import numpy as np,pandas as pd,json,msvcrt,traceback
from scipy.linalg import solve
from datetime import datetime,timezone
O=d.OUT/'task4_prior_structure_transfer';SEED=2026110305

def main():
 protocol=dict(registered_utc=datetime.now(timezone.utc).isoformat(),hypothesis='Geometry signal found in train weak priors transfers to independently generated public validation/private priors with a train-frozen recipe.',splits=['validation','private'],seed=SEED,folds=5,models=['global_mean','span_ridge1','origin_destination_ridge1','span_origin_destination_ridge1'],selection='No evaluation tuning: features/knots and ridge1 fixed from completed train study, every road and split kept; no best-fold or best-road selection.',targets='Withheld entries of published WEAK PRIOR, not hidden path-flow truth. Other prior entries are available input; fold-only positive median normalization.',train_study_protocol_sha256=d.sha(source.O/'protocol.json'),feature_source_sha256=d.sha(d.WORK/'task4_prior_structure.py'),source_sha256=d.sha(__file__),scope='Recipe transfer only, not independent system blind or S_od/S_ODME validation. Task4 baseline unchanged, no Kaggle uploads.')
 d.save(O/'protocol.json',protocol);rows=[];hashes={};modelhashes={};assignments=[]
 for pn,panel in enumerate(d.night.predictor.load_routes()):
  paths=pd.read_csv(d.DATA/'corridors'/panel/'network/path_set.csv',dtype=str);designs=source.features(paths)
  for si,split in enumerate(protocol['splits']):
   d.check_deadline();d.state(status='running',stage='task4_prior_structure_transfer',panel=panel,split=split,next='Frozen train-selected span/ridge recipe on withheld public weak-prior entries, not path truth')
   file=d.DATA/'task4'/panel/split/'synthetic_weak_prior.csv';b=pd.read_csv(file,dtype={'path_id':str}).set_index('path_id').reindex(paths.path_id).path_flow.to_numpy(float);order=np.random.default_rng(SEED+pn*100+si).permutation(len(b));fold=np.empty(len(b),int);fold[order]=np.arange(len(b))%5;assignments.extend(dict(panel=panel,split=split,path_id=p,fold=int(k)) for p,k in zip(paths.path_id,fold));hashes[str(file.relative_to(d.DATA))]=d.sha(file)
   for k in range(5):
    train=fold!=k;test=~train;scale=max(float(np.median(b[train&(b>0)])),1e-9);y=b/scale;ref=float(np.mean(y[train]));refmse=float(np.mean((ref-y[test])**2))
    for kind,X in designs.items():
     lam=0 if kind=='global_mean' else 1;penalty=np.eye(X.shape[1])*lam;penalty[0,0]=0;coef=solve(X[train].T@X[train]+penalty,X[train].T@y[train],assume_a='pos');raw=X[test]@coef;error=np.maximum(raw,0)-y[test];model=kind if lam==0 else kind+'_ridge1';mse=float(np.mean(error**2));weightfile=d.WORK/'task4_prior_structure_transfer_models'/panel/split/model/(str(k)+'.npz');weightfile.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(weightfile,coef=coef,scale=scale,seed=SEED,fold=k);modelhashes[str(weightfile.relative_to(d.ROOT))]=d.sha(weightfile);rows.append(dict(panel=panel,split=split,fold=k,model=model,normalized_weak_prior_RMSE=float(np.sqrt(mse)),normalized_weak_prior_MAE=float(np.abs(error).mean()),relative_MSE_reduction_vs_global=1-mse/max(refmse,1e-12),clipped_fraction=float((raw<0).mean()),heldout_paths=int(test.sum())))
   pd.DataFrame(rows).to_csv(O/'by_panel_split_fold.csv',index=False);d.save(O/'model_sha256.json',modelhashes);print('PRIOR STRUCTURE TRANSFER',panel,split,flush=True)
 metrics=['normalized_weak_prior_RMSE','normalized_weak_prior_MAE','relative_MSE_reduction_vs_global','clipped_fraction'];df=pd.DataFrame(rows);df.groupby(['panel','split','model'])[metrics].mean().to_csv(O/'by_panel.csv');df.groupby(['split','model'])[metrics].mean().to_csv(O/'summary.csv');pd.DataFrame(assignments).to_csv(O/'fold_assignment.csv',index=False);d.save(O/'input_sha256.json',hashes);d.state(status='stage_complete',stage='task4_prior_structure_transfer_complete',next='Assess frozen weak-prior structure transfer; if supported, preregister conservative denoising/count-conditioning stress, still no real hidden path score or Kaggle upload.')

if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True);lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:main()
 except Exception as e:d.state(status='failed_or_cutoff',stage='task4_prior_structure_transfer',error=repr(e),next='Repair transfer without changing frozen ridge/geometry recipe or held-out entries');traceback.print_exc();raise
 finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
