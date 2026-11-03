"""Is cross-month weak-prior predictability more than origin/destination effects?"""
import common as d
import task4_prior_structure as structure
import numpy as np
import pandas as pd
import hashlib
import msvcrt
import traceback
from scipy.linalg import solve
from datetime import datetime, timezone

O=d.OUT/'task4_pair_marginal_controls'
SEED=2026110316
SPLITS=['train','validation','private']


def main():
    O.mkdir(parents=True,exist_ok=True)
    d.save(O/'protocol.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),seed=SEED,folds=5,hypothesis='Initial cross-month pair feature predicts held-out public priors beyond span, but might only capture shared origin/destination margins. Residualize both source and target geometry against span plus origin/destination effects, retain shuffled-pair control.',models=['geometry','geometry_pair','geometry_shuffled_pair'],geometry='Same fixed span spline plus full origin/destination one-hot, ridge1, unpenalized intercept.',source='Other two public months only, each normalized by its positive median; mean source minus ridge1 geometry fit to source values. No target held-out values in features.',negative_control='Fixed permutation of source residual within ten normalized-span bins.',target_scale='Positive target-prior training-fold median only.',evaluation='All three public months, five fixed shuffled OD-entry folds; outcomes are supplied weak-prior entries, never true OD. Follow-up on already exposed data, not independent whole-system validation.',source_sha256=d.sha(__file__),feature_source_sha256=d.sha(d.WORK/'task4_prior_structure.py'),scope='Even repeatable pair signal could be common synthetic noise; no actual OD candidate, official baseline unchanged, no Kaggle upload.'))
    rows,hashes,modelhashes,folds=[],{},{},[]
    for panel in d.night.predictor.load_routes():
        file=d.DATA/'corridors'/panel/'network/path_set.csv';paths=pd.read_csv(file,dtype=str);hashes[str(file.relative_to(d.DATA))]=d.sha(file)
        X=structure.features(paths)['span_origin_destination'];penalty=np.eye(X.shape[1]);penalty[0,0]=0;gram=X.T@X+penalty
        priors={}
        for split in SPLITS:
            file=d.DATA/'task4'/panel/split/'synthetic_weak_prior.csv';priors[split]=pd.read_csv(file,dtype={'path_id':str}).set_index('path_id').reindex(paths.path_id).path_flow.to_numpy(float);hashes[str(file.relative_to(d.DATA))]=d.sha(file)
        panel_seed=int.from_bytes(hashlib.sha256((str(SEED)+'/'+panel).encode()).digest()[:8],'little')%(2**63);rng=np.random.default_rng(panel_seed)
        order=rng.permutation(len(paths));assignment=np.empty(len(paths),int);assignment[order]=np.arange(len(paths))%5
        folds.extend(dict(panel=panel,path_id=p,fold=int(k)) for p,k in zip(paths.path_id,assignment));pd.DataFrame(folds).to_csv(O/'fold_assignment.csv',index=False)
        origin=paths.origin_zone.str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy();dest=paths.destination_zone.str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy();bins=np.minimum(((dest-origin)/dest.max()*10).astype(int),9)
        for split in SPLITS:
            source=np.mean([v/max(float(np.median(v[v>0])),1e-9) for name,v in priors.items() if name!=split],axis=0)
            residual=source-X@solve(gram,X.T@source,assume_a='pos');shuffled=residual.copy()
            for group in np.unique(bins):
                idx=np.flatnonzero(bins==group);shuffled[idx]=residual[rng.permutation(idx)]
            designs={'geometry':X,'geometry_pair':np.column_stack([X,residual]),'geometry_shuffled_pair':np.column_stack([X,shuffled])};b=priors[split]
            for fold in range(5):
                train=assignment!=fold;test=~train;scale=max(float(np.median(b[train&(b>0)])),1e-9);y=b/scale
                for name,design in designs.items():
                    reg=np.eye(design.shape[1]);reg[0,0]=0;coef=solve(design[train].T@design[train]+reg,design[train].T@y[train],assume_a='pos');raw=design[test]@coef;prediction=np.maximum(raw,0)
                    folder=d.WORK/'task4_pair_marginal_models'/panel/split/name;folder.mkdir(parents=True,exist_ok=True);file=folder/f'fold{fold}.npz';np.savez_compressed(file,coef=coef,target_scale=scale,seed=SEED,fold=fold);modelhashes[str(file.relative_to(d.ROOT))]=d.sha(file)
                    rows.append(dict(panel=panel,target_split=split,fold=fold,model=name,heldout_paths=int(test.sum()),normalized_weak_prior_MSE=float(np.mean((prediction-y[test])**2)),pair_coefficient=float(coef[-1]) if name!='geometry' else None,prediction_clipped_fraction=float((raw<0).mean())))
        pd.DataFrame(rows).to_csv(O/'by_panel_split_fold.csv',index=False);d.state(status='running',stage='task4_pair_marginal_controls',panel=panel,next='Check source pair residual after removing span and origin/destination effects; weak-prior prediction only, not hidden OD.')
    df=pd.DataFrame(rows);df.groupby(['target_split','model'])[['normalized_weak_prior_MSE','pair_coefficient','prediction_clipped_fraction']].mean().to_csv(O/'by_split.csv')
    wide=df.pivot(index=['panel','target_split','fold'],columns='model',values='normalized_weak_prior_MSE');wide['pair_MSE_reduction_vs_geometry']=1-wide.geometry_pair/wide.geometry;wide['shuffled_MSE_reduction_vs_geometry']=1-wide.geometry_shuffled_pair/wide.geometry;wide.to_csv(O/'paired_comparison.csv');wide.groupby('target_split')[['pair_MSE_reduction_vs_geometry','shuffled_MSE_reduction_vs_geometry']].mean().to_csv(O/'paired_summary.csv')
    d.save(O/'input_sha256.json',hashes);d.save(O/'model_sha256.json',modelhashes);d.state(status='stage_complete',stage='task4_pair_marginal_controls_complete',next='Audit pair-vs-shuffled coefficients and all roads, then independently reproduce public-prior studies before claiming reusable pairing signal; never equate prior prediction with true OD gain.')


if __name__=='__main__':
    lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:main()
    except Exception as error:
        d.state(status='failed',stage='task4_pair_marginal_controls',error=repr(error),next='Preserve fixed folds, negative control and failure.');traceback.print_exc();raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
