"""Test reusable OD-pair information across public weak priors with shuffled controls."""
import common as d
import task4_prior_structure as structure
import numpy as np
import pandas as pd
import json
import hashlib
import msvcrt
import traceback
from datetime import datetime, timezone
from scipy.linalg import solve

O = d.OUT/'task4_crossmonth_pair_audit'
SEED = 2026110315
SPLITS = ['train','validation','private']


def main():
    O.mkdir(parents=True,exist_ok=True)
    protocol = dict(registered_utc=datetime.now(timezone.utc).isoformat(),hypothesis='Counts leave OD pairings ambiguous; other public months might contain stable pair-specific prior signal beyond geometry. Test held-out target-prior entries against a within-span-bin shuffled source control before considering cross-month denoising.',seed=SEED,folds=5,targets=SPLITS,source_rule='For each target month use only the other two PUBLIC weak-prior vectors. Independently scale each source by its positive median, average, remove a ridge1 span fit using source values only. Same-pair residual is a feature; never use target-month held-out values as inputs.',negative_control='Permute the source residual within ten fixed normalized-span bins using a fixed seed. Same geometry distribution, broken exact OD-pair alignment. All models retained.',target_scale='Median positive target prior in training fold only; all target fitting excludes held-out OD entries.',models=['global','span','span_pair','span_shuffled_pair'],ridge=1.,selection='No model promotion or actual OD candidate. Same previously exposed public priors; weak-prior predictability is not true OD or official-score accuracy.',source_sha256=d.sha(__file__),feature_source_sha256=d.sha(d.WORK/'task4_prior_structure.py'),scope='A cross-month correlation could reflect shared synthetic noise or seasonality; it cannot establish true latent OD structure. Preserve official Task4 baseline, no Kaggle upload.')
    d.save(O/'protocol.json',protocol)
    rows,hashes,modelhashes,folds = [],{},{},[]
    for panel in d.night.predictor.load_routes():
        pathfile=d.DATA/'corridors'/panel/'network/path_set.csv'
        paths=pd.read_csv(pathfile,dtype=str)
        X=structure.features(paths)['span']
        penalty=np.eye(X.shape[1]);penalty[0,0]=0
        gram=X.T@X+penalty
        priors={}
        hashes[str(pathfile.relative_to(d.DATA))]=d.sha(pathfile)
        for split in SPLITS:
            file=d.DATA/'task4'/panel/split/'synthetic_weak_prior.csv'
            priors[split]=pd.read_csv(file,dtype={'path_id':str}).set_index('path_id').reindex(paths.path_id).path_flow.to_numpy(float)
            assert np.isfinite(priors[split]).all() and (priors[split]>=0).all()
            hashes[str(file.relative_to(d.DATA))]=d.sha(file)
        panel_seed=int.from_bytes(hashlib.sha256((str(SEED)+'/'+panel).encode()).digest()[:8],'little')%(2**63)
        rng=np.random.default_rng(panel_seed)
        order=rng.permutation(len(paths));assignment=np.empty(len(paths),int);assignment[order]=np.arange(len(paths))%5
        folds.extend(dict(panel=panel,path_id=path,fold=int(k)) for path,k in zip(paths.path_id,assignment))
        # Save assignment before fitting or examining any held-out outcome.
        pd.DataFrame(folds).to_csv(O/'fold_assignment.csv',index=False)
        origin=paths.origin_zone.str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy()
        dest=paths.destination_zone.str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy()
        bins=np.minimum(((dest-origin)/dest.max()*10).astype(int),9)
        for split in SPLITS:
            sources=[v/max(float(np.median(v[v>0])),1e-9) for name,v in priors.items() if name!=split]
            source=np.mean(sources,axis=0)
            residual=source-X@solve(gram,X.T@source,assume_a='pos')
            shuffled=residual.copy()
            for group in np.unique(bins):
                idx=np.flatnonzero(bins==group);shuffled[idx]=residual[rng.permutation(idx)]
            designs={'global':np.ones((len(paths),1)),'span':X,'span_pair':np.column_stack([X,residual]),'span_shuffled_pair':np.column_stack([X,shuffled])}
            b=priors[split]
            for fold in range(5):
                train=assignment!=fold;test=~train
                scale=max(float(np.median(b[train&(b>0)])),1e-9);y=b/scale
                baseline=float(np.mean((y[test]-y[train].mean())**2))
                for name,design in designs.items():
                    reg=np.eye(design.shape[1]);reg[0,0]=0
                    coef=solve(design[train].T@design[train]+reg,design[train].T@y[train],assume_a='pos')
                    raw=design[test]@coef;prediction=np.maximum(raw,0);mse=float(np.mean((prediction-y[test])**2))
                    folder=d.WORK/'task4_crossmonth_pair_models'/panel/split/name
                    folder.mkdir(parents=True,exist_ok=True);file=folder/f'fold{fold}.npz'
                    np.savez_compressed(file,coef=coef,target_scale=scale,seed=SEED,fold=fold)
                    modelhashes[str(file.relative_to(d.ROOT))]=d.sha(file)
                    rows.append(dict(panel=panel,target_split=split,fold=fold,model=name,heldout_paths=int(test.sum()),normalized_weak_prior_MSE=mse,relative_MSE_reduction_vs_global=1-mse/max(baseline,1e-12),pair_coefficient=float(coef[-1]) if name.startswith('span_') else None,prediction_clipped_fraction=float((raw<0).mean())))
        pd.DataFrame(rows).to_csv(O/'by_panel_split_fold.csv',index=False)
        d.state(status='running',stage='task4_crossmonth_pair_audit',panel=panel,next='Predict withheld public weak-prior entries using other months and a shuffled negative control; not hidden OD labels or a submission.')
    df=pd.DataFrame(rows)
    metrics=['normalized_weak_prior_MSE','relative_MSE_reduction_vs_global','pair_coefficient','prediction_clipped_fraction']
    df.groupby(['target_split','model'])[metrics].mean().to_csv(O/'by_split.csv')
    df.groupby(['panel','target_split','model'])[metrics].mean().to_csv(O/'by_panel_split.csv')
    wide=df.pivot(index=['panel','target_split','fold'],columns='model',values='normalized_weak_prior_MSE')
    wide['pair_MSE_reduction_vs_span']=1-wide.span_pair/wide.span
    wide['shuffled_MSE_reduction_vs_span']=1-wide.span_shuffled_pair/wide.span
    wide.to_csv(O/'paired_comparison.csv')
    wide.groupby('target_split')[['pair_MSE_reduction_vs_span','shuffled_MSE_reduction_vs_span']].mean().to_csv(O/'paired_summary.csv')
    d.save(O/'input_sha256.json',hashes);d.save(O/'model_sha256.json',modelhashes)
    d.state(status='stage_complete',stage='task4_crossmonth_pair_audit_complete',next='Compare pair feature with shuffled control in every road/month, then verify portability; prior prediction cannot prove true OD benefit. No Kaggle upload.')


if __name__=='__main__':
    lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:main()
    except Exception as error:
        d.state(status='failed',stage='task4_crossmonth_pair_audit',error=repr(error),next='Preserve fixed folds and controls; repair implementation without changing evaluation.')
        traceback.print_exc();raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
