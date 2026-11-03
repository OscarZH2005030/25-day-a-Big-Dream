"""Separate exact OD interaction from regularized marginal feature control."""
import common as d
import task4_span_shrink_stress as base
import task4_prior_structure as structure
import task4_crossmonth_shared_noise_qr as stable
import numpy as np
import pandas as pd
import hashlib
import msvcrt
import traceback
from scipy.linalg import qr
from datetime import datetime,timezone

O=d.OUT/'task4_exact_pair_audit';SEED=2026110323;SPLITS=['train','validation','private']


def basis(X):
    Q,R,_=qr(X,mode='economic',pivoting=True)
    rank=int((np.abs(np.diag(R))>max(abs(R[0,0]),1)*1e-10).sum())
    return Q[:,:rank],rank


def main():
    O.mkdir(parents=True,exist_ok=True)
    d.save(O/'protocol.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),seed=SEED,folds=5,hypothesis='Count-source adjustment might only alter origin/destination effects; prior ridge1 marginal controls were not an exact orthogonal removal. Audit conditional count rank and isolate exact geometry-orthogonal cross-input information.',counts='Union all three Task4 incidence row sets; compare their span with exact origin/destination marginal row space. No hidden truth used.',features='Same source priors and same target folds for geometry baseline, ridge1-source pair, exactly geometry-orthogonal source pair, and shuffled-then-reprojected source pair.',projection='Pivoted QR basis of full span+origin/destination geometry with rank threshold1e-10 relative; exact source residual=(I-QQ.T)*mean normalized source prior. Within-span-bin shuffled residual reprojected into same nullspace.',target_fit='Unchanged augmented QR ridge1, unpenalized intercept, target training-fold median only. Other two public priors as source; held-out target entries not inputs.',scope='All inputs exposed. A regularized marginal predictor is distinct from exact interaction removal. Target is supplied weak prior, not hidden OD; all four models retained, no official candidate or Kaggle upload.',source_sha256=d.sha(__file__),geometry_source_sha256=d.sha(d.WORK/'task4_prior_structure.py'),numerical_source_sha256=d.sha(stable.__file__)))
    rows,ranks,folds,hashes,modelhashes=[],[],[],{},{}
    for panel in d.night.predictor.load_routes():
        priors={};As=[]
        for split in SPLITS:
            paths,A,_,b,inputs=base.load(panel,split);hashes.update(inputs);priors[split]=b;As.append(A)
        origin=paths.origin_zone.str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy();dest=paths.destination_zone.str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy()
        M=np.column_stack([np.eye(origin.max()+1)[origin],np.eye(dest.max()+1)[dest]])
        A=np.unique(np.vstack(As),axis=0);QM,rankM=basis(M);_,rankA=basis(A.T);_,rankJoint=basis(np.column_stack([M,A.T]))
        residual=A-(A@QM)@QM.T
        ranks.append(dict(panel=panel,paths=len(paths),unique_count_rows=len(A),count_rank=rankA,marginal_rank=rankM,joint_rank=rankJoint,conditional_count_rank=rankJoint-rankM,count_from_margins_relative_Frobenius_error=float(np.linalg.norm(residual)/max(np.linalg.norm(A),1e-12)),count_from_margins_max_abs_error=float(np.max(np.abs(residual)))))
        X=structure.features(paths)['span_origin_destination'];QX,_=basis(X)
        seed=int.from_bytes(hashlib.sha256((str(SEED)+'/'+panel).encode()).digest()[:8],'little')%(2**63);rng=np.random.default_rng(seed)
        order=rng.permutation(len(paths));fold=np.empty(len(paths),int);fold[order]=np.arange(len(paths))%5
        folds.extend(dict(panel=panel,path_id=p,fold=int(k)) for p,k in zip(paths.path_id,fold));pd.DataFrame(folds).to_csv(O/'fold_assignment.csv',index=False)
        bins=np.minimum(((dest-origin)/dest.max()*10).astype(int),9)
        for split in SPLITS:
            source=np.mean([b/max(float(np.median(b[b>0])),1e-9) for name,b in priors.items() if name!=split],axis=0)
            ridge_residual=source-X@stable.ridge_qr(X,source);exact=source-QX@(QX.T@source);shuffled=exact.copy()
            for group in np.unique(bins):
                idx=np.flatnonzero(bins==group);shuffled[idx]=exact[rng.permutation(idx)]
            shuffled-=QX@(QX.T@shuffled)
            assert np.linalg.norm(QX.T@exact)<1e-9*(1+np.linalg.norm(source))
            assert np.linalg.norm(QX.T@shuffled)<1e-9*(1+np.linalg.norm(source))
            designs={'geometry':X,'ridge_pair':np.column_stack([X,ridge_residual]),'exact_pair':np.column_stack([X,exact]),'exact_shuffled_pair':np.column_stack([X,shuffled])};b=priors[split]
            for k in range(5):
                train=fold!=k;test=~train;scale=max(float(np.median(b[train&(b>0)])),1e-9)
                for name,design in designs.items():
                    coef=stable.ridge_qr(design[train],b[train]/scale);raw=design[test]@coef;prediction=np.maximum(raw,0)
                    folder=d.WORK/'task4_exact_pair_models'/panel/split/name;folder.mkdir(parents=True,exist_ok=True);file=folder/f'fold{k}.npz';np.savez_compressed(file,coef=coef,target_scale=scale,seed=SEED,fold=k);modelhashes[str(file.relative_to(d.ROOT))]=d.sha(file)
                    rows.append(dict(panel=panel,target_split=split,fold=k,model=name,heldout_paths=int(test.sum()),normalized_weak_prior_MSE=float(np.mean((prediction-b[test]/scale)**2)),pair_coefficient=float(coef[-1]) if name!='geometry' else None,prediction_clipped_fraction=float((raw<0).mean())))
        pd.DataFrame(rows).to_csv(O/'by_panel_split_fold.csv',index=False);pd.DataFrame(ranks).to_csv(O/'conditional_count_rank.csv',index=False)
        d.state(status='running',stage='task4_exact_pair_audit',panel=panel,next='Separate exact OD interaction from marginal effects and test conditional count information; supplied-prior labels only, no hidden OD.')
    df=pd.DataFrame(rows);wide=df.pivot(index=['panel','target_split','fold'],columns='model',values='normalized_weak_prior_MSE')
    for name in ['ridge_pair','exact_pair','exact_shuffled_pair']:wide[name+'_MSE_reduction_vs_geometry']=1-wide[name]/wide.geometry
    wide.to_csv(O/'paired_comparison.csv');wide.groupby('target_split')[[c for c in wide if c.endswith('_MSE_reduction_vs_geometry')]].mean().to_csv(O/'paired_summary.csv');d.save(O/'input_sha256.json',hashes);d.save(O/'model_sha256.json',modelhashes)
    d.state(status='stage_complete',stage='task4_exact_pair_audit_complete',next='Review exact vs regularized controls, conditional rank and all negative roads; independently reproduce before publishing. Public-prior predictability remains distinct from true OD gain.')


if __name__=='__main__':
    lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:main()
    except Exception as error:
        d.state(status='failed',stage='task4_exact_pair_audit',error=repr(error),next='Preserve exact projection definitions and fixed folds; no posthoc model or road selection.');traceback.print_exc();raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
