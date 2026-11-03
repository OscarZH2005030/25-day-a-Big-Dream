"""Compare exact same-pair signal with same-span neighborhood transfer."""
import hashlib
import msvcrt
from datetime import datetime, timezone

import common as d
import numpy as np
import pandas as pd
import task4_exact_pair_audit as exact

SEED = 2026110336


def source_features(paths, X, Q, source):
    origin = paths.origin_zone.str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy()
    dest = paths.destination_zone.str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy()
    span = dest-origin
    residual = source-Q@(Q.T@source)
    same, adjacent, distant = [np.zeros(len(paths)) for _ in range(3)]
    supported = np.zeros(len(paths), dtype=bool)
    for length in np.unique(span):
        indices = np.flatnonzero(span==length)
        indices = indices[np.argsort(origin[indices])]
        if len(indices)<4:
            continue
        supported[indices] = True
        same[indices] = residual[indices]
        for j, row in enumerate(indices):
            neighbors = indices[max(0,j-1):j].tolist()+indices[j+1:j+2].tolist()
            adjacent[row] = residual[neighbors].mean()
        distant[indices] = residual[np.roll(indices,len(indices)//2)]
    features = {'geometry':X}
    for name, column in [('same_pair',same),('adjacent_span',adjacent),('distant_span',distant)]:
        projected = column-Q@(Q.T@column)
        assert np.linalg.norm(Q.T@projected)<1e-9*max(1.,np.linalg.norm(column))
        features[name] = np.column_stack([X,projected])
    return features, float(supported.mean())


def main():
    out = d.OUT/'task4_local_pair_transfer'
    d.save(out/'protocol.json',dict(
        registered_utc=datetime.now(timezone.utc).isoformat(), seed=SEED, folds=5,
        hypothesis='Common residuals may transfer across nearby OD pairs at fixed span. Compare locality with exact identity and a distant same-span control.',
        feature='Use two non-target supplied priors. Remove full geometry span and marginal basis, then use direct, adjacent origin-rank mean, or half-group cyclic-shift residual at the same span. Groups smaller than four zeroed in all controls before re-projection. Re-projection can mix source entries.',
        fit='Identical ridge1 and target training-fold median across all four designs.',
        scope='Exposed public weak-prior prediction. No hidden OD inference, noise-origin identification or deployment selection.',
        source_sha256=d.sha(__file__), projection_source_sha256=d.sha(exact.__file__)))
    rows, folds, hashes, models = [], [], {}, {}
    for panel in d.night.predictor.load_routes():
        priors = {}
        for split in exact.SPLITS:
            paths, _, _, prior, inputs = exact.base.load(panel,split)
            priors[split] = prior
            hashes.update(inputs)
        X = exact.structure.features(paths)['span_origin_destination']
        Q,_ = exact.basis(X)
        seed = int.from_bytes(hashlib.sha256(f'{SEED}/{panel}'.encode()).digest()[:8],'little')%(2**63)
        rng = np.random.default_rng(seed)
        fold = np.empty(len(paths),int)
        fold[rng.permutation(len(paths))] = np.arange(len(paths))%5
        folds.extend(dict(panel=panel,path_id=path,fold=int(k)) for path,k in zip(paths.path_id,fold))
        for split,target in priors.items():
            source = np.mean([p/max(float(np.median(p[p>0])),1e-9) for key,p in priors.items() if key!=split],axis=0)
            features,support = source_features(paths,X,Q,source)
            for k in range(5):
                train,test = fold!=k,fold==k
                scale = max(float(np.median(target[train & (target>0)])),1e-9)
                for method,design in features.items():
                    coef = exact.stable.ridge_qr(design[train],target[train]/scale)
                    error = np.maximum(design[test]@coef,0)-target[test]/scale
                    folder = d.WORK/'task4_local_pair_models'/panel/split/method
                    folder.mkdir(parents=True,exist_ok=True)
                    file = folder/f'fold{k}.npz'
                    np.savez_compressed(file,coef=coef,scale=scale,seed=seed,fold=k)
                    models[str(file.relative_to(d.WORK)).replace('\\','/')] = d.sha(file)
                    rows.append(dict(panel=panel,split=split,fold=k,method=method,support=support,
                                     n_test=int(test.sum()),mse=float(np.mean(error**2)),mae=float(np.mean(np.abs(error)))))
        pd.DataFrame(rows).to_csv(out/'by_panel_fold.csv',index=False)
        d.state(status='running',stage='local_pair_transfer',panel=panel)
    frame = pd.DataFrame(rows)
    pivot = frame.pivot(index=['panel','split','fold'],columns='method',values='mse')
    improvement = 1-pivot[['same_pair','adjacent_span','distant_span']].div(pivot.geometry,axis=0)
    improvement.to_csv(out/'paired_mse_improvement.csv')
    improvement.groupby('split').mean().to_csv(out/'summary.csv')
    pd.DataFrame(folds).to_csv(out/'fold_assignment.csv',index=False)
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'model_sha256.json',models)
    d.state(status='stage_complete',stage='local_pair_transfer_complete',next='Verify the three spatial controls and standalone reproduction. Interpret weak-prior locality without claiming hidden OD gains.')


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
