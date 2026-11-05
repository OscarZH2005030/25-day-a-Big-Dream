"""Measure repeatability of the unobserved pairing component of the released priors."""
import msvcrt

import numpy as np
import pandas as pd
from scipy.linalg import orth

import common as d
from audit_direct_od_measurement_design import base





def main():
    out = d.OUT/'task4_pairing_disagreement_calibration'
    d.save(out/'protocol.json',dict(source_sha256=d.sha(__file__),predictor_splits=['train','validation'],
        evaluation_splits=['private'],scope='Frozen 0.5 disagreement threshold within predictor amplitude quartiles. Aggregate weak-prior proxy diagnostics only.'))
    rows, hashes, failures=[], {}, []
    for panel in sorted(d.night.predictor.load_routes()):
        train_paths, _, _, train, provenance=base.load(panel,'train')
        hashes.update(provenance)
        train=train/train.sum()
        valid_paths, _, _, valid, provenance=base.load(panel,'validation')
        hashes.update(provenance)
        assert valid_paths.path_id.equals(train_paths.path_id)
        valid=valid/valid.sum()
        for split in ('private',):
            paths, counts, _, prior, provenance=base.load(panel,split)
            hashes.update(provenance)
            assert paths.path_id.equals(train_paths.path_id)
            target=prior/prior.sum()
            groups=np.vstack([pd.get_dummies(paths[k]).to_numpy(dtype=float).T
                              for k in ('origin_zone','destination_zone')])
            origin,destination=[paths[k].str.extract(r'^Z([0-9]+)$',expand=False).astype(int).to_numpy()
                                for k in ('origin_zone','destination_zone')]
            span=destination-origin
            bins=(span[:,None]==np.unique(span)).astype(float).T
            for view,constraints in [('margins',np.vstack((counts,groups))),
                                     ('margins_and_histogram',np.vstack((counts,groups,bins)))]:
                basis=orth(constraints.T,rcond=1e-10)
                pairs=np.column_stack((train,valid,target))
                projected=pairs-basis@(basis.T@pairs)
                source,second,truth=projected.T
                target_l1=np.abs(truth).sum()
                target_l2=truth@truth
                if min(target_l1,target_l2)<=1e-15:
                    failures.append(dict(panel=panel,view=view,error='Zero residual norm'))
                    continue
                average=.25*(source+second)
                magnitude=np.abs(source)+np.abs(second)
                disagreement=np.divide(np.abs(source-second),magnitude,out=np.zeros_like(source),where=magnitude>1e-12)
                eligible = magnitude > 1e-12
                cuts = np.quantile(magnitude[eligible], [.25, .5, .75])
                quartile = np.searchsorted(cuts, magnitude, side='right')
                error = average - truth
                for band in range(4):
                    for high in (False, True):
                        keep = eligible & (quartile == band) & ((disagreement >= .5) == high)
                        if not keep.any():
                            failures.append(dict(panel=panel, view=view, quartile=band, high=high, error='Empty stratum'))
                            continue
                        rows.append(dict(panel=panel, view=view, quartile=band, high=high,
                            cells=int(keep.sum()), proxy_l1=float(np.abs(truth[keep]).sum()),
                            error_l1=float(np.abs(error[keep]).sum()),
                            proxy_l2=float(truth[keep] @ truth[keep]),
                            error_l2=float(error[keep] @ error[keep])))

    data=pd.DataFrame(rows)
    data.to_csv(out/'by_panel.csv',index=False)
    paired = data.pivot(index=['panel','view','quartile'], columns='high',
                        values=['proxy_l1','error_l1','proxy_l2','error_l2']).dropna()
    for metric in ('l1','l2'):
        paired[metric, 'delta'] = (
            paired['error_'+metric, True] / paired['proxy_'+metric, True]
            - paired['error_'+metric, False] / paired['proxy_'+metric, False])
    paired[[('l1','delta'),('l2','delta')]].to_csv(out/'paired.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',failures)


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
