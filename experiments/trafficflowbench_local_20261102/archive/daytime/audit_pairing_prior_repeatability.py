"""Measure repeatability of the unobserved pairing component of the released priors."""
import msvcrt

import numpy as np
import pandas as pd
from scipy.linalg import orth

import common as d
from audit_direct_od_measurement_design import base





def main():
    out = d.OUT/'task4_pairing_prior_repeatability'
    d.save(out/'protocol.json',dict(source_sha256=d.sha(__file__),training_split='train',
        evaluation_splits=['validation','private'],scope='Released normalized weak-prior repeatability. The train predictor and zero comparator are fixed; no fitting to evaluation prior. No hidden OD labels.'))
    rows, hashes, failures=[], {}, []
    for panel in sorted(d.night.predictor.load_routes()):
        train_paths, _, _, train, provenance=base.load(panel,'train')
        hashes.update(provenance)
        train=train/train.sum()
        for split in ('validation','private'):
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
                paired_train=train-basis@(basis.T@train)
                paired_target=target-basis@(basis.T@target)
                train_norm=np.linalg.norm(paired_train)
                target_norm=np.linalg.norm(paired_target)
                if min(train_norm,target_norm)<=1e-12:
                    failures.append(dict(panel=panel,split=split,view=view,error='Zero residual norm'))
                    continue
                mse_ratio=np.linalg.norm(paired_train-paired_target)**2/target_norm**2
                cosine=(paired_train@paired_target)/(train_norm*target_norm)
                assert abs(cosine)<=1+1e-12
                rows.append(dict(panel=panel,split=split,view=view,cosine=cosine,
                    mse_ratio_to_zero=mse_ratio,target_pairing_energy_share=(target_norm/np.linalg.norm(target))**2))
    data=pd.DataFrame(rows)
    data.to_csv(out/'by_panel.csv',index=False)
    data.groupby(['split','view'])[['cosine','mse_ratio_to_zero','target_pairing_energy_share']].mean().to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',failures)


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
