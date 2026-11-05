"""Measure repeatability of the unobserved pairing component of the released priors."""
import msvcrt

import numpy as np
import pandas as pd
from scipy.linalg import orth

import common as d
from audit_direct_od_measurement_design import base





def main():
    out = d.OUT/'task4_pairing_prior_shrink_transfer'
    d.save(out/'protocol.json',dict(source_sha256=d.sha(__file__), predictor_split='train',
        coefficient_split='validation', evaluation_split='private',
        scope='Learn one shrinkage coefficient from other corridor families. Compare to fixed 0, .5, 1 on private released weak priors only.'))
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
                cosine=(paired_train@paired_target)/(train_norm*target_norm)
                assert abs(cosine)<=1+1e-12
                rows.append(dict(panel=panel,split=split,view=view,cosine=cosine,
                    source_energy=train_norm**2,target_energy=target_norm**2,cross_energy=paired_train@paired_target))
    data=pd.DataFrame(rows)
    data['family']=data.panel.str.rsplit('_',n=1).str[0]
    scores=[]
    for row in data.query("split=='private'").itertuples():
        calibration=data.query("split=='validation' and view==@row.view and family!=@row.family")
        assert len(calibration)>0
        alpha=float(np.clip((calibration.cross_energy/calibration.target_energy).sum()
                            /(calibration.source_energy/calibration.target_energy).sum(),0.,1.))
        for name, scale in [('zero',0.),('half',.5),('full',1.),('cross_family',alpha)]:
            ratio=1+(scale**2*row.source_energy-2*scale*row.cross_energy)/row.target_energy
            scores.append(dict(panel=row.panel,view=row.view,method=name,alpha=scale,mse_ratio=ratio))
    result=pd.DataFrame(scores)
    result.to_csv(out/'by_panel.csv',index=False)
    result.groupby(['view','method']).mse_ratio.agg(['mean','min','max']).to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',failures)


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
