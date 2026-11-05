"""Measure repeatability of the unobserved pairing component of the released priors."""
import msvcrt

import numpy as np
import pandas as pd
from scipy.linalg import orth

import common as d
from audit_direct_od_measurement_design import base





def main():
    out = d.OUT/'task4_pairing_prior_gate_matched_energy'
    d.save(out/'protocol.json',dict(source_sha256=d.sha(__file__),predictor_splits=['train','validation'],
        evaluation_splits=['private'],scope='Compare the reprojected agreement gate at its natural and matched L2 norms. Released weak prior targets only.'))
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
                agree=(source*second>0)&(np.abs(source)>1e-12)&(np.abs(second)>1e-12)
                gated=average*agree
                gated-=basis@(basis.T@gated)
                gate_norm=np.linalg.norm(gated)
                assert gate_norm>1e-12
                factor=np.linalg.norm(average)/gate_norm
                matched=gated*factor
                np.testing.assert_allclose(np.linalg.norm(matched),np.linalg.norm(average),rtol=1e-12)
                assert np.abs(constraints@gated).max()<1e-10
                for method,prediction in [('zero',np.zeros_like(source)),('average_half',average),
                    ('agreement_only',gated),('agreement_matched',matched)]:
                    error=prediction-truth
                    rows.append(dict(panel=panel,view=view,method=method,agreement_fraction=agree.mean(),gate_scale=factor,relative_l1=np.abs(error).sum()/target_l1,
                                     relative_squared_error=(error@error)/target_l2))
    data=pd.DataFrame(rows)
    data.to_csv(out/'by_panel.csv',index=False)
    data.groupby(['view','method'])[['relative_l1','relative_squared_error']].mean().to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',failures)


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
