"""Measure repeatability of the unobserved pairing component of the released priors."""
import msvcrt

import numpy as np
import pandas as pd
from scipy.linalg import orth

import common as d
from audit_direct_od_measurement_design import base





def main():
    out = d.OUT/'task4_pairing_prior_random_gate'
    d.save(out/'protocol.json',dict(source_sha256=d.sha(__file__),predictor_splits=['train','validation'],
        evaluation_splits=['private'],seed=2026110503, random_replicates=32,scope='Compare gates at matched size and norm with uniform random masks. Released weak prior targets only.'))
    rows, hashes, failures=[], {}, []
    rng=np.random.default_rng(2026110503)
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
                indices=np.argsort(-np.round(np.abs(average),14),kind='stable')[:int(agree.sum())]
                magnitude=np.zeros_like(average)
                magnitude[indices]=average[indices]
                magnitude-=basis@(basis.T@magnitude)
                magnitude*=np.linalg.norm(average)/np.linalg.norm(magnitude)
                assert np.abs(constraints@gated).max()<1e-10
                assert np.abs(constraints@magnitude).max()<1e-10
                np.testing.assert_allclose(np.linalg.norm(magnitude),np.linalg.norm(average),rtol=1e-12)
                candidates=[('zero',-1,np.zeros_like(source)),('average_half',-1,average),
                            ('magnitude_matched',-1,magnitude),('agreement_matched',-1,matched)]
                for repeat in range(32):
                    selected=rng.choice(len(average),size=int(agree.sum()),replace=False)
                    random_gate=np.zeros_like(average)
                    random_gate[selected]=average[selected]
                    random_gate-=basis@(basis.T@random_gate)
                    random_gate*=np.linalg.norm(average)/np.linalg.norm(random_gate)
                    candidates.append(('random_matched',repeat,random_gate))
                for method,repeat,prediction in candidates:
                    error=prediction-truth
                    rows.append(dict(panel=panel,view=view,method=method,repeat=repeat,agreement_fraction=agree.mean(),gate_scale=factor,relative_l1=np.abs(error).sum()/target_l1,
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
