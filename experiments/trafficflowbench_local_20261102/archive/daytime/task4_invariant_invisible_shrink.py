"""Test invisible-L1 shrinkage while preserving the other Task4 components."""
import hashlib
import msvcrt
from datetime import datetime, timezone

import common as d
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from task4_count_source_pair_stress import draw, family
from task4_public_range_shape_stress import tilt_prior

SEED=2026110346
FAMILIES=['origin_destination','sparse']
NOISES=['shared_lognormal','shared_positive_background']


def repair(flow,prior,origin,dest,n):
    size=len(flow)
    columns=np.arange(size)
    O=sparse.csr_matrix((np.ones(size),(origin,columns)),shape=(n,size))
    D=sparse.csr_matrix((np.ones(size),(dest,columns)),shape=(n,size))
    imbalance=(O-D)@(flow-prior)
    signed=prior+(imbalance[origin]-imbalance[dest])/n
    sign=np.sign(flow-prior)
    constraints=sparse.vstack([O,D,sparse.csr_matrix(sign[None,:])],format='csr')
    equality=sparse.hstack([constraints,sparse.csr_matrix(constraints.shape)],format='csr')
    identity=sparse.eye(size,format='csr')
    inequality=sparse.vstack([sparse.hstack([identity,-identity]),sparse.hstack([-identity,-identity])],format='csr')
    bounds=[(float(b),None) if s>0 else (0.,float(b)) if s<0 else (float(b),float(b))
            for b,s in zip(prior,sign)]+[(0.,None)]*size
    result=linprog(np.r_[np.zeros(size),np.ones(size)],A_ub=inequality,b_ub=np.r_[signed,-signed],
                   A_eq=equality,b_eq=constraints@flow,bounds=bounds,method='highs')
    return result,signed


def main():
    out=d.OUT/'task4_invariant_invisible_shrink'
    d.save(out/'protocol.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),seed=SEED,cases=40,
        families=FAMILIES,noises=NOISES,drift='moderate_drift',target_tv=.14,blend=[0.,.05,1.],
        hypothesis='Reducing the invisible L1 correction on the matched-score face may help OD if the baseline adds unnecessary matching distortion.',
        method='Solve nonnegative invisible-L1 LP within baseline counts, gross margins and original-prior sign orthant and L1 distance. Full LP and fixed 5% interpolation with baseline. Non-OD score components remain equal up to numerical tolerance.',
        scope='Fresh artificial seed on previously studied support and mechanisms. Does not select a deployed objective or validate hidden OD.',
        source_sha256=d.sha(__file__)))
    rows,failures,hashes=[],[],{}
    for panel in d.night.predictor.load_routes():
        paths,A,real_count,_,inputs=family.base.load(panel,'train')
        hashes.update(inputs)
        origin,dest=[paths[key].str[1:].astype(int).to_numpy() for key in ['origin_zone','destination_zone']]
        origin,dest=origin-origin.min(),dest-origin.min()
        n=int(dest.max()+1)
        midpoint=(origin+dest)/2
        position=(midpoint-midpoint.min())/np.ptp(midpoint)
        level=float(np.median(real_count[real_count>0]))
        for truth_family in FAMILIES:
            for noise in NOISES:
                case=f'{panel}/{truth_family}/{noise}'
                seed=int.from_bytes(hashlib.sha256(f'{SEED}/{case}'.encode()).digest()[:8],'little')%(2**63)
                rng=np.random.default_rng(seed)
                truth,prior,count,_=draw(paths,A,level,truth_family,noise,'moderate_drift',rng)
                prepared=tilt_prior(A,count[2],prior[2],position if seed%2 else 1-position)
                if prepared is None:
                    failures.append(dict(case=case,seed=seed,reason='TV root not bracketed'))
                    continue
                b=prepared[0]
                base,status=family.base.ridge(A,count[2],b)
                if not status['success']:
                    failures.append(dict(case=case,seed=seed,reason='Baseline solver'))
                    continue
                result,signed=repair(base,b,origin,dest,n)
                if not result.success:
                    failures.append(dict(case=case,seed=seed,reason=result.message))
                    continue
                proposed=result.x[:len(base)]
                np.testing.assert_allclose(A@proposed,A@base,rtol=1e-10,atol=1e-7)
                np.testing.assert_allclose(np.abs(proposed-b).sum(),np.abs(base-b).sum(),rtol=1e-9)
                folder=d.WORK/'task4_invariant_invisible_shrink_cases'
                folder.mkdir(parents=True,exist_ok=True)
                file=folder/(hashlib.sha256(case.encode()).hexdigest()[:16]+'.npz')
                np.savez_compressed(file,seed=seed,truth=truth[2],prior=b,counts=count[2],
                                    baseline=base,signed=signed,proposed=proposed)
                digest=d.sha(file)
                reference=family.base.metrics(base,truth[2],b,A,count[2],dest)
                for blend in [0.,.05,1.]:
                    flow=(1.-blend)*base+blend*proposed
                    scores=family.base.metrics(flow,truth[2],b,A,count[2],dest)
                    assert max(abs(scores[k]-reference[k]) for k in ['synthetic_S_link','synthetic_S_dev','synthetic_S_attr'])<1e-8
                    rows.append(dict(panel=panel,case=case,noise=noise,family=truth_family,seed=seed,blend=blend,
                                     od_delta=scores['synthetic_S_od']-reference['synthetic_S_od'],
                                     od_movement=float(np.abs(flow-base).sum()/base.sum()),
                                     artifact_sha256=digest,**scores))
        pd.DataFrame(rows).to_csv(out/'all_cases.csv',index=False)
        d.save(out/'failures.json',failures)
        d.state(status='running',stage='task4_invariant_invisible_shrink',panel=panel)
    frame=pd.DataFrame(rows)
    assert frame.case.nunique()+len(failures)==40
    frame.groupby(['noise','blend']).od_delta.agg(['count','mean','min',lambda v:int((v<-1e-12).sum())]).to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.state(status='stage_complete',stage='task4_invariant_invisible_shrink_complete',
        next='Independently verify non-OD invariance and reproduce fixed-seed outputs before publication.')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
