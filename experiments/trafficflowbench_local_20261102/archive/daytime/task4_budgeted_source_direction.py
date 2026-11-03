"""Test source-guided OD directions under a fixed matched-score movement budget."""
import hashlib
import msvcrt
from datetime import datetime, timezone

import common as d
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from task4_count_source_pair_stress import draw, family
from task4_exact_pair_audit import basis
from task4_public_range_shape_stress import tilt_prior

SEED=2026110347
BUDGET=.002
FAMILIES=['origin_destination','sparse']
NOISES=['shared_lognormal','shared_positive_background']


def fit(base, prior, origin, dest, direction):
    n=len(base)
    idx=np.arange(n)
    O=sparse.csr_matrix((np.ones(n),(origin,idx)),shape=(origin.max()+1,n))
    D=sparse.csr_matrix((np.ones(n),(dest,idx)),shape=(dest.max()+1,n))
    sign=np.sign(base-prior)
    M=sparse.vstack([O,D,sparse.csr_matrix(sign[None,:])],format='csr')
    equality=sparse.hstack([M,sparse.csr_matrix(M.shape)],format='csr')
    I=sparse.eye(n,format='csr')
    inequality=sparse.vstack([sparse.hstack([I,-I]),sparse.hstack([-I,-I]),
                              sparse.csr_matrix(np.r_[np.zeros(n),np.ones(n)][None,:])],format='csr')
    bounds=[(float(b),None) if s>0 else (0.,float(b)) if s<0 else (float(b),float(b))
            for b,s in zip(prior,sign)]+[(0.,None)]*n
    scale=max(1e-12,float(np.max(np.abs(direction))))
    return linprog(np.r_[-direction/scale,np.zeros(n)],A_ub=inequality,
                   b_ub=np.r_[base,-base,BUDGET*base.sum()],
                   A_eq=equality,b_eq=M@base,bounds=bounds,method='highs')


def main():
    out=d.OUT/'task4_budgeted_source_direction'
    d.save(out/'protocol.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),seed=SEED,cases=40,
        families=FAMILIES,noises=NOISES,drift='moderate_drift',target_tv=.14,relative_l1_cap=BUDGET,
        objective='Maximize alignment with the exact source interaction or its 95th-percentile clipped and reprojected version.',
        constraints='Hold baseline count predictions, gross margins, original-prior orthant and L1 distance fixed. Same global movement cap for both objectives.',
        scope='Fresh artificial seed on previously studied support. Source interaction is heuristic. No hidden label or deployed candidate.',
        source_sha256=d.sha(__file__)))
    rows,failures,hashes=[],[],{}
    for panel in d.night.predictor.load_routes():
        paths,A,real_count,_,inputs=family.base.load(panel,'train')
        hashes.update(inputs)
        X=family.structure.features(paths)['span_origin_destination']
        Q,_=basis(X)
        origin,dest=[paths[key].str[1:].astype(int).to_numpy() for key in ['origin_zone','destination_zone']]
        midpoint=(origin+dest)/2
        position=(midpoint-midpoint.min())/np.ptp(midpoint)
        level=float(np.median(real_count[real_count>0]))
        for truth_family in FAMILIES:
            for noise in NOISES:
                case=f'{panel}/{truth_family}/{noise}'
                seed=int.from_bytes(hashlib.sha256(f'{SEED}/{case}'.encode()).digest()[:8],'little')%(2**63)
                rng=np.random.default_rng(seed)
                truth,priors,counts,_=draw(paths,A,level,truth_family,noise,'moderate_drift',rng)
                prepared=[tilt_prior(A,c,b,position if seed%2 else 1-position) for c,b in zip(counts,priors)]
                if any(item is None for item in prepared):
                    failures.append(dict(case=case,seed=seed,reason='TV root not bracketed'))
                    continue
                priors=np.array([item[0] for item in prepared])
                b,count=priors[2],counts[2]
                base,status=family.base.ridge(A,count,b)
                if not status['success']:
                    failures.append(dict(case=case,seed=seed,reason='Baseline solver'))
                    continue
                source=np.mean([v/max(float(np.median(v[v>0])),1e-9) for v in priors[:2]],axis=0)
                raw=source-Q@(Q.T@source)
                cutoff=float(np.quantile(np.abs(raw),.95))
                clipped=np.clip(raw,-cutoff,cutoff)
                clipped-=Q@(Q.T@clipped)
                flows={'baseline':base}
                for method,direction in [('raw_source',raw),('clipped_source',clipped)]:
                    result=fit(base,b,origin,dest,direction)
                    if not result.success:
                        failures.append(dict(case=case,seed=seed,method=method,reason=result.message))
                        continue
                    f=result.x[:len(base)]
                    assert f.min()>=-1e-8 and np.abs(f-base).sum()<=BUDGET*base.sum()+1e-7
                    np.testing.assert_allclose(A@f,A@base,rtol=1e-10,atol=1e-7)
                    np.testing.assert_allclose(np.abs(f-b).sum(),np.abs(base-b).sum(),rtol=1e-9)
                    flows[method]=f
                folder=d.WORK/'task4_budgeted_source_direction_cases'
                folder.mkdir(parents=True,exist_ok=True)
                file=folder/(hashlib.sha256(case.encode()).hexdigest()[:16]+'.npz')
                np.savez_compressed(file,seed=seed,truth=truth[2],priors=priors,counts=count,
                                    raw_direction=raw,clipped_direction=clipped,**flows)
                digest=d.sha(file)
                reference=family.base.metrics(base,truth[2],b,A,count,dest)
                for method,f in flows.items():
                    scores=family.base.metrics(f,truth[2],b,A,count,dest)
                    od_delta=scores['synthetic_S_od']-reference['synthetic_S_od']
                    np.testing.assert_allclose(scores['synthetic_S_ODME']-reference['synthetic_S_ODME'],
                                               .45*od_delta,rtol=0,atol=1e-9)
                    rows.append(dict(panel=panel,case=case,seed=seed,noise=noise,method=method,
                                     od_delta=od_delta,relative_movement=float(np.abs(f-base).sum()/base.sum()),
                                     artifact_sha256=digest,**scores))
        pd.DataFrame(rows).to_csv(out/'by_case.csv',index=False)
        d.save(out/'failures.json',failures)
        d.state(status='running',stage='task4_budgeted_source_direction',panel=panel)
    frame=pd.DataFrame(rows)
    frame.groupby(['noise','method']).agg(cases=('od_delta','size'),mean_od_delta=('od_delta','mean'),
        harms=('od_delta',lambda x:int((x< -1e-12).sum())),mean_movement=('relative_movement','mean')).to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.state(status='stage_complete',stage='task4_budgeted_source_direction_complete',
        next='Verify constraints, matched budgets and portable reproduction before publication.')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
