"""Compare cross-source agreement under the same OD movement caps."""
import hashlib
import msvcrt
from datetime import datetime, timezone

import common as d
import numpy as np
import pandas as pd
from task4_count_source_pair_stress import draw, family
from task4_exact_pair_audit import basis
from task4_public_range_shape_stress import tilt_prior
from task4_budgeted_source_direction import BUDGET
from audit_budgeted_source_path_cap import capped_solution, PATH_CAP

SEED=2026110349
FAMILIES=['origin_destination','sparse']
NOISES=family.NOISES


def main():
    out=d.OUT/'task4_source_agreement_stress'
    d.save(out/'protocol.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),seed=SEED,cases=80,
        families=FAMILIES,noises=NOISES,drift='moderate_drift',target_tv=.14,relative_l1_cap=BUDGET,
        objective='Compare mean source residual with signed minimum agreement (zero on sign disagreement) before the same 95th-percentile clipping and re-projection. Both use the same path and global caps. Match realized L1 movement by rescaling both directions to the smaller movement.',path_cap=PATH_CAP,
        constraints='Hold baseline count predictions, gross margins, original-prior orthant and L1 distance fixed. Same global movement cap for both objectives.',
        scope='Fresh artificial seed on previously studied support. Source interaction is heuristic. No hidden label or deployed candidate.',
        source_sha256=d.sha(__file__),cap_source_sha256=d.sha(d.WORK/'audit_budgeted_source_path_cap.py')))
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
                residuals=[]
                for prior in priors[:2]:
                    normalized=prior/max(float(np.median(prior[prior>0])),1e-9)
                    residuals.append(normalized-Q@(Q.T@normalized))
                left,right=residuals
                raw=(left+right)/2
                consensus=np.where(np.sign(left)==np.sign(right),
                                   np.sign(left)*np.minimum(np.abs(left),np.abs(right)),0.)
                directions={}
                for method,column in [('mean_source',raw),('agreement_source',consensus)]:
                    cutoff=float(np.quantile(np.abs(column),.95))
                    clipped=np.clip(column,-cutoff,cutoff)
                    directions[method]=clipped-Q@(Q.T@clipped)
                results={method:capped_solution(base,b,origin,dest,column) for method,column in directions.items()}
                if not all(result.success for result in results.values()):
                    failures.append(dict(case=case,seed=seed,messages={k:v.message for k,v in results.items()}))
                    continue
                steps={method:result.x[:len(base)]-base for method,result in results.items()}
                sizes={method:np.abs(step).sum() for method,step in steps.items()}
                target=min(sizes.values())
                flows={'baseline':base}
                for method,step in steps.items():
                    flows[method]=base+target/max(1e-12,sizes[method])*step
                    assert np.max(np.abs(flows[method]-base)-PATH_CAP*base)<=1e-7
                    np.testing.assert_allclose(np.abs(flows[method]-base).sum(),target,rtol=1e-9,atol=1e-7)
                for f in flows.values():
                    assert f.min()>=-1e-8 and np.abs(f-base).sum()<=BUDGET*base.sum()+1e-7
                    np.testing.assert_allclose(A@f,A@base,rtol=1e-10,atol=1e-7)
                    np.testing.assert_allclose(np.abs(f-b).sum(),np.abs(base-b).sum(),rtol=1e-9)
                folder=d.WORK/'task4_source_agreement_stress_cases'
                folder.mkdir(parents=True,exist_ok=True)
                file=folder/(hashlib.sha256(case.encode()).hexdigest()[:16]+'.npz')
                np.savez_compressed(file,seed=seed,truth=truth[2],priors=priors,counts=count,
                                    mean_direction=directions['mean_source'],agreement_direction=directions['agreement_source'],**flows)
                digest=d.sha(file)
                reference=family.base.metrics(base,truth[2],b,A,count,dest)
                for method,f in flows.items():
                    scores=family.base.metrics(f,truth[2],b,A,count,dest)
                    od_delta=scores['synthetic_S_od']-reference['synthetic_S_od']
                    np.testing.assert_allclose(scores['synthetic_S_ODME']-reference['synthetic_S_ODME'],
                                               .45*od_delta,rtol=0,atol=1e-9)
                    rows.append(dict(panel=panel,case=case,seed=seed,noise=noise,method=method,source_sign_agreement=float(np.mean(np.sign(left)==np.sign(right))),
                                     od_delta=od_delta,relative_movement=float(np.abs(f-base).sum()/base.sum()),
                                     moved_paths=int((np.abs(f-base)>1e-9).sum()),
                                     artifact_sha256=digest,**scores))
        pd.DataFrame(rows).to_csv(out/'by_case.csv',index=False)
        d.save(out/'failures.json',failures)
        d.state(status='running',stage='task4_source_agreement_stress',panel=panel)
    frame=pd.DataFrame(rows)
    assert frame.case.nunique()+len(failures)==80
    frame.groupby(['noise','method']).agg(cases=('od_delta','size'),mean_od_delta=('od_delta','mean'),
        harms=('od_delta',lambda x:int((x< -1e-12).sum())),mean_movement=('relative_movement','mean'),
        mean_moved_paths=('moved_paths','mean')).to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.state(status='stage_complete',stage='task4_source_agreement_stress_complete',
        next='Verify constraints, matched budgets and portable reproduction before publication.')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
