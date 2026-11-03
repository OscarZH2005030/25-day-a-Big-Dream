"""Compare the fixed local cap on fresh four-mechanism stress."""
import hashlib
import msvcrt
from datetime import datetime, timezone

import common as d
import numpy as np
import pandas as pd
from task4_count_source_pair_stress import draw, family
from task4_exact_pair_audit import basis
from task4_public_range_shape_stress import tilt_prior
from task4_budgeted_source_direction import fit, BUDGET
from audit_budgeted_source_path_cap import capped_solution, PATH_CAP

SEED=2026110348
FAMILIES=['origin_destination','sparse']
NOISES=family.NOISES


def main():
    out=d.OUT/'task4_path_cap_extended'
    d.save(out/'protocol.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),seed=SEED,cases=80,
        families=FAMILIES,noises=NOISES,drift='moderate_drift',target_tv=.14,relative_l1_cap=BUDGET,
        objective='Maximize alignment with the fixed 95th-percentile clipped and reprojected source interaction. Compare a 5% per-path cap with global-only movement rescaled to the same realized L1 length.',path_cap=PATH_CAP,
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
                source=np.mean([v/max(float(np.median(v[v>0])),1e-9) for v in priors[:2]],axis=0)
                raw=source-Q@(Q.T@source)
                cutoff=float(np.quantile(np.abs(raw),.95))
                clipped=np.clip(raw,-cutoff,cutoff)
                clipped-=Q@(Q.T@clipped)
                original=fit(base,b,origin,dest,clipped)
                capped=capped_solution(base,b,origin,dest,clipped)
                if not (original.success and capped.success):
                    failures.append(dict(case=case,seed=seed,global_message=original.message,local_message=capped.message))
                    continue
                local_flow=capped.x[:len(base)]
                global_step=original.x[:len(base)]-base
                local_size=np.abs(local_flow-base).sum()
                factor=local_size/max(1e-12,np.abs(global_step).sum())
                assert factor<=1.+1e-9
                matched=base+min(1.,factor)*global_step
                np.testing.assert_allclose(np.abs(matched-base).sum(),local_size,rtol=1e-9,atol=1e-7)
                assert np.max(np.abs(local_flow-base)-PATH_CAP*base)<=1e-7
                flows={'baseline':base,'path_cap':local_flow,'matched_global':matched}
                for f in flows.values():
                    assert f.min()>=-1e-8 and np.abs(f-base).sum()<=BUDGET*base.sum()+1e-7
                    np.testing.assert_allclose(A@f,A@base,rtol=1e-10,atol=1e-7)
                    np.testing.assert_allclose(np.abs(f-b).sum(),np.abs(base-b).sum(),rtol=1e-9)
                folder=d.WORK/'task4_path_cap_extended_cases'
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
                                     moved_paths=int((np.abs(f-base)>1e-9).sum()),
                                     artifact_sha256=digest,**scores))
        pd.DataFrame(rows).to_csv(out/'by_case.csv',index=False)
        d.save(out/'failures.json',failures)
        d.state(status='running',stage='task4_path_cap_extended',panel=panel)
    frame=pd.DataFrame(rows)
    assert frame.case.nunique()+len(failures)==80
    frame.groupby(['noise','method']).agg(cases=('od_delta','size'),mean_od_delta=('od_delta','mean'),
        harms=('od_delta',lambda x:int((x< -1e-12).sum())),mean_movement=('relative_movement','mean'),
        mean_moved_paths=('moved_paths','mean')).to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.state(status='stage_complete',stage='task4_path_cap_extended_complete',
        next='Verify constraints, matched budgets and portable reproduction before publication.')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
