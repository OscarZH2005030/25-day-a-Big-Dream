"""Stress source agreement with a count-invisible perturbation of one source."""
import hashlib
import msvcrt

import common as d
import numpy as np
import pandas as pd
from audit_budgeted_source_path_cap import capped_solution
from od_cycle_refinement import draw_cycles, path_lookup
from task4_count_invariant_drift import rearrange
from task4_exact_pair_audit import base as inputs, basis, structure
from verify_cycle_budget import score

SEED=2026110350
SOURCE_CAP=.1


def directions(priors,Q):
    residuals=[]
    for prior in priors[:2]:
        normalized=prior/max(float(np.median(prior[prior>0])),1e-9)
        residuals.append(normalized-Q@(Q.T@normalized))
    a,b=residuals
    agreement=np.where(np.sign(a)==np.sign(b),np.sign(a)*np.minimum(np.abs(a),np.abs(b)),0.)
    result={}
    for name,values in [('mean_source',(a+b)/2),('agreement_source',agreement)]:
        cutoff=float(np.quantile(np.abs(values),.95))
        clipped=np.clip(values,-cutoff,cutoff)
        result[name]=clipped-Q@(Q.T@clipped)
    return result


def main():
    source=d.OUT/'task4_source_agreement_stress/by_case.csv'
    out=d.OUT/'task4_source_agreement_corruption'
    d.save(out/'protocol.json',dict(seed=SEED,scope='Paired posthoc perturbation on 80 reused synthetic cases. No fresh hidden validation.',
        method='Apply seeded nonnegative OD exchanges to source zero only. Limit to 0.1 source-relative L1. Preserve source gross margins and loaded counts exactly. Recompute both objectives and match the smallest movement across clean and perturbed versions.',
        source_sha256=d.sha(__file__),case_table_sha256=d.sha(source)))
    table=pd.read_csv(source).query("method=='baseline'")
    rows,failures,hashes=[],[],{}
    for panel,group in table.groupby('panel'):
        paths,A,_,_,provenance=inputs.load(panel,'train')
        hashes.update(provenance)
        origin,dest=[paths[k].str[1:].astype(int).to_numpy() for k in ['origin_zone','destination_zone']]
        Q,_=basis(structure.features(paths)['span_origin_destination'])
        lookup=path_lookup(origin,dest)
        for row in group.itertuples():
            key=hashlib.sha256(row.case.encode()).hexdigest()[:16]
            original=d.WORK/'task4_source_agreement_stress_cases'/f'{key}.npz'
            assert d.sha(original)==row.artifact_sha256
            seed=int.from_bytes(hashlib.sha256(f'{SEED}/{row.case}'.encode()).digest()[:8],'little')%(2**63)
            rng=np.random.default_rng(seed)
            with np.load(original) as z:
                priors=z['priors'].copy()
                base,b,truth,count=z['baseline'],priors[2],z['truth'],z['counts']
                moved=rearrange(priors[0],draw_cycles(lookup,4*len(base),rng),rng)
                change=moved-priors[0]
                ratio=min(1.,SOURCE_CAP*priors[0].sum()/max(1e-12,np.abs(change).sum()))
                change*=ratio
                priors[0]+=change
                assert priors[0].min()>=-1e-9
                for idx in [origin,dest]:
                    assert np.max(np.abs(np.bincount(idx,weights=change)))<1e-7
                assert np.max(np.abs(A@change))<1e-7
                proposed={}
                for method,column in directions(priors,Q).items():
                    result=capped_solution(base,b,origin,dest,column)
                    if not result.success:
                        failures.append(dict(case=row.case,seed=seed,method=method,message=result.message))
                        continue
                    proposed[method]=result.x[:len(base)]
                if len(proposed)!=2:
                    continue
                alternatives={f'clean_{method}':z[method] for method in proposed}
                alternatives.update({f'perturbed_{method}':flow for method,flow in proposed.items()})
                sizes={method:np.abs(flow-base).sum() for method,flow in alternatives.items()}
                budget=min(sizes.values())
                flows={method:base+budget/max(size,1e-12)*(alternatives[method]-base) for method,size in sizes.items()}
                folder=d.WORK/'task4_source_agreement_corruption_cases'
                folder.mkdir(parents=True,exist_ok=True)
                file=folder/f'{key}.npz'
                np.savez_compressed(file,seed=seed,source_change=change,**flows)
                digest=d.sha(file)
                reference=score(base,truth,b,A,count,dest)
                for method,flow in flows.items():
                    assert flow.min()>=-1e-8
                    np.testing.assert_allclose(np.abs(flow-base).sum(),budget,rtol=1e-9,atol=1e-7)
                    od,total=score(flow,truth,b,A,count,dest)
                    np.testing.assert_allclose(total-reference[1],.45*(od-reference[0]),rtol=0,atol=1e-10)
                    version,rule=method.split('_',1)
                    rows.append(dict(panel=panel,case=row.case,noise=row.noise,seed=seed,version=version,rule=rule,
                                     source_relative_movement=float(np.abs(change).sum()/priors[0].sum()),
                                     target_relative_movement=float(budget/base.sum()),od_delta=od-reference[0],
                                     artifact_sha256=digest))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'by_case.csv',index=False)
    wide=frame.pivot(index=['noise','rule','case'],columns='version',values='od_delta')
    wide['perturbation_effect']=wide.perturbed-wide.clean
    wide.to_csv(out/'paired_od.csv')
    wide.groupby(level=['noise','rule']).agg(cases=('clean','size'),clean_mean=('clean','mean'),
        perturbed_mean=('perturbed','mean'),mean_effect=('perturbation_effect','mean'),
        worsening=('perturbation_effect',lambda v:int((v< -1e-12).sum()))).to_csv(out/'summary.csv')
    d.save(out/'failures.json',failures)
    d.save(out/'input_sha256.json',hashes)
    d.state(status='stage_complete',stage='task4_source_agreement_corruption_complete',
        next='Independently verify source invariants and matched target movement before publication.')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
