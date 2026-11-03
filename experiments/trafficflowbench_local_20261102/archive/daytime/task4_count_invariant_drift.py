"""Change latent target pairing while fixing target counts and both margins."""
import hashlib
import msvcrt
from datetime import datetime, timezone

import common as d
import numpy as np
import pandas as pd
from od_cycle_refinement import draw_cycles, path_lookup
from task4_source_clip_stress import fit_forecasts, g, tilt_prior

SEED = 2026110340
FAMILIES = ['origin_destination','sparse']
NOISES = ['shared_lognormal','shared_positive_background']


def rearrange(flow, cycles, rng):
    moved = flow.copy()
    for a,b,c,e in cycles:
        lo = -min(moved[a],moved[e])
        hi = min(moved[b],moved[c])
        step = rng.uniform(lo,hi)
        moved[[a,b,c,e]] += np.array([step,-step,-step,step])
    return moved


def main():
    out = d.OUT/'task4_count_invariant_drift'
    d.save(out/'protocol.json',dict(
        registered_utc=datetime.now(timezone.utc).isoformat(),seed=SEED,paired_cases=40,
        families=FAMILIES,noises=NOISES,folds=5,blend=.05,
        hypothesis='Source clipping can react differently to hidden target re-pairing when counts and margins are fixed.',
        construction='Moderate shared-truth drift first. Change target alone with 4*n bounded random four-cell exchanges. Use identical source priors, target count vector, target error draws and folds across paired versions. Construct each supplied prior with TV .14; target tilt is recalibrated per version.',
        scope='Paired artificial falsification on exposed networks. Target priors change with the latent pairing. This is not an indistinguishable-observables certificate or hidden OD validation.',
        source_sha256=d.sha(__file__),forecast_sha256=d.sha(d.WORK/'task4_source_clip_stress.py'),
        cycles_sha256=d.sha(d.WORK/'od_cycle_refinement.py')))
    rows,failures,hashes=[],[],{}
    for panel in d.night.predictor.load_routes():
        paths,A,counts_ref,_,inputs=g.family.base.load(panel,'train')
        hashes.update(inputs)
        X=g.family.structure.features(paths)['span_origin_destination']
        level=float(np.median(counts_ref[counts_ref>0]))
        for family in FAMILIES:
            for noise in NOISES:
                case=f'{panel}/{family}/{noise}'
                seed=int.from_bytes(hashlib.sha256(f'{SEED}/{case}'.encode()).digest()[:8],'little')%(2**63)
                rng=np.random.default_rng(seed)
                common,origin,dest=g.family.base.truth_draw(paths,A,level,family,rng)
                n=len(common)
                truth=np.array([common*rng.lognormal(-.35**2/2,.35,n)*m for m in [.85,1.,1.15]])
                counts=truth@A.T
                cycles=draw_cycles(path_lookup(origin,dest),4*n,rng)
                alternate=rearrange(truth[2],cycles,rng)
                delta=alternate-truth[2]
                tolerance=1e-10*max(1.,truth[2].sum())
                assert alternate.min()>=-1e-12
                assert max(np.max(np.abs(A@delta)),np.max(np.abs(np.bincount(origin,weights=delta))),
                           np.max(np.abs(np.bincount(dest,weights=delta))))<tolerance
                if noise=='shared_lognormal':
                    factors=np.exp(rng.normal(-.6**2/2,.6,n)+rng.normal(-.2**2/2,.2,(3,n)))
                    priors=truth*factors
                    alt_prior=alternate*factors[2]
                else:
                    background=.8*rng.lognormal(-.7**2/2,.7,n)+.1*rng.lognormal(-.7**2/2,.7,(3,n))
                    priors=truth+truth.mean(axis=1,keepdims=True)*background
                    alt_prior=alternate+truth[2].mean()*background[2]
                midpoint=(origin+dest)/2
                position=(midpoint-midpoint.min())/np.ptp(midpoint)
                if seed%2==0:
                    position=1-position
                prepared=[tilt_prior(A,count,np.maximum(prior,1e-6),position)
                          for count,prior in zip([*counts,counts[2]],[*priors,alt_prior])]
                if any(item is None for item in prepared):
                    failures.append(dict(case=case,seed=seed,reason='TV root not bracketed'))
                    continue
                fold=np.empty(n,int)
                fold[rng.permutation(n)]=np.arange(n)%5
                folder=d.WORK/'task4_count_invariant_drift_cases'/hashlib.sha256(case.encode()).hexdigest()[:16]
                folder.mkdir(parents=True,exist_ok=True)
                for variant,target,item in [('moderate',truth[2],prepared[2]),('repaired',alternate,prepared[3])]:
                    b=np.array([prepared[0][0],prepared[1][0],item[0]])
                    forecast,coefs,scales,pair_scales,cutoff=fit_forecasts(X,b,fold)
                    flows={}
                    for method,prior in [('baseline',b[2]),*((name,.95*b[2]+.05*value) for name,value in forecast.items())]:
                        flow,status=g.family.base.ridge(A,counts[2],prior)
                        if status['success']:
                            flows[method]=flow
                        else:
                            failures.append(dict(case=case,seed=seed,variant=variant,method=method,reason='flow solver'))
                    file=folder/f'{variant}.npz'
                    np.savez_compressed(file,seed=seed,truth=target,counts=counts,prior=b,fold=fold,
                                        scales=scales,cutoff=cutoff,cycles=cycles,
                                        **{k+'_coef':v for k,v in coefs.items()},
                                        **{k+'_rms':v for k,v in pair_scales.items()},**flows)
                    digest=d.sha(file)
                    for method,flow in flows.items():
                        rows.append(dict(case=case,panel=panel,family=family,noise=noise,variant=variant,method=method,
                                         seed=seed,relative_target_change=float(np.abs(delta).sum()/truth[2].sum()),
                                         artifact_sha256=digest,
                                         **g.family.base.metrics(flow,target,b[2],A,counts[2],dest)))
        pd.DataFrame(rows).to_csv(out/'all_cases.csv',index=False)
        d.save(out/'failures.json',failures)
        d.state(status='running',stage='count_invariant_drift',panel=panel)
    frame=pd.DataFrame(rows)
    wide=frame.pivot(index=['noise','variant','case'],columns='method',values='synthetic_S_ODME')
    delta=wide[['raw005','clipped005']].sub(wide.baseline,axis=0)
    delta.groupby(level=['noise','variant']).agg(['count','mean','min',lambda x:int((x<-1e-12).sum())]).to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.state(status='stage_complete',stage='count_invariant_drift_complete',next='Verify paired invariants and portable runner before interpreting results.')


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
