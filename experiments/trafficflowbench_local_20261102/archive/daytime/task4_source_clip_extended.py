"""Compare source-residual clipping under two shared error mechanisms."""
import hashlib
import msvcrt
from datetime import datetime, timezone

import common as d
import task4_count_source_pair_stress as g
from task4_public_range_shape_stress import tilt_prior, TARGET_TV
from task4_source_clip_stress import fit_forecasts
import numpy as np
import pandas as pd

SEED = 2026110338
FAMILIES = g.family.TRUTHS
NOISES = g.family.NOISES


def main():
    w_source = d.WORK/'task4_source_clip_stress.py'
    out = d.OUT/'task4_source_clip_extended'
    d.save(out/'protocol.json',dict(
        registered_utc=datetime.now(timezone.utc).isoformat(), seed=SEED, cases=160,
        truth_families=FAMILIES, noise_families=NOISES, drift='moderate_drift',
        shape_tv=TARGET_TV, folds=5, blend=.05,
        hypothesis='Clipping extreme source interactions may limit error transfer while retaining persistent matching structure.',
        design='Raw exact geometry-orthogonal source versus symmetric clipping at its 95th absolute percentile followed by exact re-projection. Both columns scaled to unit RMS within target training fold. Same ridge1, five folds and 5% correction. Cutoff uses source inputs only.',
        scope='New seed on previously studied artificial mechanisms. No tuning against hidden OD, blind-system or official score claim.',
        source_sha256=d.sha(__file__), tilt_source_sha256=d.sha(d.WORK/'task4_public_range_shape_stress.py'),
        generator_sha256=d.sha(g.__file__), frozen_forecast_source_sha256=d.sha(w_source)))
    rows, failures, hashes = [], [], {}
    for panel in d.night.predictor.load_routes():
        paths,A,count,_,inputs=g.family.base.load(panel,'train')
        hashes.update(inputs)
        X=g.family.structure.features(paths)['span_origin_destination']
        ends=[paths[key].str.extract(r'^Z(\d+)$',expand=False).astype(float).to_numpy()
              for key in ['origin_zone','destination_zone']]
        midpoint=sum(ends)/2
        position=(midpoint-midpoint.min())/np.ptp(midpoint)
        for family in FAMILIES:
            for noise in NOISES:
                case=f'{panel}/{family}/{noise}'
                seed=int.from_bytes(hashlib.sha256(f'{SEED}/{case}'.encode()).digest()[:8],'little')%(2**63)
                rng=np.random.default_rng(seed)
                truth,prior,counts,dest=g.draw(paths,A,float(np.median(count[count>0])),family,noise,'moderate_drift',rng)
                variants=[tilt_prior(A,c,b,position if seed%2 else 1-position) for c,b in zip(counts,prior)]
                if any(item is None for item in variants):
                    failures.append(dict(case=case,seed=seed,reason='TV root not bracketed'))
                    continue
                b=np.array([item[0] for item in variants])
                fold=np.empty(len(paths),int)
                fold[rng.permutation(len(paths))]=np.arange(len(paths))%5
                forecasts,coefs,scales,pair_scales,cutoff=fit_forecasts(X,b,fold)
                flows={}
                for method,prior_input in [('baseline',b[2]),*((name,.95*b[2]+.05*forecast) for name,forecast in forecasts.items())]:
                    flow,status=g.family.base.ridge(A,counts[2],prior_input)
                    if status['success']:
                        flows[method]=flow
                    else:
                        failures.append(dict(case=case,seed=seed,method=method,reason='flow solver failure'))
                folder=d.WORK/'task4_source_clip_extended_cases'/hashlib.sha256(case.encode()).hexdigest()[:16]
                folder.mkdir(parents=True,exist_ok=True)
                file=folder/'case.npz'
                np.savez_compressed(file,seed=seed,truth=truth,prior=b,counts=counts,fold=fold,
                                    scales=scales,cutoff=cutoff,
                                    **{name+'_coef':coef for name,coef in coefs.items()},
                                    **{name+'_rms':rms for name,rms in pair_scales.items()},**flows)
                digest=d.sha(file)
                for method,flow in flows.items():
                    rows.append(dict(case=case,panel=panel,family=family,noise=noise,seed=seed,method=method,
                                     artifact_sha256=digest,**g.family.base.metrics(flow,truth[2],b[2],A,counts[2],dest)))
        pd.DataFrame(rows).to_csv(out/'all_cases.csv',index=False)
        d.save(out/'failures.json',failures)
        d.state(status='running',stage='source_clip_extended',panel=panel)
    frame=pd.DataFrame(rows)
    wide=frame.pivot(index=['noise','case'],columns='method',values='synthetic_S_ODME')
    deltas=wide[['raw005','clipped005']].sub(wide.baseline,axis=0)
    summary = []
    for noise, group in deltas.groupby(level='noise'):
        for method in group.columns:
            delta = group[method].dropna()
            summary.append(dict(noise=noise,method=method,cases=len(delta),mean_delta=float(delta.mean()),
                                worst_delta=float(delta.min()),harms=int((delta < -1e-12).sum())))
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    d.save(out/'input_sha256.json',hashes)
    d.state(status='stage_complete',stage='source_clip_extended_complete',next='Verify scores, source cutoff and fold-local scaling and portable run. Retain all harms and avoid selecting a penalty from this stress alone.')


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
