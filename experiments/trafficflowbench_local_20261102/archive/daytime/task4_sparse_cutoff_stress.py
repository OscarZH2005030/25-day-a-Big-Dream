"""Isolate the Huber cutoff on fresh sparse demand stress."""
import hashlib
import msvcrt
from datetime import datetime, timezone

import common as d
import task4_count_source_pair_stress as g
from task4_public_range_shape_stress import tilt_prior, TARGET_TV
import numpy as np
import pandas as pd

SEED = 2026110333
FAMILIES = ['sparse']
NOISES = ['shared_lognormal', 'shared_positive_background']


def huber_objective(design, target, coef, cutoff):
    residual = np.abs(design@coef-target)
    loss = np.where(residual<=cutoff, .5*residual**2, cutoff*(residual-.5*cutoff))
    return float(loss.sum()+.5*np.sum(coef[1:]**2))


def fit_forecasts(X, priors, fold):
    source = np.mean([b/max(float(np.median(b[b>0])),1e-9) for b in priors[:2]],axis=0)
    design = np.column_stack([X,source-X@g.qr.ridge_qr(X,source)])
    prior = priors[2]
    forecasts = {name:np.zeros(len(prior)) for name in ['squared005','median_cutoff005','mean_floor005']}
    coefs = {name:[] for name in forecasts}
    scales, diagnostics = [], []
    for k in range(5):
        train, test = fold!=k, fold==k
        scale = max(float(np.median(prior[train & (prior>0)])),1e-9)
        scales.append(scale)
        features, target = design[train],prior[train]/scale
        squared = g.qr.ridge_qr(features,target)
        # Keep the same normalized labels and ridge penalty for both Huber fits.
        floor = max(1., float(target.mean()))
        fitted = [('squared005', squared)]
        for name, cutoff in [('median_cutoff005', 1.), ('mean_floor005', floor)]:
            robust = squared.copy()
            initial = previous = huber_objective(features, target, robust, cutoff)
            for _ in range(8):
                residual = np.abs(features @ robust - target)
                root_weight = 1 / np.sqrt(np.maximum(1., residual / cutoff))
                robust = g.qr.ridge_qr(features * root_weight[:, None], target * root_weight)
                current = huber_objective(features, target, robust, cutoff)
                assert current <= previous + 1e-9 * max(1., previous)
                previous = current
            diagnostics.append((k, cutoff * scale, initial, previous))
            fitted.append((name, robust))
        for name, coef in fitted:
            coefs[name].append(coef)
            forecasts[name][test]=np.maximum(design[test]@coef,0)*scale
    return forecasts,coefs,scales,diagnostics


def main():
    out = d.OUT/'task4_sparse_cutoff_stress'
    d.save(out/'protocol.json',dict(
        registered_utc=datetime.now(timezone.utc).isoformat(), seed=SEED, cases=20,
        truth_families=FAMILIES, noise_families=NOISES, drift='moderate_drift',
        shape_tv=TARGET_TV, folds=5, blend=.05,
        hypothesis='A floor at the fold training mean can prevent the physical Huber cutoff collapsing at a sparse median.',
        design='Same features, target normalization, folds and ridge penalty. Fixed eight IRLS steps. Physical cutoff equals median or max(median, mean), each computed on the target training fold.',
        scope='New seed on previously studied artificial mechanisms. No tuning against hidden OD, blind-system or official score claim.',
        source_sha256=d.sha(__file__), tilt_source_sha256=d.sha(d.WORK/'task4_public_range_shape_stress.py'),
        generator_sha256=d.sha(g.__file__)))
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
                forecasts,coefs,scales,diagnostics=fit_forecasts(X,b,fold)
                flows={}
                for method,prior_input in [('baseline',b[2]),*((name,.95*b[2]+.05*forecast) for name,forecast in forecasts.items())]:
                    flow,status=g.family.base.ridge(A,counts[2],prior_input)
                    if status['success']:
                        flows[method]=flow
                    else:
                        failures.append(dict(case=case,seed=seed,method=method,reason='flow solver failure'))
                folder=d.WORK/'task4_sparse_cutoff_cases'/hashlib.sha256(case.encode()).hexdigest()[:16]
                folder.mkdir(parents=True,exist_ok=True)
                file=folder/'case.npz'
                np.savez_compressed(file,seed=seed,truth=truth,prior=b,counts=counts,fold=fold,
                                    scales=scales,cutoff_diagnostics=diagnostics,
                                    **{name+'_coef':coef for name,coef in coefs.items()},**flows)
                digest=d.sha(file)
                for method,flow in flows.items():
                    rows.append(dict(case=case,panel=panel,family=family,noise=noise,seed=seed,method=method,
                                     artifact_sha256=digest,**g.family.base.metrics(flow,truth[2],b[2],A,counts[2],dest)))
        pd.DataFrame(rows).to_csv(out/'all_cases.csv',index=False)
        d.save(out/'failures.json',failures)
        d.state(status='running',stage='sparse_cutoff_stress',panel=panel)
    frame=pd.DataFrame(rows)
    wide=frame.pivot(index=['noise','case'],columns='method',values='synthetic_S_ODME')
    deltas=wide[['squared005','median_cutoff005','mean_floor005']].sub(wide.baseline,axis=0)
    summary = []
    for noise, group in deltas.groupby(level='noise'):
        for method in group.columns:
            delta = group[method].dropna()
            summary.append(dict(noise=noise, method=method, cases=len(delta),
                                mean_delta=float(delta.mean()), worst_delta=float(delta.min()),
                                harms=int((delta < -1e-12).sum())))
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    d.save(out/'input_sha256.json',hashes)
    d.state(status='stage_complete',stage='sparse_cutoff_stress_complete',next='Verify scores, fixed IRLS iterations and portable run. Retain all harms and avoid selecting robust loss from this stress alone.')


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
