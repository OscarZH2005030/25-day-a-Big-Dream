"""Compare fixed Huber and squared-loss prior forecasts on fresh shape stress."""
import hashlib
import msvcrt
from datetime import datetime, timezone

import common as d
import task4_count_source_pair_stress as g
from task4_public_range_shape_stress import tilt_prior, TARGET_TV
import numpy as np
import pandas as pd

SEED = 2026110332
FAMILIES = ['origin_destination', 'sparse']
NOISES = ['shared_lognormal', 'shared_positive_background']


def huber_objective(design, target, coef):
    residual = np.abs(design@coef-target)
    loss = np.where(residual<=1., .5*residual**2, residual-.5)
    return float(loss.sum()+.5*np.sum(coef[1:]**2))


def fit_forecasts(X, priors, fold):
    source = np.mean([b/max(float(np.median(b[b>0])),1e-9) for b in priors[:2]],axis=0)
    design = np.column_stack([X,source-X@g.qr.ridge_qr(X,source)])
    prior = priors[2]
    forecasts = {name:np.zeros(len(prior)) for name in ['squared005','huber005']}
    coefs = {name:[] for name in forecasts}
    scales, objective_drops = [], []
    for k in range(5):
        train, test = fold!=k, fold==k
        scale = max(float(np.median(prior[train & (prior>0)])),1e-9)
        scales.append(scale)
        features, target = design[train],prior[train]/scale
        squared = g.qr.ridge_qr(features,target)
        robust = squared.copy()
        initial = previous = huber_objective(features,target,robust)
        for _ in range(8):
            residual = np.abs(features@robust-target)
            root_weight = 1/np.sqrt(np.maximum(1.,residual))
            robust = g.qr.ridge_qr(features*root_weight[:,None],target*root_weight)
            current = huber_objective(features,target,robust)
            assert current<=previous+1e-9*max(1.,previous)
            previous = current
        objective_drops.append(initial-previous)
        for name,coef in [('squared005',squared),('huber005',robust)]:
            coefs[name].append(coef)
            forecasts[name][test]=np.maximum(design[test]@coef,0)*scale
    return forecasts,coefs,scales,objective_drops


def main():
    out = d.OUT/'task4_robust_prior_forecast'
    d.save(out/'protocol.json',dict(
        registered_utc=datetime.now(timezone.utc).isoformat(), seed=SEED, cases=40,
        truth_families=FAMILIES, noise_families=NOISES, drift='moderate_drift',
        shape_tv=TARGET_TV, folds=5, blend=.05,
        hypothesis='Limiting large target-prior residual influence may reduce fragile cross-input forecasts.',
        design='Identical squared and Huber features and folds. Ridge penalty 1 excludes intercept. Fixed eight IRLS iterations with Huber delta 1 in training-median units. No convergence or optimality claim.',
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
                forecasts,coefs,scales,drops=fit_forecasts(X,b,fold)
                flows={}
                for method,prior_input in [('baseline',b[2]),*((name,.95*b[2]+.05*forecast) for name,forecast in forecasts.items())]:
                    flow,status=g.family.base.ridge(A,counts[2],prior_input)
                    if status['success']:
                        flows[method]=flow
                    else:
                        failures.append(dict(case=case,seed=seed,method=method,reason='flow solver failure'))
                folder=d.WORK/'task4_robust_prior_cases'/hashlib.sha256(case.encode()).hexdigest()[:16]
                folder.mkdir(parents=True,exist_ok=True)
                file=folder/'case.npz'
                np.savez_compressed(file,seed=seed,truth=truth,prior=b,counts=counts,fold=fold,
                                    scales=scales,huber_objective_drops=drops,
                                    squared_coef=coefs['squared005'],huber_coef=coefs['huber005'],**flows)
                digest=d.sha(file)
                for method,flow in flows.items():
                    rows.append(dict(case=case,panel=panel,family=family,noise=noise,seed=seed,method=method,
                                     artifact_sha256=digest,**g.family.base.metrics(flow,truth[2],b[2],A,counts[2],dest)))
        pd.DataFrame(rows).to_csv(out/'all_cases.csv',index=False)
        d.save(out/'failures.json',failures)
        d.state(status='running',stage='robust_prior_forecast',panel=panel)
    frame=pd.DataFrame(rows)
    wide=frame.pivot(index=['noise','case'],columns='method',values='synthetic_S_ODME')
    deltas=wide[['squared005','huber005']].sub(wide.baseline,axis=0)
    deltas.groupby('noise').agg(['count','mean','min',lambda s:int((s < -1e-12).sum())]).to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.state(status='stage_complete',stage='robust_prior_forecast_complete',next='Verify scores, fixed IRLS iterations and portable run. Retain all harms and avoid selecting robust loss from this stress alone.')


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
