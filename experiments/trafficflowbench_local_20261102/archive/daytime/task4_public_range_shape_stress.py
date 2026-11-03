"""Stress fixed prior forecasts at a public-range observable count shape mismatch."""
import hashlib
import msvcrt
from datetime import datetime, timezone

import common as d
import task4_count_source_pair_stress as g
import numpy as np
import pandas as pd
from scipy.optimize import brentq

SEED = 2026110331
TARGET_TV = .14
NOISES = ['shared_lognormal', 'shared_positive_background']


def tilt_prior(A, count, prior, position):
    total = count.sum()
    observed_shape = count / total
    coverage = A.sum(axis=0)

    def tilted(strength):
        flow = prior * np.exp(strength * position)
        return flow * (total / (coverage @ flow))

    def excess(strength):
        return .5 * np.abs(A @ tilted(strength) / total - observed_shape).sum() - TARGET_TV

    if excess(0.) >= 0 or excess(24.) < 0:
        return None
    strength = brentq(excess, 0., 24., xtol=1e-10)
    return tilted(strength), strength


def main():
    out = d.OUT/'task4_public_range_shape_stress'
    d.save(out/'protocol.json', dict(
        utc=datetime.now(timezone.utc).isoformat(), seed=SEED, cases=80,
        noise_families=NOISES, drift='moderate_drift', folds=5, target_count_shape_TV=TARGET_TV,
        methods=['baseline','geometry005','pair005'],
        generator='Normalize bulk count total, then use a monotone spatial exponential tilt to hit count-shape TV .14 per input. Alternate signed direction by case seed parity. Retain unbracketed roots as failures.',
        rationale='Public max-across-input TV ranged .106 to .190 versus at most .085 in previous artificial cases. Target .14 was chosen from that exposed diagnostic.',
        scope='Matches one observable statistic under an artificial error mechanism. No hidden-label inference, whole-system blind evaluation or deployment selection.',
        source_sha256=d.sha(__file__), generator_sha256=d.sha(g.__file__)))
    rows, failures, hashes = [], [], {}
    for panel in d.night.predictor.load_routes():
        paths, A, count, _, inputs = g.family.base.load(panel,'train')
        hashes.update(inputs)
        X = g.family.structure.features(paths)['span_origin_destination']
        ends = [paths[key].str.extract(r'^Z(\d+)$',expand=False).astype(float).to_numpy()
                for key in ['origin_zone','destination_zone']]
        midpoint = sum(ends)/2
        position = (midpoint-midpoint.min()) / np.ptp(midpoint)
        level = float(np.median(count[count>0]))
        for truth_family in g.family.TRUTHS:
            for noise in NOISES:
                case = f'{panel}/{truth_family}/{noise}'
                seed = int.from_bytes(hashlib.sha256(f'{SEED}/{case}'.encode()).digest()[:8],'little')%(2**63)
                rng = np.random.default_rng(seed)
                truth, prior, counts, dest = g.draw(paths,A,level,truth_family,noise,'moderate_drift',rng)
                direction = position if seed%2 else 1-position
                variants = [tilt_prior(A,c,b,direction) for c,b in zip(counts,prior)]
                if any(v is None for v in variants):
                    failures.append(dict(case=case,panel=panel,seed=seed,reason='TV root not bracketed', failed_inputs=[i for i,v in enumerate(variants) if v is None]))
                    continue
                b = np.array([v[0] for v in variants])
                fold = np.empty(len(paths),int)
                fold[rng.permutation(len(paths))] = np.arange(len(paths))%5
                forecasts, coefs, scales, _, _, _ = g.crossfit(X,A,counts,b,fold)
                loaded = np.array([A@prior_group for prior_group in b])
                tv = .5*np.abs(loaded/loaded.sum(axis=1,keepdims=True)-counts/counts.sum(axis=1,keepdims=True)).sum(axis=1)
                bulk_bias = np.max(np.abs(loaded.sum(axis=1)/counts.sum(axis=1)-1))
                assert np.max(np.abs(tv-TARGET_TV)) < 1e-9 and bulk_bias < 1e-12
                flows = {}
                for method, prior_in in [('baseline',b[2]),('geometry005',.95*b[2]+.05*forecasts['geometry']),
                                         ('pair005',.95*b[2]+.05*forecasts['raw_pair'])]:
                    flow,status = g.family.base.ridge(A,counts[2],prior_in)
                    assert status['success'], (case,method)
                    flows[method] = flow
                folder = d.WORK/'task4_public_range_shape_cases'/hashlib.sha256(case.encode()).hexdigest()[:16]
                folder.mkdir(parents=True,exist_ok=True)
                file = folder/'case.npz'
                np.savez_compressed(file, seed=seed, truth=truth, supplied_prior=b, original_prior=prior, counts=counts,
                                    fold=fold, scales=scales, tilt=[v[1] for v in variants],
                                    geometry_coef=coefs['geometry'], pair_coef=coefs['raw_pair'], **flows)
                digest = d.sha(file)
                for method,flow in flows.items():
                    rows.append(dict(case=case,panel=panel,truth_family=truth_family,noise=noise,seed=seed,method=method,
                                     max_bulk_bias=bulk_bias,max_shape_tv=float(tv.max()),artifact_sha256=digest,
                                     **g.family.base.metrics(flow,truth[2],b[2],A,counts[2],dest)))
        pd.DataFrame(rows).to_csv(out/'all_cases.csv',index=False)
        d.save(out/'failures.json',failures)
        d.state(status='running',stage='public_range_shape_stress',panel=panel)
    frame = pd.DataFrame(rows)
    assert frame.case.nunique()+len(failures)==80
    reference = frame[frame.method=='baseline'].set_index('case')
    summary = []
    for (noise,method), group in frame.groupby(['noise','method']):
        q = group.set_index('case')
        delta = q.synthetic_S_ODME-reference.loc[q.index].synthetic_S_ODME
        od = q.synthetic_S_od-reference.loc[q.index].synthetic_S_od
        summary.append(dict(noise=noise,method=method,cases=len(q),mean_delta=float(delta.mean()),
                            worst_delta=float(delta.min()),composite_harms=int((delta < -1e-12).sum()),
                            od_harms=int((od < -1e-12).sum())))
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    d.save(out/'input_sha256.json',hashes)
    d.state(status='stage_complete',stage='public_range_shape_stress_complete',next='Independently read back scores and verify portable reproduction before publication.')


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
