"""Test a fixed count-total gate under globally scaled positive background."""
import hashlib
import msvcrt
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import common as d
import task4_count_source_pair_stress as g

SEED = 2026110330


def main():
    out = d.OUT/'task4_bias_cancellation_stress'
    d.save(out/'protocol.json', dict(
        registered_utc=datetime.now(timezone.utc).isoformat(), seed=SEED, cases=40,
        drift='moderate_drift', noise='shared_positive_background', threshold=.1, strength=.05,
        hypothesis='Bulk count normalization can hide positive background from the fixed one-sided gate.',
        control='Same truth and folds with original supplied-style prior versus globally rescaled prior. Rescaling uses only observable total counts for each input.',
        scope='Constructed artificial mechanism, not a public generation model or official validation.',
        source_sha256=d.sha(__file__), generator_sha256=d.sha(g.__file__)))
    rows, hashes = [], {}
    for panel in d.night.predictor.load_routes():
        paths, A, count, _, sources = g.family.base.load(panel, 'train')
        hashes.update(sources)
        X = g.family.structure.features(paths)['span_origin_destination']
        level = float(np.median(count[count > 0]))
        for family in g.family.TRUTHS:
            case = f'{panel}/{family}'
            seed = int.from_bytes(hashlib.sha256(f'{SEED}/{case}'.encode()).digest()[:8], 'little') % (2**63)
            rng = np.random.default_rng(seed)
            truth, prior, counts, dest = g.draw(paths, A, level, family, 'shared_positive_background', 'moderate_drift', rng)
            fold = np.empty(len(paths), int)
            fold[rng.permutation(len(paths))] = np.arange(len(paths)) % 5
            loaded = A.sum(axis=0) @ prior.T
            normalized = prior * (counts.sum(axis=1)/loaded)[:, None]
            for variant, b in [('original', prior), ('bulk_normalized', normalized)]:
                bias = (A.sum(axis=0) @ b.T)/counts.sum(axis=1)-1
                admitted = bool(bias.max() <= .1)
                forecasts, coefs, scales, _, success, _ = g.crossfit(X, A, counts, b, fold)
                assert success
                f0, baseline_status = g.family.base.ridge(A, counts[2], b[2])
                f1, direct_status = g.family.base.ridge(A, counts[2], .95*b[2]+.05*forecasts['raw_pair'])
                assert baseline_status['success'] and direct_status['success']
                folder = d.WORK/'task4_bias_cancellation_cases'/panel/family
                folder.mkdir(parents=True, exist_ok=True)
                file = folder/f'{variant}.npz'
                np.savez_compressed(file, seed=seed, truth=truth, counts=counts, prior=b, fold=fold,
                                    coefficients=coefs['raw_pair'], scales=scales, baseline=f0, direct=f1)
                for method, flow in [('baseline', f0), ('raw005', f1), ('gated005', f1 if admitted else f0)]:
                    rows.append(dict(case=case, panel=panel, family=family, variant=variant, method=method,
                                     seed=seed, gate_enabled=admitted, max_count_bias=float(bias.max()),
                                     artifact_sha256=d.sha(file), **g.family.base.metrics(flow, truth[2], b[2], A, counts[2], dest)))
        pd.DataFrame(rows).to_csv(out/'all_cases.csv', index=False)
        d.state(status='running', stage='bias_cancellation_stress', panel=panel)
    frame = pd.DataFrame(rows)
    paired = frame.pivot(index=['case', 'variant'], columns='method', values='synthetic_S_ODME')
    delta = (paired.gated005 - paired.baseline).rename('delta').reset_index()
    delta.groupby('variant').delta.agg(['count','mean','min',lambda x: int((x < -1e-12).sum())]).rename(columns={'<lambda_0>':'harms'}).to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json', hashes)
    d.state(status='stage_complete', stage='bias_cancellation_stress_complete', next='Verify the paired control and compare gated harm within each supplied-prior variant. Do not tune the threshold.')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        main()
