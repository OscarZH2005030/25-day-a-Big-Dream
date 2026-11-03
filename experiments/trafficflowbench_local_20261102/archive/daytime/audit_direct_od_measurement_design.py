"""Compare hypothetical direct OD measurement designs on released networks."""
import hashlib
import msvcrt

import numpy as np
import pandas as pd

import common as d
from task4_exact_pair_audit import base, basis
from task4_extra_measurement_audit import indicators

SEED = 2026110354


def design(null_features, order, weights, budgets):
    residual = null_features.copy()
    initial = np.einsum('ij,ij->i', residual, residual)
    selected = []
    used = np.zeros(len(initial), dtype=bool)
    results = {}
    for step in range(1, max(budgets)+1):
        diagonal = np.einsum('ij,ij->i', residual, residual)
        if order is None:
            # Greedy trace reduction under prior-weighted squared uncertainty.
            covariance = residual @ residual.T
            gain = np.divide(weights @ (covariance**2), diagonal,
                             out=np.zeros_like(diagonal), where=diagonal>1e-12)
            gain[used] = -1
            j = int(np.argmax(gain))
        else:
            j = int(order[step-1])
        used[j] = True
        if diagonal[j] > 1e-12:
            direction = residual[j] / np.sqrt(diagonal[j])
            residual -= np.outer(residual @ direction, direction)
            selected.append(j)
        if step in budgets:
            final = np.einsum('ij,ij->i', residual, residual)
            results[step] = (len(selected), final.sum()/initial.sum(), (weights @ final)/(weights @ initial))
    return results


def main():
    out = d.OUT/'task4_direct_od_measurement_design'
    budgets = [4,16]
    d.save(out/'protocol.json', dict(seed=SEED, splits=['validation','private'], budgets=budgets,
        code_sha256=d.sha(__file__), design='Condition a unit isotropic OD perturbation on exact public count and hypothetical gross margin rows. Compare direct OD cell measurements by uniform ordering, top weak prior and greedy weak-prior-weighted trace reduction.',
        scope='Hypothetical linear measurement design. No direct OD labels are available or imposed, and isotropic variance is a diagnostic geometry.'))
    rows, hashes, failures = [], {}, []
    for panel in d.night.predictor.load_routes():
        for split in ('validation','private'):
            paths,A,_,prior,provenance = base.load(panel,split)
            hashes.update(provenance)
            O,D = [indicators(paths[k]).toarray() for k in ('origin_zone','destination_zone')]
            Q, rank = basis(np.vstack([A,O,D]).T)
            projector = np.eye(len(paths)) - Q @ Q.T
            np.testing.assert_allclose(np.trace(projector), len(paths)-rank, atol=1e-7,rtol=0)
            weights = np.asarray(prior, dtype=float)
            assert weights.shape == (len(paths),) and weights.min() >= 0 and weights.sum()>0
            weights = weights / weights.sum()
            seed = int.from_bytes(hashlib.sha256(f'{SEED}/{panel}/{split}'.encode()).digest()[:8], 'little')
            rng = np.random.default_rng(seed)
            variants = [('uniform',rng.permutation(len(paths)),i) for i in range(8)]
            variants += [('top_prior',np.argsort(-weights,kind='stable'),0),('greedy_weighted_trace',None,0)]
            for method, order, repeat in variants:
                for budget, (gain, trace, weighted) in design(projector,order,weights,budgets).items():
                    assert 0 <= trace <= 1+1e-8 and 0 <= weighted <= 1+1e-8
                    rows.append(dict(panel=panel,split=split,seed=seed,repeat=repeat,method=method,budget=budget,
                                     paths=len(paths),starting_nullity=len(paths)-rank,rank_gain=gain,
                                     remaining_trace_ratio=trace,remaining_weighted_trace_ratio=weighted))
            print(panel,split,flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'by_panel.csv',index=False)
    frame.groupby(['split','method','budget'])[['rank_gain','remaining_trace_ratio','remaining_weighted_trace_ratio']].mean().to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',failures)
    d.state(status='stage_complete',stage='task4_direct_od_measurement_design_complete',
            next='Check linear algebra independently and assess sensitivity to prior weighting.')


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
