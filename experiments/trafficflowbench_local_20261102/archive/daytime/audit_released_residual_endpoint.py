"""Check covariance-shape transfer on released Task4 count residuals."""
import msvcrt

import numpy as np
import pandas as pd
from scipy.linalg import null_space
from scipy.linalg import eigh

import common as d
from audit_direct_od_measurement_design import base


def independent_count_rows(counts):
    # Fix the original observation coordinates for seeded draws.
    orthogonal = []
    selected = []
    for i, row in enumerate(counts):
        residual = row.astype(float)
        for _ in range(2):
            for vector in orthogonal:
                residual -= (residual @ vector) * vector
        norm = np.linalg.norm(residual)
        if norm > 1e-8:
            orthogonal.append(residual / norm)
            selected.append(i)
    return selected




SHARES = (.99, .999, .9999, .999999, 1.)


def evaluate(panel, split):
    paths, incidence, counts, prior, hashes = base.load(panel, split)
    rows = independent_count_rows(incidence)
    matrix = incidence[rows]
    residual = counts[rows] - matrix @ prior
    origins, destinations = [pd.get_dummies(paths[key]).to_numpy(dtype=float).T
                             for key in ('origin_zone', 'destination_zone')]
    oz, dz = [paths[key].str.extract(r'^Z(\d+)$', expand=False).astype(float).to_numpy()
              for key in ('origin_zone', 'destination_zone')]
    span = dz - oz
    directions = matrix @ np.column_stack((prior, prior * (span - prior @ span / prior.sum())))
    contrast = null_space(directions.T).T
    projected = matrix @ np.vstack((origins,destinations)).T
    shared = .5 * projected @ projected.T
    independent = matrix @ matrix.T
    results = []
    for name, view in [('raw', np.eye(len(rows))), ('prior_and_span',contrast)]:
        white = view @ independent @ view.T
        covariance = view @ shared @ view.T
        values, vectors = eigh(covariance, white)
        positive = values > values[-1] * 1e-10
        rank = int(positive.sum())
        assert rank>0 and values.min()>=-1e-9
        energy = (vectors.T @ view @ residual)**2
        for share in SHARES:
            spectrum = (1-share)+share*np.maximum(values,0)
            score = np.mean(np.log(spectrum)) + np.log(np.mean(energy/spectrum)) if share<1 or positive.all() else None
            results.append(dict(panel=panel, split=split, view=name, fraction=share,
                dimension=len(values), endpoint_rank=rank,
                off_support_energy_share=energy[~positive].sum()/energy.sum(),
                min_positive_endpoint_eigenvalue=values[positive].min(),
                max_endpoint_eigenvalue=values.max(), score_without_basis_constant=score))
    return results, hashes


def main():
    out = d.OUT/'task4_released_residual_endpoint'
    d.save(out/'protocol.json',dict(source_sha256=d.sha(__file__), splits=['train','validation'],
        fractions=SHARES,power=0.,origin_share=.5,scope='Released residual score at the exposed boundary, using a fixed generalized eigenspace. No hidden OD inference.'))
    rows, hashes = [], {}
    for panel in sorted(d.night.predictor.load_routes()):
        for split in ('train','validation'):
            result, provenance = evaluate(panel, split)
            rows.extend(result)
            hashes.update(provenance)
    data=pd.DataFrame(rows)
    data.to_csv(out/'by_panel.csv',index=False)
    data.groupby(['split','view','fraction']).score_without_basis_constant.agg(['count','mean','min','max']).to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',[])


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
