"""Check covariance-shape transfer on released Task4 count residuals."""
import msvcrt

import numpy as np
import pandas as pd
from scipy.linalg import null_space
from scipy.linalg import cholesky, solve_triangular

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



BASELINE = (.5, .25, .5)


def score_split(paths, incidence, counts, prior, weights):
    rows = independent_count_rows(incidence)
    matrix = incidence[rows]
    residual = counts[rows] - matrix @ prior
    origins, destinations = [pd.get_dummies(paths[key]).to_numpy(dtype=float).T
                             for key in ('origin_zone', 'destination_zone')]
    oz, dz = [paths[key].str.extract(r'^Z(\d+)$', expand=False).astype(float).to_numpy()
              for key in ('origin_zone', 'destination_zone')]
    span = dz - oz
    directions = matrix @ np.column_stack((prior, prior * (span - weights @ span)))
    contrast = null_space(directions.T).T
    assert contrast.shape[0] == len(rows) - 2
    views = {'raw': np.eye(len(rows)), 'prior_and_span': contrast}
    scores = {name: {} for name in views}
    for power in (0., .5, 1.):
        variance = np.maximum(weights, .1 / len(paths))**power
        variance /= variance.mean()
        for fraction in (0., .25, .5, .75, .9, .97, .99):
            diagonal = (matrix * ((1-fraction)*variance)) @ matrix.T
            scaled = matrix * np.sqrt(fraction*variance)
            origin, destination = scaled @ origins.T, scaled @ destinations.T
            for share in ((.5,) if fraction == 0. else (0., .5, 1.)):
                covariance = diagonal + share*(origin @ origin.T) + (1-share)*(destination @ destination.T)
                for name, view in views.items():
                    factor = cholesky(view @ covariance @ view.T, lower=True)
                    whitened = solve_triangular(factor, view @ residual, lower=True)
                    scale = np.mean(whitened**2)
                    assert scale > 0
                    score = 2*np.log(factor.diagonal()).mean() + np.log(scale)
                    scores[name][power,fraction,share] = float(score)
    image, _ = np.linalg.qr(incidence @ matrix.T, mode='reduced')
    orthogonal = counts - image @ (image.T @ counts)
    row_map = solve_triangular(cholesky(matrix @ matrix.T, lower=True), matrix @ incidence.T, lower=True)
    whitened_counts = solve_triangular(cholesky(matrix @ matrix.T, lower=True), counts[rows], lower=True)
    reconstructed = row_map.T @ whitened_counts
    diagnostics = dict(rank=len(rows), count_rows=len(counts),
        projection_relative_l2=np.linalg.norm(orthogonal)/np.linalg.norm(counts),
        independent_reconstruction_relative_l2=np.linalg.norm(reconstructed-counts)/np.linalg.norm(counts))
    return scores, diagnostics


def main():
    out = d.OUT/'task4_released_residual_boundary'
    d.save(out/'protocol.json',dict(source_sha256=d.sha(__file__), train='train', evaluation='validation',
        baseline=BASELINE, powers=[0,.5,1], shared_fractions=[0.,.25,.5,.75,.9,.97,.99], origin_shares=[0,.5,1],
        scope='Compare profiled covariance shape score on released count-minus-prior residuals. No hidden OD labels, random draws, or assumed residual provenance.'))
    rows, scores, hashes, compatibility = [], [], {}, []
    for panel in sorted(d.night.predictor.load_routes()):
        paths, incidence, counts, prior, h = base.load(panel,'train')
        vpaths, vincidence, vcounts, vprior, vh = base.load(panel,'validation')
        hashes.update(h)
        hashes.update(vh)
        assert paths.path_id.equals(vpaths.path_id)
        weights = prior/prior.sum()
        train, tcheck = score_split(paths, incidence, counts, prior, weights)
        valid, vcheck = score_split(paths, vincidence, vcounts, vprior, weights)
        compatibility.extend([dict(panel=panel,split='train',**tcheck), dict(panel=panel,split='validation',**vcheck)])
        for view in train:
            chosen = min(train[view],key=train[view].get)
            oracle = min(valid[view],key=valid[view].get)
            for model in train[view]:
                scores.append(dict(panel=panel,view=view,power=model[0],shared_fraction=model[1],
                    origin_share=model[2],train_score=train[view][model],validation_score=valid[view][model]))
            rows.append(dict(panel=panel,view=view,chosen_power=chosen[0],chosen_shared_fraction=chosen[1],
                chosen_origin_share=chosen[2],validation_oracle_origin_share=oracle[2],
                train_delta=train[view][chosen]-train[view][BASELINE],
                validation_delta=valid[view][chosen]-valid[view][BASELINE],
                validation_oracle_delta=valid[view][oracle]-valid[view][BASELINE]))
    result=pd.DataFrame(rows)
    pd.DataFrame(compatibility).to_csv(out/'count_compatibility.csv',index=False)
    result.to_csv(out/'by_panel.csv',index=False)
    pd.DataFrame(scores).to_csv(out/'candidate_scores.csv',index=False)
    result.groupby('view')[['train_delta','validation_delta','validation_oracle_delta']].mean().to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',[])


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
