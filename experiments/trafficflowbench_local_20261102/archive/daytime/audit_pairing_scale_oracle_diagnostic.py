"""Measure repeatability of the unobserved pairing component of the released priors."""
import msvcrt

import numpy as np
import pandas as pd
from scipy.linalg import orth

import common as d
from audit_direct_od_measurement_design import base





def main():
    out = d.OUT/'task4_pairing_scale_oracle_diagnostic'
    d.save(out/'protocol.json',dict(source_sha256=d.sha(__file__),predictor_splits='two non-target released splits',geometry_split='private',
        evaluation_splits=['train','validation','private'],scope='Diagnostic target-oracle multipliers for L1 and squared error in each orthogonal subspace.'))
    rows, hashes, failures=[], {}, []
    for panel in sorted(d.night.predictor.load_routes()):
        split_data = {}
        for split in ('train', 'validation', 'private'):
            paths, counts, _, prior, provenance = base.load(panel, split)
            hashes.update(provenance)
            split_data[split] = paths, counts, prior / prior.sum()
        paths, counts, _ = split_data['private']
        assert all(paths.path_id.equals(item[0].path_id) for item in split_data.values())
        for split, (_, _, target) in split_data.items():
            train, valid = [item[2] for name, item in split_data.items() if name != split]
            groups=np.vstack([pd.get_dummies(paths[k]).to_numpy(dtype=float).T
                              for k in ('origin_zone','destination_zone')])
            origin,destination=[paths[k].str.extract(r'^Z([0-9]+)$',expand=False).astype(int).to_numpy()
                                for k in ('origin_zone','destination_zone')]
            span=destination-origin
            bins=(span[:,None]==np.unique(span)).astype(float).T
            for view, constraints in [('margins', np.vstack((counts, groups)))]:
                basis = orth(constraints.T, rcond=1e-10)
                pairs = np.column_stack((train, valid, target))
                source, second, truth = (pairs - basis @ (basis.T @ pairs)).T
                target_l1 = np.abs(truth).sum()
                target_l2 = truth @ truth
                if min(target_l1, target_l2) <= 1e-15:
                    failures.append(dict(panel=panel, split=split, view=view, error='Zero residual norm'))
                    continue
                average = .25 * (source + second)
                histogram_basis = orth(np.vstack((constraints, bins)).T, rcond=1e-10)
                free = average - histogram_basis @ (histogram_basis.T @ average)
                fixed = average - free
                for component, direction in [('whole', average), ('histogram', fixed), ('free', free)]:
                    nonzero = np.abs(direction) > 1e-15
                    energy = direction @ direction
                    if energy <= 1e-30:
                        failures.append(dict(panel=panel, split=split, component=component, error='Zero direction'))
                        continue
                    remainder = average - direction
                    residual = truth - remainder
                    ratios = residual[nonzero] / direction[nonzero]
                    weights = np.abs(direction[nonzero])
                    order = np.argsort(ratios, kind='stable')
                    index = np.searchsorted(np.cumsum(weights[order]), .5 * weights.sum())
                    l1_scale = max(0., ratios[order[index]])
                    l2_scale = max(0., (residual @ direction) / energy)
                    rows.append(dict(panel=panel, split=split, component=component,
                                     l1_scale=float(l1_scale), l2_scale=float(l2_scale)))
    data=pd.DataFrame(rows)
    data.to_csv(out/'by_panel.csv',index=False)
    data.groupby(['split','component'])[['l1_scale','l2_scale']].agg(['min','median','max']).to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',failures)


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
