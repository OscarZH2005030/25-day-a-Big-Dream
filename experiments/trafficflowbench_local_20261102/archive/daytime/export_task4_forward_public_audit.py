"""Export the low-strength forward reprojection for the release inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import task4_od_cycle_stress as stress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    base = stress.generator.family.base
    prediction = pd.read_csv(args.baseline, dtype={'path_id': str})
    assert len(prediction) == 70708
    records = []
    for (panel, departure), group in prediction.groupby(['panel', 'departure_time'], sort=False):
        split = {'SYN_PM_VALIDATION': 'validation', 'SYN_PM_PRIVATE': 'private'}[departure]
        paths, matrix, counts, prior, hashes = base.load(panel, split)
        historical = []
        # Validation has one earlier split. The forecast takes two source arrays.
        for source in ('train', 'train' if split == 'validation' else 'validation'):
            source_paths, _, _, source_prior, provenance = base.load(panel, source)
            assert paths.path_id.equals(source_paths.path_id)
            hashes.update(provenance)
            historical.append(source_prior)
        aligned = group.set_index('path_id').loc[paths.path_id]
        frozen = aligned.path_flow.to_numpy()
        features = stress.generator.family.structure.features(paths)['span_origin_destination']
        basis, _ = stress.exact.basis(features)
        seed = int.from_bytes(hashlib.sha256(f'{panel}/{split}/2026110643'.encode()).digest()[:8], 'little')
        rng = np.random.default_rng(seed)
        fold = np.empty(len(paths), int)
        fold[rng.permutation(len(paths))] = np.arange(len(paths)) % 5
        forecast, _, _ = stress.forecasts(features, basis, [*historical, prior], fold)
        geometry_mse = float(np.mean((forecast['geometry'] - prior)**2))
        pair_mse = float(np.mean((forecast['pair'] - prior)**2))
        direct, status = base.ridge(matrix, counts, .95 * prior + .05 * forecast['pair'])
        assert status['success'], (panel, split, status)
        origin = paths.origin_zone.str.extract(r'^Z([0-9]+)$', expand=False).astype(int).to_numpy()
        dest = paths.destination_zone.str.extract(r'^Z([0-9]+)$', expand=False).astype(int).to_numpy()
        lookup = stress.cycle.path_lookup(origin, dest)
        rectangles = stress.cycle.draw_cycles(lookup, 4 * len(paths), rng)
        first, _ = stress.cycle.refine_cycles(frozen, direct, prior, rectangles)
        reproject_rng = np.random.default_rng(seed ^ 2026110641)
        rectangles = stress.cycle.draw_cycles(lookup, 4 * len(paths), reproject_rng)
        result, _ = stress.cycle.refine_cycles(frozen, first, prior, rectangles)
        diff = result - frozen
        assert np.isfinite(result).all() and result.min() >= -1e-10
        assert np.abs(matrix @ diff).max() < 1e-7
        assert all(np.abs(np.bincount(zones, weights=diff)).max() < 1e-7 for zones in (origin, dest))
        assert abs(np.abs(result-prior).sum()-np.abs(frozen-prior).sum()) < 1e-7
        by_path = pd.Series(np.maximum(result, 0.), index=paths.path_id)
        prediction.loc[group.index, 'path_flow'] = group.path_id.map(by_path)
        records.append(dict(panel=panel, split=split, seed=seed, moved_l1=float(np.abs(diff).sum()),
                            max_count_change=float(np.abs(matrix @ diff).max()), geometry_prior_mse=geometry_mse, pair_prior_mse=pair_mse, prior_mse_reduction=1.-pair_mse/max(geometry_mse,1e-12), inputs=hashes))
    assert len(records) == 20 and any(row['moved_l1'] > 0. for row in records)
    prediction.to_csv(args.output, index=False)
    args.output.with_suffix('.json').write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
