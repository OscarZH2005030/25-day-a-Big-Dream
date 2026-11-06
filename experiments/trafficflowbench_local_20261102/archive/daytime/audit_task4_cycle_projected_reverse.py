"""Compare deterministic OD flow blends with randomized endpoint choice."""
import hashlib
import msvcrt

import numpy as np
import pandas as pd

import common as d
import od_cycle_refinement as cycle
import task4_span_shrink_stress as structure


def main(source=None):
    source = source or d.OUT / 'task4_od_cycle_stress' / 'all_cases.csv'
    out = d.OUT / 'task4_cycle_projected_reverse'
    rows = []
    hashes = {}
    networks = {}
    network_hashes = {}
    for case, group in pd.read_csv(source).groupby('case'):
        meta = group.set_index('method')
        folder = d.WORK / 'task4_od_cycle_cases' / hashlib.sha256(case.encode()).hexdigest()[:16]
        input_path, pred_path = folder / 'inputs.npz', folder / 'predictions.npz'
        hashes[case] = [d.sha(input_path), d.sha(pred_path)]
        assert hashes[case] == meta.loc['baseline', ['input_artifact_sha256', 'prediction_artifact_sha256']].tolist()
        with np.load(input_path) as data, np.load(pred_path) as pred:
            truth = data['artificial_truth'][2]
            total = truth.sum()
            baseline = pred['baseline']
            def score(flow):
                return max(0., 1. - np.abs(flow - truth).sum() / total)
            base = score(baseline)
            panel = meta.loc['baseline', 'panel']
            if panel not in networks:
                paths, A, _, _, provenance = structure.load(panel, 'train')
                network_hashes.update(provenance)
                origin = paths.origin_zone.str.extract(r'^Z([0-9]+)$', expand=False).astype(int).to_numpy()
                dest = paths.destination_zone.str.extract(r'^Z([0-9]+)$', expand=False).astype(int).to_numpy()
                networks[panel] = A, origin, dest, cycle.path_lookup(origin, dest)
            A, origin, dest, lookup = networks[panel]
            rng = np.random.default_rng(int(data['seed']) ^ 2026110641)
            rectangles = cycle.draw_cycles(lookup, 4 * len(baseline), rng)
            assert np.max(np.abs(A[:,rectangles[:,0]]-A[:,rectangles[:,1]]-A[:,rectangles[:,2]]+A[:,rectangles[:,3]])) == 0
            prior = data['artificial_prior'][2]
            for method in ('cycle005', 'cycle025'):
                candidate = pred[method]
                for sign in (-1, 1):
                    target = baseline + sign * (candidate - baseline)
                    refined, metrics = cycle.refine_cycles(baseline, target, prior, rectangles)
                    diff = refined - baseline
                    count_drift = np.max(np.abs(A @ diff))
                    marginal_drift = max(np.abs(np.bincount(zones, weights=diff)).max() for zones in (origin, dest))
                    prior_drift = abs(np.abs(refined-prior).sum()-np.abs(baseline-prior).sum())
                    assert max(count_drift, marginal_drift, prior_drift) < 1e-7
                    record = meta.loc[method]
                    rows.append(dict(panel=panel, noise=record.noise, drift=record.drift,
                                     method=method, sign=sign, cap=metrics['relative_L1_movement'],
                                     gain=score(refined)-base))
    frame = pd.DataFrame(rows)
    assert len(frame) == 1920
    out.mkdir(parents=True, exist_ok=True)
    frame.groupby(['panel', 'noise', 'drift', 'method', 'sign']).agg(
        cases=('gain', 'size'), mean_gain=('gain', 'mean'),
        min_gain=('gain', 'min'), mean_cap=('cap', 'mean'),
        no_movement=('cap', lambda values: (values <= 1e-12).sum())
    ).to_csv(out / 'by_joint_stratum.csv')
    d.save(out / 'protocol.json', dict(source_sha256=d.sha(__file__), case_table_sha256=d.sha(source),
                                      input_prediction_hashes=hashes, network_hashes=network_hashes, cycle_source_sha256=d.sha(cycle.__file__), seed_xor=2026110641,
                                      scope='Exposed synthetic cases; fixed weights and fixed endpoint predictions.'))


if __name__ == '__main__':
    with (d.ROOT / 'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        main()
