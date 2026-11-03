"""Compare aligned and opposed errors using the same feasible OD exchange."""
import hashlib
import msvcrt

import numpy as np
import pandas as pd

import common as d
from audit_source_agreement_common_corruption import directions, capped_solution, inputs, basis, structure, score


def main():
    out = d.OUT / 'task4_source_error_correlation'
    source = d.OUT / 'task4_source_agreement_common_corruption/by_case.csv'
    table = pd.read_csv(source, dtype={'seed': 'int64'}).query("version == 'clean' and rule == 'mean_source'")
    d.save(out / 'protocol.json', dict(
        seed=2026110351, source_sha256=d.sha(source), code_sha256=d.sha(__file__),
        hypothesis='Positive source error alignment can change OD reliability even at equal perturbation amplitude.',
        design='Use the same seeded exchange in both scenarios. Cap it for feasibility under both signs in both source priors. Compare (+,+) and (+,-) at equal realized target movement for both rules and the clean controls.',
        scope='Reused synthetic development cases. No hidden OD evaluation.'))
    rows, failures, hashes = [], [], {}
    for panel, group in table.groupby('panel'):
        paths, A, _, _, provenance = inputs.load(panel, 'train')
        hashes.update(provenance)
        origin, dest = [paths[k].str[1:].astype(int).to_numpy() for k in ['origin_zone', 'destination_zone']]
        Q, _ = basis(structure.features(paths)['span_origin_destination'])
        for row in group.itertuples():
            name = hashlib.sha256(row.case.encode()).hexdigest()[:16] + '.npz'
            exchange = d.WORK / 'task4_source_agreement_common_corruption_cases' / name
            assert d.sha(exchange) == row.artifact_sha256
            with np.load(exchange) as z, np.load(d.WORK / 'task4_source_agreement_stress_cases' / name) as ref:
                priors, base, truth, count = [ref[k] for k in ('priors', 'baseline', 'truth', 'counts')]
                delta = z['source_change']
                nonzero = np.abs(delta) > 1e-12
                headroom = np.min(priors[:2], axis=0)
                factor = min(1., float(np.min(headroom[nonzero] / np.abs(delta[nonzero])))) if nonzero.any() else 1.
                delta = factor * delta
                assert np.max(np.abs(A @ delta)) < 1e-7
                for idx in (origin, dest):
                    assert np.max(np.abs(np.bincount(idx, weights=delta))) < 1e-7
                alternatives = {(method, 'clean'): ref[method] for method in ('mean_source', 'agreement_source')}
                for scenario, sign in [('aligned', 1.), ('opposed', -1.)]:
                    changed = priors.copy()
                    changed[:2] += np.array([1., sign])[:, None] * delta
                    assert changed.min() >= -1e-8
                    for method, column in directions(changed, Q).items():
                        result = capped_solution(base, priors[2], origin, dest, column)
                        if result.success:
                            alternatives[method, scenario] = result.x[:len(base)]
                        else:
                            failures.append(dict(case=row.case, method=method, scenario=scenario, error=result.message))
                if len(alternatives) != 6:
                    continue
                sizes = {k: np.abs(f-base).sum() for k, f in alternatives.items()}
                budget = min(sizes.values())
                scores = {}
                flows = {}
                b = priors[2]
                base_score = score(base, truth, b, A, count, dest)
                for key, f in alternatives.items():
                    flow = base + (f-base) * budget / max(sizes[key], 1e-12)
                    od, total = score(flow, truth, b, A, count, dest)
                    np.testing.assert_allclose(total-base_score[1], .45*(od-base_score[0]), atol=1e-10, rtol=0)
                    scores[key] = od
                    flows["_".join(key)] = flow
                folder = d.WORK / "task4_source_error_correlation_cases"
                folder.mkdir(parents=True, exist_ok=True)
                artifact = folder / name
                np.savez_compressed(artifact, seed=row.seed, source_change=delta, **flows)
                digest = d.sha(artifact)
                for method in ('mean_source', 'agreement_source'):
                    clean, aligned, opposed = [scores[method, s] for s in ('clean', 'aligned', 'opposed')]
                    rows.append(dict(case=row.case, panel=panel, noise=row.noise, seed=row.seed, rule=method,
                                     source_movement=np.abs(delta).sum()/priors[:2].sum(axis=1).min(),
                                     target_movement=budget/base.sum(), aligned_effect=aligned-clean,
                                     opposed_effect=opposed-clean, aligned_minus_opposed=aligned-opposed, artifact_sha256=digest))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'by_case.csv', index=False)
    frame.groupby(['noise', 'rule']).agg(
        cases=('case', 'size'), source_movement=('source_movement', 'mean'),
        aligned=('aligned_effect', 'mean'), opposed=('opposed_effect', 'mean'),
        alignment_contrast=('aligned_minus_opposed', 'mean')).to_csv(out / 'summary.csv')
    d.save(out / 'failures.json', failures)
    d.save(out / 'input_sha256.json', hashes)
    d.state(status='stage_complete', stage='task4_source_error_correlation_complete',
            next='Check effective perturbation amplitude and validate aligned versus opposed effects.')


if __name__ == '__main__':
    with (d.ROOT / 'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        main()
