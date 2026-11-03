"""Read back all saved synthetic cases and verify paired result denominators."""
import common as d
import task4_smallnet_posterior_stress as study
import numpy as np
import pandas as pd
import hashlib

O = d.OUT / 'task4_smallnet_posterior_stress_retry'
df = pd.read_csv(O / 'all_cases.csv')
assert len(df) == 96 * 7 and df.case.nunique() == 96
errors = []
for case, group in df.groupby('case'):
    artifact = d.WORK / 'task4_smallnet_posterior_cases' / (hashlib.sha256(case.encode()).hexdigest()[:16] + '.npz')
    assert set(group.artifact_sha256) == {d.sha(artifact)}
    z = np.load(artifact)
    Q, _, _, _, di = study.geometry(int(group.nodes.iloc[0]))
    truth, b, c, samples = (z[k] for k in ['artificial_truth', 'artificial_prior', 'artificial_counts', 'posterior_samples'])
    assert np.max(np.abs(Q @ truth - c)) < 1e-9
    assert samples.min() >= 0
    assert np.max(np.abs(samples.reshape(-1, len(b)) @ Q.T - c)) < 1e-8
    eligible = bool(group.posterior_converged.iloc[0])
    actions = study.actions(Q, c, b, samples) if eligible else {'weak_prior': (b, True), 'baseline': study.base.ridge(Q, c, b)}
    for _, row in group.iterrows():
        if not row.action_available:
            continue
        prediction = actions[row.method][0]
        metrics = study.base.metrics(prediction, truth, b, Q, c, di)
        errors.extend(abs(float(row[k]) - value) for k, value in metrics.items())
assert max(errors) < 1e-10
baseline = df[df.method == 'baseline'][['case', 'synthetic_S_ODME', 'synthetic_S_od']]
paired = df.merge(baseline, on='case', suffixes=('', '_baseline'))
paired = paired[paired.posterior_converged & paired.action_available].copy()
paired['score_delta'] = paired.synthetic_S_ODME - paired.synthetic_S_ODME_baseline
paired['OD_delta'] = paired.synthetic_S_od - paired.synthetic_S_od_baseline
paired.groupby(['scenario', 'true_noise_factor', 'method']).agg(paired_cases=('case', 'count'), mean_score_delta=('score_delta', 'mean'), worst_score_delta=('score_delta', 'min'), mean_OD_delta=('OD_delta', 'mean')).to_csv(O / 'paired_by_assumption.csv')
d.save(O / 'independent_readback_verification.json', dict(cases=96, converged_cases=int(df.groupby('case').posterior_converged.first().sum()), metric_max_abs_error=max(errors), all_artifact_hashes_checked=True, all_draws_nonnegative_and_count_conserving=True, verifier_sha256=d.sha(__file__), scope='Synthetic known-latent results only; no hidden OD validation.'))
print('Verified 96 synthetic artifacts, all available action scores and paired denominators:', max(errors))
