"""Enumerate shared panel sign reversals for the rule-by-amplitude interaction."""
import itertools

import numpy as np
import pandas as pd

import common as d


def main():
    source = d.OUT / 'task4_source_error_amplitude_panel_check/by_panel.csv'
    out = d.OUT / 'task4_source_error_rule_interaction'
    d.save(out/'protocol.json', dict(input_sha256=d.sha(source), code_sha256=d.sha(__file__),
        scope='Descriptive sign-reversal reference on ten reused synthetic road panels.',
        design='Subtract mean-rule amplitude contrast from agreement-rule amplitude contrast within panel and noise. Enumerate all 1024 common panel sign assignments. Compare absolute studentized values with their joint maximum across four noise families.',
        assumptions='A probability interpretation would require joint sign symmetry across independent panel blocks, which is not established here.'))
    frame = pd.read_csv(source)
    wide = frame.pivot(index=['panel','noise'], columns='rule', values='delta')
    panel = (wide.agreement_source-wide.mean_source).unstack('noise')
    assert panel.shape == (10,4) and not panel.isna().any().any()
    values = panel.to_numpy()
    sign = np.array(list(itertools.product([-1.,1.],repeat=len(panel))))
    samples = sign[:,:,None]*values
    statistics = np.abs(samples.mean(axis=1))/(samples.std(axis=1,ddof=1)/np.sqrt(len(panel)))
    observed = statistics[-1]
    tail = statistics >= observed-1e-12
    maximum = statistics.max(axis=1)
    summary = pd.DataFrame(dict(mean=values.mean(axis=0), positive_panels=(values>0).sum(axis=0),
        statistic=observed, raw_reference_fraction=tail.mean(axis=0),
        joint_reference_fraction=(maximum[:,None]>=observed-1e-12).mean(axis=0)),index=panel.columns)
    panel.to_csv(out/'by_panel.csv')
    summary.to_csv(out/'summary.csv')
    d.save(out/'failures.json', [])
    d.state(status='stage_complete', stage='task4_source_error_rule_interaction_complete',
            next='Verify the finite sign reference and evaluate whether to move beyond reused artificial targets.')
    print(summary.to_string())


if __name__=='__main__':
    main()
