"""Assess transfer of OD measurement choices from validation to private priors."""
import msvcrt

import numpy as np
import pandas as pd

import common as d
from audit_direct_od_measurement_design import base, basis, indicators


def main():
    out = d.OUT/'task4_direct_od_measurement_pooling'
    d.save(out/'protocol.json', dict(code_sha256=d.sha(__file__), split='train and validation → private',
        budget=16, design='Compare normalized train-only and validation-only weights with their fixed 50:50 average, scoring on private weights at identical private geometry. Private weights are an oracle comparator.',
        scope='Hypothetical direct measurements. Private weak prior is a public input, not supervision.'))
    rows, hashes, failures = [], {}, []
    for panel in d.night.predictor.load_routes():
        paths,A,_,prior,h = base.load(panel,'private')
        source,_,_,source_prior,hs = base.load(panel,'validation')
        train,_,_,train_prior,ht = base.load(panel,'train')
        assert paths.path_id.equals(train.path_id)
        hashes.update(ht)
        hashes.update(h)
        hashes.update(hs)
        assert paths.path_id.equals(source.path_id)
        matrices = [indicators(paths[k]).toarray() for k in ('origin_zone','destination_zone')]
        Q,_ = basis(np.vstack([A,*matrices]).T)
        initial = np.eye(len(paths))-Q@Q.T
        evaluation = prior / prior.sum()
        reference = evaluation @ initial.diagonal()
        selected = {}
        for label,w in [('train',train_prior),('validation',source_prior),('pooled',.5*(train_prior/train_prior.sum()+source_prior/source_prior.sum())),('private',prior)]:
            weights = w / w.sum()
            for rule in ('top_prior','greedy'):
                covariance = initial.copy()
                order = np.argsort(-weights,kind='stable')
                used = np.zeros(len(paths),dtype=bool)
                rank_gain = 0
                for step in range(16):
                    variance = covariance.diagonal().copy()
                    if rule == 'top_prior':
                        j = order[step]
                    else:
                        gain = np.divide(np.einsum('i,ij,ij->j',weights,covariance,covariance),variance,
                                         out=np.zeros_like(variance),where=variance>1e-12)
                        gain[used] = -1
                        j = int(np.argmax(gain))
                    used[j] = True
                    if variance[j] > 1e-12:
                        column = covariance[:,j].copy()
                        covariance -= np.outer(column,column) / variance[j]
                        rank_gain += 1
                selected[label,rule] = set(np.flatnonzero(used))
                rows.append(dict(panel=panel,weight_source=label,method=rule,rank_gain=rank_gain,
                                 remaining_private_trace_ratio=(evaluation @ covariance.diagonal())/reference))
        for row in rows[-8:]:
            method = row['method']
            row['shared_cells'] = len(selected[row['weight_source'],method] & selected['private',method])
        print(panel,flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'by_panel.csv',index=False)
    frame.groupby(['weight_source','method'])[['rank_gain','remaining_private_trace_ratio','shared_cells']].mean().to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',failures)
    d.state(status='stage_complete',stage='task4_direct_od_measurement_pooling_complete',
            next='Verify the transferred ordering and private-weighted diagnostics.')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
