"""Test which hypothetical trip-span summaries constrain OD pairing."""
import hashlib
import msvcrt
from datetime import datetime, timezone

import common as d
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from task4_exact_pair_audit import base, basis

SEED = 2026110341


def indicators(values):
    indices = pd.factorize(values, sort=True)[0]
    return sparse.csr_matrix((np.ones(len(values)),(indices,np.arange(len(values)))),
                             shape=(indices.max()+1,len(values)))


def main():
    out = d.OUT/'task4_extra_measurement_audit'
    prior_audit = d.OUT/'task4_metric_invariant/witnesses.csv'
    d.save(out/'protocol.json',dict(
        registered_utc=datetime.now(timezone.utc).isoformat(),seed=SEED,groups=20,
        hypothesis='Global and marginal trip-span summaries constrain some OD pair directions, while leaving count-compatible ambiguity.',
        measurements='Zone-index span, not measured distance. Compare existing count and margin rows with total span, global span histogram, and histogram plus origin/destination span first moments.',
        witness='One fixed random linear objective per panel and split, shared by both new measurement sets. Nonnegative flows in baseline prior orthant, fixed counts, margins and original-prior L1, plus hypothetical summaries fixed at baseline.',
        scope='Measurement design and real-network feasible counterexamples. No extra measurements are available or imposed on an official candidate.',
        source_sha256=d.sha(__file__), previous_witness_table_sha256=d.sha(prior_audit)))
    original = pd.read_csv(prior_audit)
    rows, ranks, hashes, failures = [], [], {}, []
    for panel in d.night.predictor.load_routes():
        for split in ['validation','private']:
            paths,A,_,prior,inputs = base.load(panel,split)
            hashes.update(inputs)
            origin,dest = [paths[key].str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy()
                           for key in ['origin_zone','destination_zone']]
            span = dest-origin
            assert (span>0).all()
            O,D,H = map(indicators,[origin,dest,span])
            moments = sparse.vstack([O.multiply(span),D.multiply(span)],format='csr')
            existing = sparse.vstack([sparse.csr_matrix(A),O,D],format='csr')
            configurations = {
                'counts_margins': existing,
                'plus_total_span': sparse.vstack([existing,sparse.csr_matrix(span[None,:])],format='csr'),
                'plus_histogram': sparse.vstack([existing,H],format='csr'),
                'plus_histogram_and_moments': sparse.vstack([existing,H,moments],format='csr'),
            }
            for method,matrix in configurations.items():
                _,rank=basis(matrix.T.toarray())
                ranks.append(dict(panel=panel,split=split,method=method,paths=len(paths),rank=rank,
                                  linear_nullity=len(paths)-rank))
            assert ranks[-3]['rank']==ranks[-4]['rank']
            old = d.WORK/'task4_metric_invariant'/panel/split/'linear_objective_0_NOT_A_SUBMISSION.parquet'
            expected=original[(original.panel==panel)&(original.split==split)&(original.replicate==0)].iloc[0]
            assert d.sha(old)==expected.witness_sha256
            flow=pd.read_parquet(old).set_index('path_id').loc[paths.path_id,'baseline_flow'].to_numpy()
            sign=np.sign(flow-prior)
            bounds=[(float(b),None) if s>0 else (0.,float(b)) if s<0 else (float(b),float(b))
                    for b,s in zip(prior,sign)]
            seed=int.from_bytes(hashlib.sha256(f'{SEED}/{panel}/{split}'.encode()).digest()[:8],'little')%(2**63)
            objective=np.random.default_rng(seed).normal(size=len(flow))
            for method in ['plus_histogram','plus_histogram_and_moments']:
                matrix=sparse.vstack([configurations[method],sparse.csr_matrix(sign[None,:])],format='csr')
                target=matrix@flow
                result=linprog(objective,A_eq=matrix,b_eq=target,bounds=bounds,method='highs')
                if not result.success:
                    failures.append(dict(panel=panel,split=split,method=method,seed=seed,message=result.message))
                    continue
                delta=result.x-flow
                error=float(np.max(np.abs(matrix@delta))/max(1.,np.max(np.abs(target))))
                prior_error=float(abs(np.abs(result.x-prior).sum()-np.abs(flow-prior).sum())/max(1.,np.abs(flow-prior).sum()))
                assert error<1e-9 and prior_error<1e-9 and result.x.min()>=-1e-9
                folder=d.WORK/'task4_extra_measurement_witnesses'/panel/split
                folder.mkdir(parents=True,exist_ok=True)
                file=folder/f'{method}.npz'
                np.savez_compressed(file,baseline=flow,witness=result.x,seed=seed,objective=objective)
                rows.append(dict(panel=panel,split=split,method=method,seed=seed,
                                 relative_od_movement=float(np.abs(delta).sum()/flow.sum()),
                                 relative_constraint_error=error,relative_prior_error=prior_error,
                                 old_witness_sha256=d.sha(old),artifact_sha256=d.sha(file)))
        pd.DataFrame(rows).to_csv(out/'witness_summary.csv',index=False)
        pd.DataFrame(ranks).to_csv(out/'ranks.csv',index=False)
        d.save(out/'failures.json',failures)
        d.state(status='running',stage='extra_measurement_audit',panel=panel)
    assert len(rows)+len(failures)==40
    frame=pd.DataFrame(rows)
    frame.groupby('method').relative_od_movement.agg(['count','min','median','max']).to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.state(status='stage_complete',stage='extra_measurement_audit_complete',
            next='Independently verify rank increments, constraints and portable solver reproduction. Hypothetical summaries cannot enter deployed fitting.')


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
