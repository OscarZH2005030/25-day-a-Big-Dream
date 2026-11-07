"""Compare balanced designs on the same unmeasured OD cells."""
import argparse
import hashlib
import msvcrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import pinvh

import common as d
from audit_direct_od_measurement_design import base

EXPONENTS=(0.,.5,1.,.25,.75)
HOLDOUT_SEED = 2026110707


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--span-correlation', action='store_true',
                        help='Assign half the shared variance to exact trip-span groups')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--span-shared-weight', type=float, default=.125)
    parser.add_argument('--fixed-holdout', action='store_true',
                        help='Reserve a deterministic quarter of OD cells from all probe selections')
    args = parser.parse_args(argv)
    if not 0 <= args.span_shared_weight <= .25:
        parser.error('span shared weight must be between 0 and 0.25')
    if args.span_shared_weight != .125 and (not args.span_correlation or args.output_dir is None):
        parser.error('custom span weight requires --span-correlation and a separate --output-dir')
    if args.fixed_holdout and args.output_dir is None:
        parser.error('fixed holdout requires a separate --output-dir')
    stage = 'task4_direct_od_span_covariance' if args.span_correlation else 'task4_direct_od_marginal_measurement_gain'
    out = args.output_dir or d.OUT / stage
    span_weight = args.span_shared_weight if args.span_correlation else 0.
    margin_weight = (.25 - span_weight) / 2
    shared_weights = {'origin': margin_weight, 'destination': margin_weight, 'span': span_weight}
    d.save(out/'protocol.json',dict(code_sha256=d.sha(__file__), split='private', budget=16,
        exponents=EXPONENTS, shared_weights=shared_weights, design='Add the specified shared-group covariance before conditioning on counts, holding marginal variances fixed. Exact span is destination zone index minus origin zone index. Select by maximizing the worst normalized marginal gain over both covariance models and three design exponents. Freeze all six sets across three selection models and evaluate under both joint and diagonal covariance on cells measured by none of the six policies. Also score unweighted variance reduction of the origin and destination totals using the same fixed measurements.',
        fixed_holdout=dict(seed=HOLDOUT_SEED, fraction=.25,
                           rule='SHA256 of seed, road and path ID, first byte below 64') if args.fixed_holdout else None,
        scope='Hypothetical direct measurements. Variance laws are sensitivity assumptions, not fitted to hidden OD errors.'))
    rows,hashes=[],{}
    holdout_provenance = {}
    for panel in d.night.predictor.load_routes():
        paths,A,_,private,h=base.load(panel,'private')
        t,_,_,train,ht=base.load(panel,'train')
        v,_,_,valid,hv=base.load(panel,'validation')
        assert paths.path_id.equals(t.path_id) and paths.path_id.equals(v.path_id)
        for provenance in (h,ht,hv):
            hashes.update(provenance)
        w=.5*(train/train.sum()+valid/valid.sum())
        evaluation=private/private.sum()
        n=len(paths)
        heldout = np.zeros(n, dtype=bool)
        if args.fixed_holdout:
            heldout = np.array([hashlib.sha256(f'{HOLDOUT_SEED}:{panel}:{path}'.encode()).digest()[0] < 64
                                for path in paths.path_id])
            assert heldout.any() and (~heldout).sum() >= 16
            holdout_provenance[panel] = {'cells': int(heldout.sum()),
                                         'mask_sha256': hashlib.sha256(heldout.tobytes()).hexdigest()}
        origin,_=pd.factorize(paths.origin_zone,sort=True)
        same_origin=origin[:,None]==origin[None,:]
        destination,_=pd.factorize(paths.destination_zone,sort=True)
        same_destination=destination[:,None]==destination[None,:]
        shared = shared_weights['origin'] * same_origin + shared_weights['destination'] * same_destination
        if args.span_correlation:
            origin_index, destination_index = [paths[key].str.extract(r'^Z([0-9]+)$', expand=False).astype(int).to_numpy()
                                                for key in ('origin_zone', 'destination_zone')]
            span = destination_index - origin_index
            shared += shared_weights['span'] * (span[:, None] == span[None, :])
        starts=[]
        diagonal_starts=[]
        for power in EXPONENTS:
            variance=np.maximum(w,.1/n)**power
            variance/=variance.mean()
            sd=np.sqrt(variance)
            prior_cov=.75*np.diag(variance)+shared*sd[:,None]*sd[None,:]
            np.testing.assert_allclose(prior_cov.diagonal(),variance,rtol=0,atol=1e-12)
            cross=A@prior_cov
            covariance=prior_cov-cross.T@pinvh(cross@A.T)@cross
            np.testing.assert_allclose(A@covariance,0,atol=1e-8,rtol=0)
            starts.append(covariance)
            diagonal_cross=A*variance
            diagonal_starts.append(np.diag(variance)-diagonal_cross.T@pinvh(diagonal_cross@A.T)@diagonal_cross)
        outcomes={}
        selection_models={'joint':starts[:3], 'diagonal':diagonal_starts[:3], 'balanced':starts[:3]+diagonal_starts[:3]}
        objectives = ('total', 'other', 'heldout') if args.fixed_holdout else ('total', 'other')
        for selection, name in [(key, objective) for key in selection_models for objective in objectives]:
            matrices=[c.copy() for c in selection_models[selection]]
            used=np.zeros(n,dtype=bool)
            objective_weights = w * heldout if name == 'heldout' else w
            for _ in range(16):
                diagonals=[c.diagonal().copy() for c in matrices]
                gains=np.array([np.divide(np.einsum('i,ij,ij->j',objective_weights,c,c),var,
                                          out=np.zeros(n),where=var>1e-12)
                                for c,var in zip(matrices,diagonals)])
                eligible=(~used)&(~heldout)&np.all(np.array(diagonals)>1e-12,axis=0)
                if name=="other":
                    gains-=w*np.array(diagonals)
                gains=np.maximum(gains,0.)
                gains[:,~eligible]=0
                assert eligible.any()
                objective=(gains/np.maximum(gains.max(axis=1,keepdims=True),1e-30)).min(axis=0)
                objective[~eligible]=-np.inf
                j=int(np.argmax(objective))
                used[j]=True
                for c,var in zip(matrices,diagonals):
                    column=c[:,j].copy()
                    if var[j]>1e-12:
                        c-=np.outer(column,column)/var[j]
            outcomes[selection,name]=used
        origin_rows=pd.get_dummies(paths.origin_zone).to_numpy(dtype=float).T
        destination_rows=pd.get_dummies(paths.destination_zone).to_numpy(dtype=float).T
        common=heldout if args.fixed_holdout else ~np.logical_or.reduce(list(outcomes.values()))
        for (selection,name),used in outcomes.items():
            for law,covariances in [('joint',starts),('diagonal',diagonal_starts)]:
                for power,start in zip(EXPONENTS,covariances):
                    cross=start[:,used]
                    block=start[np.ix_(used,used)]
                    inverse=pinvh(block)
                    reduction=evaluation*np.einsum('ij,ij->i',cross@inverse,cross)
                    marginal_gains={}
                    for label,groups in [('origin',origin_rows),('destination',destination_rows)]:
                        projected=groups@cross
                        baseline=np.einsum('ij,ij->',groups@start,groups)
                        marginal_gains[label+'_reduction']=np.einsum('ij,ij->',projected@inverse,projected)/baseline
                    scale=evaluation@start.diagonal()
                    row = dict(panel=panel, design=name, selection=selection, exponent=power, law=law,
                               total_reduction=reduction.sum()/scale,
                               common_reduction=reduction[common].sum()/scale, **marginal_gains)
                    if args.fixed_holdout:
                        row['heldout_relative_reduction'] = reduction[heldout].sum() / (
                            evaluation[heldout] @ start.diagonal()[heldout])
                    rows.append(row)
        print(panel, flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'by_panel.csv',index=False)
    metrics = ['total_reduction', 'common_reduction', 'origin_reduction', 'destination_reduction']
    if args.fixed_holdout:
        metrics.append('heldout_relative_reduction')
        d.save(out/'holdout_sha256.json', holdout_provenance)
    frame.groupby(['design','selection','exponent','law'])[metrics].mean().to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',[])
    d.state(status='stage_complete',stage=out.name+'_complete',
            next='Compare gains with total-target selections using matched roads and variance laws.')


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
