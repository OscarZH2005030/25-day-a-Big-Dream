"""Decompose frozen Task4 corrections into net and invisible components."""
import msvcrt
from datetime import datetime, timezone

import common as d
import numpy as np
import pandas as pd
from task4_span_shrink_stress import load


def zone_net(flow,origin,dest,n):
    return np.bincount(origin,weights=flow,minlength=n)-np.bincount(dest,weights=flow,minlength=n)


def main():
    out=d.OUT/'task4_net_repair'
    anchor=d.ROOT/'work/review-repo/experiments/task4/robust_odme_20261027/outputs/task4_path_flow.csv'
    d.save(out/'protocol.json',dict(
        registered_utc=datetime.now(timezone.utc).isoformat(),seed=None,baseline_sha256=d.sha(anchor),
        scope='Released counts and frozen predictions only. Net recovery is conditional on the static exact-count interpretation. Signed corrections are diagnostics, not candidate flows.',
        decomposition='Recover net targets from ordered cut differences. The minimum Euclidean-norm path repair of net imbalance r is (r_origin-r_destination)/n on the complete triangular support. Project the frozen baseline change similarly, leaving a count-null residual.',
        source_sha256=d.sha(__file__)))
    baseline=pd.read_csv(anchor,dtype={'panel':str,'path_id':str,'departure_time':str})
    rows,hashes,failures=[],{},[]
    for panel in d.night.predictor.load_routes():
        for split in ['validation','private']:
            paths,A,c,prior,inputs=load(panel,split)
            hashes.update(inputs)
            origin,dest=[paths[k].str[1:].astype(int).to_numpy() for k in ['origin_zone','destination_zone']]
            origin,dest=origin-origin.min(),dest-origin.min()
            n=int(dest.max()+1)
            assert len(paths)==n*(n-1)//2 and len(set(zip(origin,dest)))==len(paths)
            cuts=np.array([(origin<=k)&(k<dest) for k in range(n-1)],dtype=np.int8)
            cut_lookup={np.packbits(row).tobytes():i for i,row in enumerate(cuts)}
            mapped=np.array([cut_lookup.get(np.packbits(row.astype(np.int8)).tobytes(),-1) for row in A])
            if (mapped<0).any() or len(set(mapped))!=n-1:
                failures.append(dict(panel=panel,split=split,reason='Incomplete exact cut mapping'))
                continue
            cut_totals=np.array([np.mean(c[mapped==k]) for k in range(n-1)])
            disagreement=float(np.max(np.abs(cut_totals[mapped]-c)))
            if disagreement>1e-9*max(1.,np.max(np.abs(c))):
                failures.append(dict(panel=panel,split=split,reason='Inconsistent duplicate cut values',max_disagreement=disagreement))
                continue
            reference_net=np.diff(np.r_[0.,cut_totals,0.])
            error=reference_net-zone_net(prior,origin,dest,n)
            minnorm=(error[origin]-error[dest])/n
            np.testing.assert_allclose(A@(prior+minnorm),c,rtol=1e-10,atol=1e-7)
            priorfile=d.DATA/'task4'/panel/split/'synthetic_weak_prior.csv'
            departure=pd.read_csv(priorfile,usecols=['departure_time'],dtype=str).departure_time.iloc[0]
            flow=baseline[(baseline.panel==panel)&(baseline.departure_time==departure)].set_index('path_id').loc[paths.path_id,'path_flow'].to_numpy()
            change=flow-prior
            net_change=zone_net(change,origin,dest,n)
            visible=(net_change[origin]-net_change[dest])/n
            invisible=change-visible
            scale=max(1.,np.linalg.norm(change))
            assert np.linalg.norm(A@invisible)<1e-9*scale
            assert abs(visible@invisible)<1e-9*scale**2
            folder=d.WORK/'task4_net_repair_cases'/panel
            folder.mkdir(parents=True,exist_ok=True)
            file=folder/f'{split}.npz'
            np.savez_compressed(file,reference_net=reference_net,minnorm=minnorm,visible=visible,invisible=invisible)
            rows.append(dict(panel=panel,split=split,cut_groups=n-1,duplicate_count_rows=len(c)-(n-1),
                             duplicate_disagreement=disagreement,
                             signed_repair_negative_fraction=float(np.mean(prior+minnorm<0)),
                             frozen_change_invisible_energy_fraction=float((invisible@invisible)/max(1e-30,change@change)),
                             max_count_null_residual=float(np.max(np.abs(A@invisible))),
                             artifact_sha256=d.sha(file)))
    pd.DataFrame(rows).to_csv(out/'by_panel.csv',index=False)
    d.save(out/'failures.json',failures)
    d.save(out/'input_sha256.json',hashes)
    d.state(status='stage_complete',stage='task4_net_repair_complete',
        next='Independently verify the analytical projection and portable reproduction before publication.')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
