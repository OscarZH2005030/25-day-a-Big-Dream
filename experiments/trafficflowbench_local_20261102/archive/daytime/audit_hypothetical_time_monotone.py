"""Compare physical projections on the completed hypothetical inverse cases."""
import msvcrt

import common as d
import numpy as np
import pandas as pd
from scipy.optimize import isotonic_regression
from audit_hypothetical_time_conditioning import arrival_operator


def enforce_origin_totals(flow, origin, dest, totals):
    clipped = np.maximum(flow,0.)
    monotone = np.empty_like(flow)
    fallback = 0
    for zone,total in enumerate(totals):
        ids = np.flatnonzero(origin==zone)
        ids = ids[np.argsort(dest[ids])]
        if clipped[ids].sum()>0:
            clipped[ids] *= total/clipped[ids].sum()
        else:
            clipped[ids] = total/len(ids)
            fallback += 1
        downstream = np.r_[np.cumsum(flow[ids][::-1])[::-1],0.]
        inner = downstream[1:-1]
        if len(inner):
            inner = np.clip(isotonic_regression(inner,increasing=False).x,0.,total)
        monotone[ids] = -np.diff(np.r_[total,inner,0.])
    return clipped,monotone,fallback


def measurement_error(flow,origin,dest,times,spread,reference,noise):
    residual_energy, observed_energy = 0.,0.
    for cut in range(len(reference)-1):
        live = (origin<=cut)&(cut<dest)
        origin_flows = np.bincount(origin[live],weights=flow[live],minlength=cut+1)
        operator = arrival_operator(times,cut-np.arange(cut+1),spread)
        clean = operator@reference[cut,:cut+1]
        observed = clean+.01*np.linalg.norm(clean)*noise[cut]
        residual = operator@origin_flows-observed
        residual_energy += float(residual@residual)
        observed_energy += float(observed@observed)
    return float(np.sqrt(residual_energy/observed_energy))


def main():
    src=d.OUT/'task4_hypothetical_time_conditioning'
    out=d.OUT/'task4_hypothetical_time_monotone'
    d.save(out/'protocol.json',dict(
        scope='Posthoc estimator comparison on the same 80 artificial inverse cases. Uses hypothetical exact origin totals. No new trial, candidate or actual observation.',
        methods='Raw deconvolution, nonnegative path clipping and per-origin rescaling, cumulative isotonic projection with fixed origin and zero terminal boundaries.',
        source_sha256=d.sha(__file__),case_table_sha256=d.sha(src/'by_panel.csv')))
    rows=[]
    table=pd.read_csv(src/'by_panel.csv')
    for panel,group in table.groupby('panel'):
        paths=pd.read_csv(d.DATA/'corridors'/panel/'network/path_set.csv',dtype=str)
        origin,dest=[paths[key].str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy()
                     for key in ['origin_zone','destination_zone']]
        origin,dest=origin-origin.min(),dest-origin.min()
        times=np.arange(-24,int(dest.max()+1)+24)
        for spread,actions in group.groupby('spread'):
            file=d.WORK/'task4_hypothetical_time_conditioning_cases'/panel/f'spread{spread}.npz'
            assert set(actions.artifact_sha256)=={d.sha(file)}
            with np.load(file) as z:
                truth=z['truth']
                totals=np.bincount(origin,weights=truth)
                for row in actions.itertuples():
                    raw=z['reconstructed_od'][0 if row.cutoff<.01 else 1]
                    clipped,monotone,fallback=enforce_origin_totals(raw,origin,dest,totals)
                    for method,flow in [('raw',raw),('clip_and_scale',clipped),('isotonic',monotone)]:
                        if method!='raw':
                            assert flow.min()>=-1e-12
                            np.testing.assert_allclose(np.bincount(origin,weights=flow),totals,rtol=1e-12)
                        rows.append(dict(panel=panel,spread=spread,cutoff=row.cutoff,method=method,
                                         uniform_fallback_origins=fallback if method=='clip_and_scale' else 0,
                                         relative_od_L2=float(np.linalg.norm(flow-truth)/np.linalg.norm(truth)),
                                         relative_observation_L2=measurement_error(flow,origin,dest,times,spread,
                                                                                  z['downstream'],z['perturbation'])))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'by_panel.csv',index=False)
    frame.groupby(['spread','cutoff','method'])[['relative_od_L2','relative_observation_L2']].median().to_csv(out/'summary.csv')
    d.state(status='stage_complete',stage='hypothetical_time_monotone_complete',
        next='Review posthoc numerical robustness and scope. Hypothetical origin totals do not justify an official OD candidate.')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
