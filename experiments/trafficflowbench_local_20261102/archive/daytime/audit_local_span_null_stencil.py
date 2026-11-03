"""Find a local exact-integer OD direction surviving hypothetical span summaries."""
import msvcrt

import common as d
import numpy as np
import pandas as pd
from task4_span_shrink_stress import load


def main():
    out = d.OUT/'task4_local_span_null_stencil'
    source = d.OUT/'task4_extra_measurement_audit/witness_summary.csv'
    stencil = np.zeros((4,4),dtype=np.int64)
    second = np.outer([1,-2,1],[1,-2,1])
    stencil[:3,:3] += second
    stencil[1:,1:] -= second
    r,c = np.nonzero(stencil)
    weights = stencil[r,c]
    assert not stencil.sum(axis=0).any() and not stencil.sum(axis=1).any()
    assert not (stencil @ np.arange(4)).any() and not (np.arange(4) @ stencil).any()
    assert all(stencil[np.indices(stencil.shape)[1]-np.indices(stencil.shape)[0]==k].sum()==0
               for k in range(-3,4))
    d.save(out/'protocol.json',dict(scope='Integer local null direction with a feasible numerical step around frozen flow. Hypothetical summaries only.',
        stencil=stencil.tolist(),nonzero_cells=len(weights),source_sha256=d.sha(__file__),anchor_table_sha256=d.sha(source)))
    rows=[]
    table=pd.read_csv(source).query("method=='plus_histogram_and_moments'")
    for row in table.itertuples():
        paths,A,_,prior,_=load(row.panel,row.split)
        origin,dest=[paths[key].str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy()
                     for key in ['origin_zone','destination_zone']]
        lookup=dict(zip(zip(origin,dest),range(len(paths))))
        anchor=d.WORK/'task4_extra_measurement_witnesses'/row.panel/row.split/f'{row.method}.npz'
        assert d.sha(anchor)==row.artifact_sha256
        with np.load(anchor) as z:
            flow=z['baseline']
        sign=np.sign(flow-prior)
        lower=np.where(sign>0,prior,0.)
        upper=np.where(sign<0,prior,np.inf)
        best=None
        eligible=0
        for i in np.unique(origin):
            for j in np.unique(dest):
                indices=[lookup.get((i+a,j+b)) for a,b in zip(r,c)]
                if None in indices or (sign[indices]==0).any() or sign[indices]@weights!=0:
                    continue
                v=flow[indices]
                a=(lower[indices]-v)/weights
                b=(upper[indices]-v)/weights
                lo=float(np.minimum(a,b).max())
                hi=float(np.maximum(a,b).min())
                step=.5*(lo if abs(lo)>abs(hi) else hi)
                if abs(step)<=1e-12:
                    continue
                eligible+=1
                if best is None or abs(step)>abs(best[0]):
                    best=(step,indices)
        movement=0.
        if best is not None:
            step,indices=best
            assert np.array_equal(A[:,indices]@weights,np.zeros(len(A)))
            changed=flow.copy()
            changed[indices]+=step*weights
            assert changed.min()>=-1e-12
            np.testing.assert_allclose(np.abs(changed-prior).sum(),np.abs(flow-prior).sum(),rtol=1e-12)
            movement=float(np.abs(changed-flow).sum()/flow.sum())
        rows.append(dict(panel=row.panel,split=row.split,eligible_placements=eligible,relative_od_movement=movement))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'by_panel.csv',index=False)
    d.save(out/'summary.json',dict(nonzero_cells=len(weights),groups=len(frame),
        feasible_groups=int((frame.eligible_placements>0).sum()),minimum_movement=float(frame.relative_od_movement.min()),
        maximum_movement=float(frame.relative_od_movement.max())))
    d.state(status='stage_complete',stage='local_span_null_stencil_complete',
        next='Verify integer stencil and numerical witness steps independently before publication.')


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
