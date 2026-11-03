"""Independently check the local stencil and its portable reproduction."""
import json
import msvcrt
import subprocess
import sys

import common as d
import numpy as np
import pandas as pd
from task4_span_shrink_stress import load


def main():
    name = 'task4_local_span_null_stencil'
    out = d.OUT/name
    protocol = json.loads((out/'protocol.json').read_text())
    v = np.array(protocol['stencil'],dtype=np.int64)
    assert np.count_nonzero(v)==12
    r,c = np.nonzero(v)
    weights = v[r,c]
    for tags in [r,c,c-r]:
        assert all(weights[tags==tag].sum()==0 for tag in np.unique(tags))
    for axis,tags,other in [(0,c,r),(1,r,c)]:
        assert all((weights*other)[tags==tag].sum()==0 for tag in np.unique(tags)),axis
    frame = pd.read_csv(out/'by_panel.csv')
    errors = []
    for row in frame.itertuples():
        paths,A,_,prior,_ = load(row.panel,row.split)
        origin,dest = [paths[k].str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy()
                       for k in ['origin_zone','destination_zone']]
        by_pair = dict(zip(zip(origin,dest),range(len(paths))))
        file = d.WORK/'task4_extra_measurement_witnesses'/row.panel/row.split/'plus_histogram_and_moments.npz'
        with np.load(file) as data:
            flow = data['baseline']
        count, best = 0, 0.
        for i,j in by_pair:
            cells = [by_pair.get((i+a,j+b),-1) for a,b in zip(r,c)]
            if -1 in cells:
                continue
            f,b = flow[cells],prior[cells]
            signs = np.sign(f-b)
            if not signs.all() or signs@weights!=0:
                continue
            # Separate positive and negative directions. Each limit is a
            # distance to a nonnegativity or original-prior boundary.
            sizes = []
            for direction in [weights,-weights]:
                lower = np.where(signs>0,b,0.)
                upper = np.where(signs<0,b,np.inf)
                room = np.where(direction>0,upper-f,f-lower)
                sizes.append(.5*float(np.min(room/np.abs(direction))))
            step = max(sizes)
            if step<=1e-12:
                continue
            count += 1
            if step>best:
                best = step
                assert np.max(np.abs(A[:,cells]@weights))==0
        assert count==row.eligible_placements
        movement = best*np.abs(weights).sum()/flow.sum()
        errors.append(abs(movement-row.relative_od_movement))
    assert len(frame)==20 and max(errors)<1e-12
    d.save(out/'independent_readback.json',dict(groups=20,max_movement_error=max(errors),
        source_sha256=d.sha(__file__)))
    repeat = d.WORK/'publication_smoke/local_span_stencil'
    runner = d.WORK/'publish-repo/experiments/trafficflowbench_local_20261102/task4/audit_local_span_stencil.py'
    d.state(status='running',stage='local_span_stencil_reproduction')
    subprocess.run([sys.executable,'-X','utf8',str(runner),'--release-root',str(d.DATA),'--output',str(repeat)],check=True)
    pd.testing.assert_frame_equal(pd.read_csv(out/'by_panel.csv'),pd.read_csv(repeat/name/'by_panel.csv'),check_exact=True)
    assert (out/'summary.json').read_bytes()==(repeat/name/'summary.json').read_bytes()
    d.save(out/'portable_reproduction.json',dict(groups=20,exact_result_match=True,entrypoint_sha256=d.sha(runner)))
    d.state(status='stage_complete',stage='local_span_stencil_verified')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
