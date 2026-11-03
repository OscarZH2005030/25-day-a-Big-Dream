"""Measure identifiability under idealized time-resolved zone-cut counts."""
import hashlib
import msvcrt
from datetime import datetime, timezone

import common as d
import numpy as np
import pandas as pd
from task4_exact_pair_audit import basis

SEED = 2026110342


def main():
    out = d.OUT/'task4_hypothetical_time_bins'
    d.save(out/'protocol.json',dict(
        registered_utc=datetime.now(timezone.utc).isoformat(),seed=SEED,widths=[1,8,16,32],
        construction='Idealized single known departure pulse, travel time to each zone cut equals cut minus origin zone index. Group traversing trips into arrival bins of the stated width.',
        scope='Hypothetical design using published OD support. No travel-time, detector, mapping or alignment assumptions are asserted for actual data.',
        check='Width one identifies origin-resolved downstream counts. Recover each OD by differencing consecutive cuts on one fixed integer draw per network.',
        source_sha256=d.sha(__file__)))
    rows, hashes = [], {}
    for panel in d.night.predictor.load_routes():
        file=d.DATA/'corridors'/panel/'network/path_set.csv'
        hashes[str(file.relative_to(d.DATA))]=d.sha(file)
        paths=pd.read_csv(file,dtype=str)
        origin,dest=[paths[k].str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy()
                     for k in ['origin_zone','destination_zone']]
        zones=np.unique(np.r_[origin,dest])
        assert np.array_equal(zones,np.arange(zones.min(),zones.max()+1))
        assert len(paths)==len(zones)*(len(zones)-1)//2 and len(set(zip(origin,dest)))==len(paths)
        origin=origin-zones.min()
        dest=dest-zones.min()
        n=len(zones)
        margins=np.vstack([np.arange(n-1)[:,None]==origin,np.arange(1,n)[:,None]==dest])
        _,rank_margin=basis(margins.T.astype(float))
        seed=int.from_bytes(hashlib.sha256(f'{SEED}/{panel}'.encode()).digest()[:8],'little')%(2**63)
        truth=np.random.default_rng(seed).integers(0,1000,size=len(paths))
        counts=np.zeros((n,n),dtype=np.int64)
        for cut in range(n-1):
            live=(origin<=cut)&(cut<dest)
            np.add.at(counts[cut],origin[live],truth[live])
        reconstruction=counts[dest-1,origin]-counts[dest,origin]
        assert np.array_equal(reconstruction,truth)
        rows.append(dict(panel=panel,width=1,paths=len(paths),marginal_rank=rank_margin,
                         joint_rank=len(paths),rank_beyond_margins=len(paths)-rank_margin,
                         linear_nullity=0,reconstruction_max_error=0,seed=seed))
        for width in [8,16,32]:
            measurements=[]
            for cut in range(n-1):
                live=(origin<=cut)&(cut<dest)
                group=(cut-origin)//width
                for bin_id in np.unique(group[live]):
                    measurements.append(live&(group==bin_id))
            _,rank=basis(np.vstack([margins,*measurements]).T.astype(float))
            rows.append(dict(panel=panel,width=width,paths=len(paths),marginal_rank=rank_margin,
                             joint_rank=rank,rank_beyond_margins=rank-rank_margin,
                             linear_nullity=len(paths)-rank,reconstruction_max_error=None,seed=seed))
        pd.DataFrame(rows).to_csv(out/'by_panel_width.csv',index=False)
        d.state(status='running',stage='hypothetical_time_bins',panel=panel)
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',[])
    d.state(status='stage_complete',stage='hypothetical_time_bins_complete',
        next='Independently check coarse-bin ranks and the width-one reconstruction before publication.')


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
