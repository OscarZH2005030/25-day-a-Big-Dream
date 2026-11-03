"""Exact rank of unit-bin counts with known origin-specific pulse offsets."""
import hashlib
import msvcrt
from datetime import datetime, timezone

import common as d
import numpy as np
import pandas as pd

SEED = 2026110343


def measure(origin,dest,flow,groups,n):
    counts=np.zeros((n,len(np.unique(groups))),dtype=np.int64)
    for cut in range(n-1):
        live=(origin<=cut)&(cut<dest)
        np.add.at(counts[cut],groups[origin[live]],flow[live])
    return counts


def main():
    out = d.OUT/'task4_hypothetical_time_alias'
    d.save(out/'protocol.json',dict(
        registered_utc=datetime.now(timezone.utc).isoformat(),seed=SEED,
        design='Known integer pulse epoch per origin and deterministic delay cut minus origin. Origins with equal epoch-minus-origin share an arrival stream. Compare all-zero epochs, bounded jitter, half compensation and full compensation.',
        proof='With origin margins, difference consecutive counts in each stream and add entering origin total to recover stream-specific destinations. The resulting bipartite graph per stream is connected through the last destination. Rank is origin nodes plus reachable destination nodes minus one per stream.',
        scope='Idealized measurement design. No actual travel-time, pulse, detector mapping or independent timing evidence. Jitter is known to the hypothetical observer.',
        source_sha256=d.sha(__file__)))
    rows, hashes=[],{}
    for panel in d.night.predictor.load_routes():
        file=d.DATA/'corridors'/panel/'network/path_set.csv'
        hashes[str(file.relative_to(d.DATA))]=d.sha(file)
        paths=pd.read_csv(file,dtype=str)
        origin,dest=[paths[k].str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy()
                     for k in ['origin_zone','destination_zone']]
        origin,dest=origin-origin.min(),dest-origin.min()
        n=int(dest.max()+1)
        assert len(paths)==n*(n-1)//2 and len(set(zip(origin,dest)))==len(paths) and (origin<dest).all()
        lookup=dict(zip(zip(origin,dest),range(len(paths))))
        seed=int.from_bytes(hashlib.sha256(f'{SEED}/{panel}'.encode()).digest()[:8],'little')%(2**63)
        rng=np.random.default_rng(seed)
        flow=rng.integers(0,1000,size=len(paths))
        zones=np.arange(n-1)
        epochs={'simultaneous':np.zeros(n-1,dtype=int),'jitter4':rng.integers(-4,5,n-1),
                'half_compensation':zones//2,'full_compensation':zones}
        for scenario,epoch in epochs.items():
            _,groups=np.unique(epoch-zones,return_inverse=True)
            counts=measure(origin,dest,flow,groups,n)
            entries=np.zeros_like(counts)
            np.add.at(entries,(origin,groups[origin]),flow)
            by_destination=np.zeros_like(counts)
            np.add.at(by_destination,(dest,groups[origin]),flow)
            recovered=counts[:-1]-counts[1:]+entries[1:]
            np.testing.assert_array_equal(recovered,by_destination[1:])
            rank=0
            witness=None
            for group in np.unique(groups):
                starts=zones[groups==group]
                rank+=len(starts)+(n-1-int(starts.min()))-1
                if len(starts)>1 and starts[1]<n-2 and witness is None:
                    a,b=starts[:2]
                    witness=np.zeros(len(paths),dtype=np.int64)
                    witness[[lookup[a,n-2],lookup[a,n-1],lookup[b,n-2],lookup[b,n-1]]]=[1,-1,-1,1]
            if rank<len(paths):
                assert witness is not None and not measure(origin,dest,witness,groups,n).any()
            rows.append(dict(panel=panel,scenario=scenario,paths=len(paths),streams=len(np.unique(groups)),
                             exact_rank=rank,rank_beyond_margins=rank-(2*n-3),linear_nullity=len(paths)-rank,
                             unit_reconstruction_error=0,exact_four_cell_witness=witness is not None,seed=seed))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'by_panel.csv',index=False)
    frame.groupby('scenario')[['streams','rank_beyond_margins','linear_nullity']].agg(['min','max']).to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',[])
    d.state(status='stage_complete',stage='hypothetical_time_alias_complete',
        next='Verify grouping rank with independent graph construction and portable reproduction.')


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
