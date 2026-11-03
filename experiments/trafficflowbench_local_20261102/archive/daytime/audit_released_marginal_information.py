"""Identify which marginal queries belong to the released count row space."""
import msvcrt
from datetime import datetime, timezone

import common as d
import numpy as np
import pandas as pd
from task4_exact_pair_audit import base, basis


def main():
    out = d.OUT/'task4_released_marginal_information'
    d.save(out/'protocol.json',dict(
        registered_utc=datetime.now(timezone.utc).isoformat(),seed=None,
        scope='Linear observability audit for the released static count operator. Does not infer hidden margins, add detectors or calibrate a candidate.',
        queries='Individual nonempty origin and destination indicators, origin minus destination at each zone, and total demand.',
        method='Project each query onto the pivoted QR count row space. Record relative L2 residual and exact single-row equality separately. 1e-9 diagnostic relative tolerance.',
        source_sha256=d.sha(__file__)))
    rows,hashes=[],{}
    for panel in d.night.predictor.load_routes():
        for split in ['train','validation','private']:
            paths,A,_,_,inputs=base.load(panel,split)
            hashes.update(inputs)
            Q,rank=basis(A.T)
            origin=paths.origin_zone.to_numpy()
            dest=paths.destination_zone.to_numpy()
            queries=[('total','all',np.ones(len(paths)))]
            for zone in sorted(set(origin)|set(dest)):
                o=(origin==zone).astype(float)
                v=(dest==zone).astype(float)
                for name,query in [('origin',o),('destination',v),('net',o-v)]:
                    if query.any():
                        queries.append((name,zone,query))
            matrix=np.array([query for _,_,query in queries])
            residual=matrix-(matrix@Q)@Q.T
            relative=np.linalg.norm(residual,axis=1)/np.linalg.norm(matrix,axis=1)
            for index,(query_type,zone,query) in enumerate(queries):
                direct=bool(np.any(np.all(A==query,axis=1)))
                if direct:
                    assert relative[index]<1e-9
                rows.append(dict(panel=panel,split=split,query=query_type,zone=zone,
                                 count_rank=rank,relative_residual=float(relative[index]),
                                 identified=bool(relative[index]<1e-9),exact_single_row=direct))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'by_zone.csv',index=False)
    frame.groupby(['panel','split','query']).agg(
        queries=('identified','size'),identified=('identified','sum'),
        direct=('exact_single_row','sum'),median_residual=('relative_residual','median')).to_csv(out/'by_panel.csv')
    frame.groupby('query').agg(queries=('identified','size'),identified=('identified','sum'),
                               direct=('exact_single_row','sum')).to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',[])
    d.state(status='stage_complete',stage='released_marginal_information_complete',
        next='Independently audit the row-space classification. These are structural queries, not verified hidden margins.')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
