"""Test whether all released static path-incidence rows add matching information."""
import msvcrt
from datetime import datetime, timezone

import common as d
import numpy as np
import pandas as pd
from task4_exact_pair_audit import basis


def main():
    out = d.OUT/'task4_full_path_incidence'
    d.save(out/'protocol.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),seed=None,
        scope='Structural test of every released path-link incidence row, including rows absent from Task4 counts. No detector measurements, physical mapping or time alignment assumed.',
        identity='Compare A[:,o,d] with A[:,o,last] + A[:,first,d] - A[:,first,last] in exact integer arithmetic.',
        source_sha256=d.sha(__file__)))
    rows, hashes, failures = [], {}, []
    for panel in d.night.predictor.load_routes():
        net=d.DATA/'corridors'/panel/'network'
        pathfile,incfile=net/'path_set.csv',net/'path_link_incidence.csv'
        paths=pd.read_csv(pathfile,dtype=str)
        incidence=pd.read_csv(incfile,dtype=str)
        for file in [pathfile,incfile]:
            hashes[str(file.relative_to(d.DATA))]=d.sha(file)
        origin,dest=[paths[key].str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy()
                     for key in ['origin_zone','destination_zone']]
        assert not paths.path_id.duplicated().any()
        lookup=dict(zip(zip(origin,dest),range(len(paths))))
        assert len(lookup)==len(paths)
        first,last=int(origin.min()),int(dest.max())
        links=pd.Index(incidence.link_id.unique()).sort_values()
        A=np.zeros((len(links),len(paths)),dtype=np.int16)
        link_index=links.get_indexer(incidence.link_id)
        path_index=pd.Index(paths.path_id).get_indexer(incidence.path_id)
        assert (path_index>=0).all() and (link_index>=0).all()
        A[link_index,path_index]=1
        right=A[:,[lookup[o,last] for o in origin]]
        left=A[:,[lookup[first,v] for v in dest]]
        mismatch=A-(right+left-A[:,lookup[first,last],None])
        failed_rows=np.flatnonzero(np.any(mismatch,axis=1))
        for i in failed_rows:
            failures.append(dict(panel=panel,link_id=links[i],mismatched_paths=int(np.count_nonzero(mismatch[i]))))
        counted=set()
        for split in ['train','validation','private']:
            file=d.DATA/'task4'/panel/split/'synthetic_link_counts.csv'
            hashes[str(file.relative_to(d.DATA))]=d.sha(file)
            counted.update(pd.read_csv(file,usecols=['link_id'],dtype=str).link_id)
        uncounted=~links.isin(counted)
        margins=np.column_stack([np.unique(origin)[None,:]==origin[:,None],
                                 np.unique(dest)[None,:]==dest[:,None]]).astype(float)
        _,margin_rank=basis(margins)
        _,joint_rank=basis(np.column_stack([margins,A.T]))
        rows.append(dict(panel=panel,paths=len(paths),all_incidence_rows=len(links),
                         uncounted_rows=int(uncounted.sum()),failed_rows=len(failed_rows),
                         failed_uncounted_rows=int(uncounted[failed_rows].sum()),
                         maximum_integer_error=int(np.abs(mismatch).max()),
                         rank_beyond_margins=joint_rank-margin_rank))
    pd.DataFrame(rows).to_csv(out/'by_panel.csv',index=False)
    d.save(out/'failures.json',failures)
    d.save(out/'input_sha256.json',hashes)
    d.state(status='stage_complete',stage='full_path_incidence_audit_complete',
        next='Independently verify all-link identity and rank before publishing. Released incidence is not a time-dependent detector model.')


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
