"""Compare the two invisible-repair directions at the same L1 movement."""
import hashlib
import msvcrt

import common as d
import numpy as np
import pandas as pd
from task4_span_shrink_stress import load
from verify_cycle_budget import score


def main():
    source=d.OUT/'task4_invariant_shrink_tiebreak/by_case.csv'
    out=d.OUT/'task4_equal_movement_ties'
    d.save(out/'protocol.json',dict(
        scope='Posthoc same-case direction comparison. No threshold tuning or fresh validation.',
        design='Scale default-tie direction to the full minimum-move tie L1 length. Preserve all non-OD components. The scaled default endpoint need not retain the optimal primary objective.',
        fractions=[.05,1.],source_sha256=d.sha(__file__),case_table_sha256=d.sha(source)))
    table=pd.read_csv(source).query("method=='minimum_move_tie' and blend==1")
    anchors=pd.read_csv(d.OUT/'task4_invariant_invisible_shrink/all_cases.csv').query('blend==0').set_index('case')
    rows=[]
    for panel,group in table.groupby('panel'):
        paths,A,_,_,_=load(panel,'train')
        dest=pd.factorize(paths.destination_zone,sort=True)[0]
        for row in group.itertuples():
            name=hashlib.sha256(row.case.encode()).hexdigest()[:16]+'.npz'
            original=d.WORK/'task4_invariant_invisible_shrink_cases'/name
            tied=d.WORK/'task4_invariant_shrink_tiebreak_cases'/name
            assert d.sha(original)==anchors.loc[row.case,'artifact_sha256'] and d.sha(tied)==row.artifact_sha256
            with np.load(original) as z,np.load(tied) as other:
                base,truth,prior,count=[z[k] for k in ['baseline','truth','prior','counts']]
                default,closest=z['proposed']-base,other['closest']-base
                mass_default,mass_closest=np.abs(default).sum(),np.abs(closest).sum()
                if mass_default<=1e-12:
                    assert mass_closest<=1e-7
                    scaled=np.zeros_like(default)
                else:
                    factor=mass_closest/mass_default
                    assert factor<=1.+1e-9
                    scaled=min(1.,factor)*default
                np.testing.assert_allclose(np.abs(scaled).sum(),mass_closest,rtol=1e-9,atol=1e-7)
                reference=score(base,truth,prior,A,count,dest)
                for fraction in [.05,1.]:
                    for method,direction in [('scaled_default',scaled),('minimum_move',closest)]:
                        f=base+fraction*direction
                        od,total=score(f,truth,prior,A,count,dest)
                        np.testing.assert_allclose(total-reference[1],.45*(od-reference[0]),rtol=0,atol=1e-10)
                        rows.append(dict(panel=panel,case=row.case,noise=row.noise,seed=row.seed,fraction=fraction,
                                         method=method,relative_movement=float(np.abs(f-base).sum()/base.sum()),
                                         od_delta=od-reference[0]))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'by_case.csv',index=False)
    wide=frame.pivot(index=['noise','fraction','case'],columns='method',values='od_delta')
    wide['minimum_move_advantage']=wide.minimum_move-wide.scaled_default
    wide.to_csv(out/'paired_od.csv')
    wide.groupby(level=['noise','fraction']).agg(cases=('minimum_move','size'),
        mean_minimum_move=('minimum_move','mean'),mean_scaled_default=('scaled_default','mean'),
        wins=('minimum_move_advantage',lambda x:int((x>1e-12).sum())),
        losses=('minimum_move_advantage',lambda x:int((x< -1e-12).sum()))).to_csv(out/'summary.csv')
    d.state(status='stage_complete',stage='task4_equal_movement_ties_complete',
        next='Verify equal movement and interpret the difference in direction on reused artificial cases.')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
