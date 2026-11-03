"""Compare weak-prior forecast rank and OD rank on the existing half-pair stress."""
import hashlib
import msvcrt
import common as d
import numpy as np
import pandas as pd
from task4_count_source_pair_stress import family, qr

def main():
    src=d.OUT/'task4_half_pair_stress'
    out=d.OUT/'task4_half_pair_surrogate_audit'
    d.save(out/'protocol.json',dict(scope='Posthoc on existing cases, no selection or fresh evaluation.',source_sha256=d.sha(__file__),case_table_sha256=d.sha(src/'all_cases.csv')))
    rows=[]
    cases=pd.read_csv(src/'all_cases.csv')
    for panel,group in cases.groupby('panel'):
        paths,_,_,_,_=family.base.load(panel,'train')
        X=family.structure.features(paths)['span_origin_destination']
        for case,actions in group.groupby('case'):
            key=hashlib.sha256(case.encode()).hexdigest()[:16]
            file=d.WORK/'task4_half_pair_cases'/key/'case.npz'
            assert set(actions.artifact_sha256)=={d.sha(file)}
            with np.load(file) as z:
                priors=z['prior']
                source=np.mean([b/max(float(np.median(b[b>0])),1e-9) for b in priors[:2]],axis=0)
                pair=source-X@qr.ridge_qr(X,source)
                for row in actions.query("method!='baseline'").itertuples():
                    prediction=np.empty(len(X))
                    for k,rms in enumerate(z['pair_scales']):
                        test=z['fold']==k
                        design=np.column_stack([X[test],pair[test]/rms])
                        prediction[test]=np.maximum(design@z[row.method+'_coef'][k],0)*z['scales'][k]
                    loss=np.abs(prediction-priors[2]).sum()/priors[2].sum()
                    rows.append(dict(panel=panel,case=case,noise=row.noise,method=row.method,prior_relative_L1=loss,
                                     synthetic_S_od=row.synthetic_S_od,synthetic_S_ODME=row.synthetic_S_ODME))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'by_case.csv',index=False)
    ranks=[]
    for case,group in frame.groupby('case'):
        chosen=group.loc[group.prior_relative_L1.idxmin()]
        ranks.append(dict(case=case,panel=chosen.panel,noise=chosen.noise,prior_best=chosen.method,
                          od_best=group.loc[group.synthetic_S_od.idxmax()].method,
                          od_regret=float(group.synthetic_S_od.max()-chosen.synthetic_S_od)))
    ranking=pd.DataFrame(ranks)
    ranking.to_csv(out/'ranks.csv',index=False)
    ranking['agreement']=ranking.prior_best==ranking.od_best
    ranking.groupby('noise').agg(cases=('case','count'),agreements=('agreement','sum'),mean_od_regret=('od_regret','mean')).to_csv(out/'summary.csv')
    d.state(status='stage_complete',stage='half_pair_published_and_surrogate_audited',published_commit='f267e93b7fd758ffbc291936c46db76c415dfbb4',next='Compare surrogate rank disagreement and investigate signal transfer without tuning on reused cases.')

if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
