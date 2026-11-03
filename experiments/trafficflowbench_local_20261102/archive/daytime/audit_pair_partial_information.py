"""Audit the partial ridge information behind the observed shrinkage."""
import hashlib
import msvcrt
import common as d
import numpy as np
import pandas as pd
from task4_count_source_pair_stress import family, qr

def main():
    src=d.OUT/'task4_pair_standardization_stress'
    out=d.OUT/'task4_pair_partial_information'
    d.save(out/'protocol.json',dict(scope='Algebra and readback on exposed cases only.',source_sha256=d.sha(__file__),case_table_sha256=d.sha(src/'all_cases.csv'),formula='k = u.T @ (u - X ridge(X,u)). Standardized/raw coefficient contribution ratio = (k+1/rms^2)/(k+1).'))
    rows=[]
    cases=pd.read_csv(src/'all_cases.csv').query("method=='baseline'")
    for panel,group in cases.groupby('panel'):
        paths,_,_,_,_=family.base.load(panel,'train')
        X=family.structure.features(paths)['span_origin_destination']
        for row in group.itertuples():
            key=hashlib.sha256(row.case.encode()).hexdigest()[:16]
            file=d.WORK/'task4_pair_standardization_cases'/key/'case.npz'
            assert d.sha(file)==row.artifact_sha256
            with np.load(file) as z:
                priors=z['prior']
                source=np.mean([b/max(float(np.median(b[b>0])),1e-9) for b in priors[:2]],axis=0)
                pair=source-X@qr.ridge_qr(X,source)
                for k,rms in enumerate(z['pair_scales']):
                    train=z['fold']!=k
                    u=pair[train]/rms
                    features=X[train]
                    information=float(u@(u-features@qr.ridge_qr(features,u)))
                    ratio=(information+1/rms**2)/(information+1)
                    predicted=z['original_coef'][k,-1]*rms*ratio
                    observed=z['standardized_coef'][k,-1]
                    error=float(abs(predicted-observed)/max(1.,abs(observed)))
                    assert error<1e-9
                    rows.append(dict(panel=panel,case=row.case,family=row.family,noise=row.noise,fold=k,
                                     partial_information=information,predicted_shrink_ratio=ratio,relative_identity_error=error))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'by_fold.csv',index=False)
    frame.groupby(['family','noise'])[['partial_information','predicted_shrink_ratio','relative_identity_error']].agg(['min','median','max']).to_csv(out/'summary.csv')
    d.state(status='stage_complete',stage='pair_standardization_published_and_information_audited',published_commit='af4d963becfd106e80aba1a9d97720238185e02f',next='Consider fixed sample-size-scaled pair penalty based on partial information. Existing cases diagnose strength only and do not select it.')

if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
