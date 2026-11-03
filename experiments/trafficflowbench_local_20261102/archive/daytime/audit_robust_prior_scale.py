"""Describe Huber cutoff and residual activity on the existing fixed folds."""
import hashlib
import msvcrt
import common as d
import numpy as np
import pandas as pd
from task4_robust_prior_forecast import g

def main():
    source_dir=d.OUT/'task4_robust_prior_forecast'
    out=d.OUT/'task4_robust_prior_scale_audit'
    d.save(out/'protocol.json',dict(scope='Posthoc existing-fold diagnostic. No cutoff fitting or fresh evaluation.',source_sha256=d.sha(__file__),case_table_sha256=d.sha(source_dir/'all_cases.csv')))
    rows=[]
    cases=pd.read_csv(source_dir/'all_cases.csv').query("method == 'baseline'")
    for panel,group in cases.groupby('panel'):
        paths,_,_,_,_=g.family.base.load(panel,'train')
        X=g.family.structure.features(paths)['span_origin_destination']
        for row in group.itertuples():
            key=hashlib.sha256(row.case.encode()).hexdigest()[:16]
            file=d.WORK/'task4_robust_prior_cases'/key/'case.npz'
            assert d.sha(file)==row.artifact_sha256
            with np.load(file) as z:
                priors=z['prior']
                src=np.mean([b/max(float(np.median(b[b>0])),1e-9) for b in priors[:2]],axis=0)
                design=np.column_stack([X,src-X@g.qr.ridge_qr(X,src)])
                for fold,scale in enumerate(z['scales']):
                    train=z['fold']!=fold
                    target=priors[2,train]/scale
                    residual=design[train]@z['squared_coef'][fold]-target
                    rows.append(dict(panel=panel,case=row.case,family=row.family,noise=row.noise,fold=fold,
                                     physical_cutoff=scale,clipped_residual_fraction=float(np.mean(np.abs(residual)>1)),
                                     huber_objective_drop=float(z['huber_objective_drops'][fold])))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'by_fold.csv',index=False)
    frame.groupby(['family','noise'])[['physical_cutoff','clipped_residual_fraction']].agg(['min','median','max']).to_csv(out/'summary.csv')
    d.state(status='stage_complete',stage='robust_prior_published_and_scale_audited',published_commit='7a0902a966bba071039daf69eb7529db366eb113',next='Examine scale sensitivity before a fresh fixed loss comparison. Reused folds do not select a cutoff.')

if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
