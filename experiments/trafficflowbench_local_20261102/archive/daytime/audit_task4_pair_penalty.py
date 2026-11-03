"""Measure the effective ridge penalty after standardizing the pair column."""
import hashlib
import msvcrt
import common as d
import numpy as np
import pandas as pd
from task4_count_source_pair_stress import family, qr

def diagnostic(X, priors):
    scales=[max(float(np.median(v[v>0])),1e-9) for v in priors]
    source=np.mean([b/s for b,s in zip(priors,scales)],axis=0)
    residual=source-X@qr.ridge_qr(X,source)
    rms=float(np.sqrt(np.mean(residual**2)))
    return dict(source_scale_min=min(scales),source_scale_max=max(scales),pair_rms=rms,standardized_pair_penalty=1/rms**2 if rms>0 else None)

def main():
    out=d.OUT/'task4_pair_penalty_audit'
    src=d.OUT/'task4_sparse_cutoff_stress/all_cases.csv'
    d.save(out/'protocol.json',dict(scope='Descriptive existing cases and public inputs. No estimator or penalty selected.',source_sha256=d.sha(__file__),case_table_sha256=d.sha(src),identity='For z=rms*u, penalty beta^2 on z equals theta^2/rms^2 on u with theta=rms*beta.'))
    cases=pd.read_csv(src).query("method=='baseline'")
    rows,hashes=[],{}
    for panel,group in cases.groupby('panel'):
        paths,_,_,_,inputs=family.base.load(panel,'train')
        hashes.update(inputs)
        X=family.structure.features(paths)['span_origin_destination']
        for row in group.itertuples():
            key=hashlib.sha256(row.case.encode()).hexdigest()[:16]
            file=d.WORK/'task4_sparse_cutoff_cases'/key/'case.npz'
            assert d.sha(file)==row.artifact_sha256
            with np.load(file) as z:
                rows.append(dict(panel=panel,case=row.case,group=row.noise,**diagnostic(X,z['prior'][:2])))
        sources=[]
        for split in ['train','validation']:
            _,_,_,prior,inputs=family.base.load(panel,split)
            hashes.update(inputs)
            sources.append(prior)
        rows.append(dict(panel=panel,case=panel,group='public_train_validation',**diagnostic(X,sources)))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'by_panel_case.csv',index=False)
    frame.groupby('group')[['pair_rms','standardized_pair_penalty']].agg(['min','median','max']).to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.state(status='stage_complete',stage='sparse_cutoff_published_and_pair_penalty_audited',published_commit='709b935b90590035ba2f5be90557621115fcf0ed',next='Consider fresh fixed standardized-pair versus original-penalty control. Diagnostic is exposed and does not select a penalty.')

if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
