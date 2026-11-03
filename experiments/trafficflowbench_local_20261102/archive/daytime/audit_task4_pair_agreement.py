"""Measure cross-input residual agreement at the same and nearby pairs."""
import itertools
import msvcrt
import common as d
import numpy as np
import pandas as pd
from task4_local_pair_transfer import exact,source_features

def cosine(a,b):
    return float((a@b)/max(1e-12,np.linalg.norm(a)*np.linalg.norm(b)))

def main():
    out=d.OUT/'task4_pair_agreement_audit'
    d.save(out/'protocol.json',dict(scope='Posthoc exposed public input residual agreement. No latent labels or choice of forecasting rule.',source_sha256=d.sha(__file__),locality_source_sha256=d.sha(d.WORK/'task4_local_pair_transfer.py')))
    rows,hashes=[],{}
    for panel in d.night.predictor.load_routes():
        priors={}
        for split in exact.SPLITS:
            paths,_,pcount,prior,inputs=exact.base.load(panel,split)
            hashes.update(inputs)
            priors[split]=prior/max(float(np.median(prior[prior>0])),1e-9)
        X=exact.structure.features(paths)['span_origin_destination']
        Q,_=exact.basis(X)
        columns={split:source_features(paths,X,Q,p)[0] for split,p in priors.items()}
        for left,right in itertools.permutations(priors,2):
            target=priors[right]-Q@(Q.T@priors[right])
            for method in ['same_pair','adjacent_span','distant_span']:
                rows.append(dict(panel=panel,source=left,target=right,method=method,
                                 residual_cosine=cosine(columns[left][method][:,-1],target)))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'by_panel_source_target.csv',index=False)
    frame.groupby('method').residual_cosine.agg(['min','median','max']).to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.state(status='stage_complete',stage='local_pair_published_and_agreement_audited',published_commit='c40b250',next='Use exposed residual agreement to design a frozen falsification control. It cannot distinguish persistent truth and persistent error.')

if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
