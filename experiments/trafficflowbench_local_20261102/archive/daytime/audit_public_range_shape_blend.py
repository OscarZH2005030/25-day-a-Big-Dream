import hashlib
import msvcrt
import common as d
import numpy as np
import pandas as pd
import task4_span_shrink_stress as base
from verify_cycle_budget import score

def main():
    original=d.OUT/'task4_public_range_shape_stress'
    out=d.OUT/'task4_public_range_shape_blend'
    d.save(out/'protocol.json',dict(weights=[.25,.5,.75],scope='Posthoc fixed flow interpolation on existing 80 cases. No model selection or fresh validation.',source_sha256=d.sha(__file__),input_sha256=d.sha(original/'all_cases.csv')))
    rows=[]
    cases=pd.read_csv(original/'all_cases.csv').query("method == 'baseline'")
    for panel,group in cases.groupby('panel'):
        paths,A,_,_,_=base.load(panel,'train')
        dest=pd.factorize(paths.destination_zone,sort=True)[0]
        for row in group.itertuples():
            key=hashlib.sha256(row.case.encode()).hexdigest()[:16]
            file=d.WORK/'task4_public_range_shape_cases'/key/'case.npz'
            assert d.sha(file)==row.artifact_sha256
            with np.load(file) as z:
                for weight in [.25,.5,.75]:
                    flow=(1-weight)*z['geometry005']+weight*z['pair005']
                    od,total=score(flow,z['truth'][2],z['supplied_prior'][2],A,z['counts'][2],dest)
                    rows.append(dict(case=row.case,panel=panel,noise=row.noise,pair_weight=weight,delta=total-row.synthetic_S_ODME,od_delta=od-row.synthetic_S_od))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'all_cases.csv',index=False)
    frame.groupby(['noise','pair_weight']).delta.agg(['mean','min',lambda x:int((x < -1e-12).sum())]).to_csv(out/'summary.csv')
    d.state(status='stage_complete',stage='public_range_shape_published_and_blend_checked',published_commit='d72b4c336482943adc2810827c24d5b2d2f4def9',next='Review matched-shape error mechanisms and consider a frozen follow-up targeting OD pair error. Do not select a mixture from these reused cases.')

if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
