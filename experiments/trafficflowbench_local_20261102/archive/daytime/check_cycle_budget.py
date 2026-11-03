"""Conditional numerical movement bounds on existing artificial cases, not validation."""
import common as d
import task4_span_shrink_stress as base
import numpy as np
import pandas as pd
import hashlib,msvcrt

def main():
    out=d.OUT/'task4_cycle_budget_check';protocol=out/'protocol.json';assert protocol.exists()
    df=pd.read_csv(d.OUT/'task4_od_cycle_stress/all_cases.csv');rows=[]
    for panel,group in df.groupby('panel'):
        paths,A,_,_,_=base.load(panel,'train');dest=pd.factorize(paths.destination_zone,sort=True)[0];K=float(A.sum(axis=0).max())
        for case,case_rows in group.groupby('case'):
            folder=d.WORK/'task4_od_cycle_cases'/hashlib.sha256(case.encode()).hexdigest()[:16]
            assert set(case_rows.input_artifact_sha256)=={d.sha(folder/'inputs.npz')}
            assert set(case_rows.prediction_artifact_sha256)=={d.sha(folder/'predictions.npz')}
            z=np.load(folder/'inputs.npz');pred=np.load(folder/'predictions.npz');truth=z['artificial_truth'][2];b=z['artificial_prior'][2];c=z['observed_counts'][2];f0=pred['baseline'];ref=base.metrics(f0,truth,b,A,c,dest)
            actual_epsilon=float(np.abs(A@truth-c).sum()/c.sum())
            for name in ['cycle005','cycle025']:
                delta=pred[name]-f0;movement=float(np.abs(delta).sum())
                for epsilon in [0.,.05]:
                    lower=(1-epsilon)*float(c.sum())/K;compatible=actual_epsilon<=epsilon+1e-12
                    for rho in [.001,.005]:
                        t=min(1.,rho*lower/max(movement,1e-12));f=f0+t*delta;m=base.metrics(f,truth,b,A,c,dest)
                        drift=max(abs(m[k]-ref[k]) for k in ['synthetic_S_link','synthetic_S_dev','synthetic_S_attr'])
                        od=m['synthetic_S_od']-ref['synthetic_S_od'];score=m['synthetic_S_ODME']-ref['synthetic_S_ODME']
                        assert drift<1e-9 and np.abs(f-f0).sum()<=rho*lower+1e-8
                        if compatible:assert abs(od)<=rho+1e-9 and abs(score)<=.45*rho+1e-9
                        rows.append(dict(panel=panel,case=case,method=name,epsilon=epsilon,rho=rho,blend=t,count_compatible=compatible,actual_count_NMAE=actual_epsilon,OD_delta=od,Task4_delta=score,max_other_metric_drift=drift,conditional_Task4_radius=.45*rho))
    frame=pd.DataFrame(rows);frame.to_csv(out/'all_budget_checks.csv',index=False)
    frame.groupby(['method','epsilon','rho']).agg(cases=('case','count'),compatible=('count_compatible','sum'),mean_delta=('Task4_delta','mean'),worst_delta=('Task4_delta','min'),mean_blend=('blend','mean')).to_csv(out/'summary.csv')
    d.save(out/'verification.json',dict(rows=len(frame),protocol_sha256=d.sha(protocol),source_sha256=d.sha(__file__),all_checks_passed=True,scope='Conditional inequalities on existing exposed artificial cases; not new validation or hidden-score safety.'))
    d.state(status='stage_complete',stage='cycle_budget_numeric_check_complete',next='Independently verify movement-bound audit; design fresh frozen stress only if assessing empirical behavior. Hidden count-compatibility remains unverified; no real replacement.')

if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:main()
        finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
