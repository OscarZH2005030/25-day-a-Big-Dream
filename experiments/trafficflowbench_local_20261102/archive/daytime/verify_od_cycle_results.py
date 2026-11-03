"""Read back every saved cycle action using independent NumPy scoring formulas."""
import common as d
import task4_span_shrink_stress as base
import numpy as np
import pandas as pd
import hashlib


def main():
    out=d.OUT/'task4_od_cycle_stress';df=pd.read_csv(out/'all_cases.csv');assert len(df)==2880 and df.case.nunique()==480
    errors=[];invariants=[];target_errors=[];hashes=0
    for panel,group in df.groupby('panel'):
        paths,A,_,_,_=base.load(panel,'train');origin=paths.origin_zone.str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy();dest=paths.destination_zone.str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy()
        for case,rows in group.groupby('case'):
            folder=d.WORK/'task4_od_cycle_cases'/hashlib.sha256(case.encode()).hexdigest()[:16];inputs=folder/'inputs.npz';predictions=folder/'predictions.npz'
            assert set(rows.input_artifact_sha256)=={d.sha(inputs)} and set(rows.prediction_artifact_sha256)=={d.sha(predictions)};hashes+=2
            z=np.load(inputs);p=np.load(predictions);truth=z['artificial_truth'][2];b=z['artificial_prior'][2];c=z['observed_counts'][2];f0=p['baseline'];rows=rows.set_index('method');true_dest=np.bincount(dest,weights=truth);true_dest/=true_dest.sum();trueD=max(float(np.abs(truth-b).sum()),1e-12)
            for name in rows.index:
                f=p[name];assert np.isfinite(f).all() and f.min()>=-1e-9
                relative=float(np.abs(f-truth).sum()/truth.sum());od=max(0,1-relative);link=max(0,1-float(np.abs(A@f-c).sum()/c.sum()));dev=float(np.exp(-abs(float(np.abs(f-b).sum())/trueD-1)));at=np.bincount(dest,weights=f);at/=at.sum();attr=max(0,1-float(.5*np.abs(at-true_dest).sum()));score=.45*od+.25*link+.15*dev+.15*attr
                calculated=dict(synthetic_S_od=od,synthetic_S_link=link,synthetic_S_dev=dev,synthetic_S_attr=attr,synthetic_S_ODME=score,true_relative_OD_L1=relative,relative_L1_movement=float(np.abs(f-f0).sum()/max(f0.sum(),1e-12)))
                errors.extend(abs(float(rows.loc[name,key])-value) for key,value in calculated.items())
                if name.startswith('cycle'):
                    delta=f-f0;drift=max(float(np.max(np.abs(A@delta))),float(np.max(np.abs(np.bincount(origin,weights=delta)))),float(np.max(np.abs(np.bincount(dest,weights=delta)))),abs(float(np.abs(f-b).sum()-np.abs(f0-b).sum())))
                    assert drift<1e-7;invariants.append(drift)
                    target=p[name.replace('cycle','direct')];distance=float(np.abs(f-target).sum());target_errors.append(abs(distance-float(rows.loc[name,'target_L1_after'])));assert distance<=float(np.abs(f0-target).sum())+1e-8
                    # The L1 descent surrogate also bounds movement by twice its initial target distance.
                    assert np.abs(f-f0).sum()<=2*np.abs(target-f0).sum()+1e-8
    assert max(errors)<1e-10 and max(target_errors)<1e-9
    result=dict(cases=480,actions=2880,input_and_prediction_artifact_hashes_verified=hashes,independent_score_formula='NumPy direct OD L1, count reconstruction, original-prior deviation and bincount destination TV',maximum_score_or_movement_error=max(errors),maximum_constraint_drift=max(invariants),maximum_target_distance_readback_error=max(target_errors),all_L1_surrogates_nonincreasing=True,source_sha256=d.sha(__file__),scope='Numerical audit of known artificial labels only; no hidden OD validation or guaranteed OD improvement.')
    d.save(out/'independent_all_case_readback.json',result);print(result)


if __name__=='__main__':main()
