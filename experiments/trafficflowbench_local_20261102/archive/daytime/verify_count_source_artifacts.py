"""Read back fixed source-control cases and check observations, fits and six scores."""
import common as d
import task4_count_source_pair_stress as study
import numpy as np
import pandas as pd
import hashlib


def main():
    folder=d.OUT/'task4_count_source_pair_stress';df=pd.read_csv(folder/'all_cases.csv');assert len(df)==2880 and df.case.nunique()==480
    rng=np.random.default_rng(2026110324);cases=[]
    for sparse in [False,True]:
        choices=sorted(df.loc[(df.truth=='sparse')==sparse,'case'].unique());cases.extend(rng.choice(choices,10,replace=False).tolist())
    d.save(folder/'readback_protocol.json',dict(seed=2026110324,cases=cases,selection='Ten sparse and ten non-sparse numerical checks selected without using scores; not scientific heldout evaluation.',source_sha256=d.sha(__file__)))
    errors=[];flow_errors=[];count_errors=[];projection_errors=[]
    for case in cases:
        panel=case.split('/')[0];paths,A,_,_,_=study.family.base.load(panel,'train');X=study.family.structure.features(paths)['span_origin_destination']
        file=d.WORK/'task4_count_source_pair_cases'/(hashlib.sha256(case.encode()).hexdigest()[:16]+'.npz');z=np.load(file);rows=df[df.case==case].set_index('method');assert set(rows.artifact_sha256)=={d.sha(file)}
        truth,priors,counts,fold=z['artificial_truth'],z['artificial_prior'],z['observed_counts'],z['fold']
        count_errors.append(float(np.max(np.abs(truth@A.T-counts))))
        predictions,coefs,scales,source_flows,source_ok,gains=study.crossfit(X,A,counts,priors,fold)
        assert source_ok and rows.source_solver_success.all();flow_errors.append(float(np.max(np.abs(source_flows-z['source_count_estimates']))))
        assert np.array_equal(scales,z['target_training_scales'])
        for name in coefs:assert np.allclose(np.array(coefs[name]),z[name+'_coef'],atol=1e-12,rtol=1e-12)
        dest=paths.destination_zone.str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy()
        for name,(kind,alpha) in study.METHODS.items():
            prior=priors[2] if kind=='raw' else (1-alpha)*priors[2]+alpha*predictions[kind]
            prediction,status=study.family.base.ridge(A,counts[2],prior);assert status['success']
            metrics=study.family.base.metrics(prediction,truth[2],priors[2],A,counts[2],dest)
            errors.extend(abs(float(rows.loc[name,key])-value) for key,value in metrics.items())
        for key in ['raw_pair','count_pair']:
            projection_errors.append(abs(gains[key]-float(rows[key+'_prior_MSE_reduction'].iloc[0])))
    assert max(errors)<1e-10 and max(flow_errors)<1e-9 and max(count_errors)<1e-8 and max(projection_errors)<1e-10
    result=dict(cases=20,actions_recomputed=120,maximum_score_error=max(errors),maximum_source_estimate_error=max(flow_errors),maximum_generated_observed_count_error=max(count_errors),maximum_prior_prediction_gain_error=max(projection_errors),artifact_hashes_checked=True,source_sha256=d.sha(__file__),scope='Numerical/readback verification of artificial controls only; source latent paths verified against generated observations but are not features.')
    d.save(folder/'independent_readback_verification.json',result);print(result)


if __name__=='__main__':main()
