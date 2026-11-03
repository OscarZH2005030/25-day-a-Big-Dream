"""Independent SVD and score readback on fixed samples of the repaired study."""
import common as d
import task4_crossmonth_shared_noise_stress as study
import task4_crossmonth_shared_noise_qr as qr
import numpy as np
import pandas as pd
import json
import hashlib
from scipy.linalg import lstsq

O=d.OUT/'task4_crossmonth_shared_noise_stress_qr'


def svd_ridge(X,y):
    penalty=np.eye(X.shape[1]);penalty[0,0]=0
    A=np.vstack([X,penalty]);b=np.r_[y,np.zeros(X.shape[1])]
    scale=np.maximum(np.linalg.norm(A,axis=0),1e-12)
    coef,_,rank,_=lstsq(A/scale,b,cond=1e-12,lapack_driver='gelsd')
    assert rank==X.shape[1]
    return coef/scale


def objective(X,y,coef):
    return float(np.sum((X@coef-y)**2)+np.sum(coef[1:]**2))


def main():
    assert json.loads((O/'generation_identity.json').read_text())['cases']==960
    df=pd.read_csv(O/'all_cases.csv')
    assert len(df)==4800 and df.case.nunique()==960
    rng=np.random.default_rng(2026110319)
    cases=[]
    for sparse in [False,True]:
        choices=sorted(df.loc[(df.truth=='sparse')==sparse,'case'].unique())
        cases.extend(rng.choice(choices,size=10,replace=False).tolist())
    d.save(O/'independent_verification_protocol.json',dict(seed=2026110319,cases=cases,selection='Ten sparse and ten non-sparse cases chosen without using scores; numerical verification, not new scientific heldout evaluation.',implementation='Compare pivoted QR saved predictions with independently recomputed SVD augmented ridge; reconstruct all five actions and score against original supplied prior.',source_sha256=d.sha(__file__)))
    score_errors=[];prediction_errors=[];objective_errors=[];gain_errors=[]
    for case in cases:
        panel=case.split('/')[0]
        paths,A,_,_,_=study.base.load(panel,'train');X=study.structure.features(paths)['span_origin_destination']
        file=d.WORK/'shared_noise_qr_retry/task4_crossmonth_shared_noise_cases'/(hashlib.sha256(case.encode()).hexdigest()[:16]+'.npz')
        z=np.load(file);rows=df[df.case==case].set_index('method')
        assert set(rows.artifact_sha256)=={d.sha(file)}
        priors=z['artificial_prior'];truth=z['artificial_truth'][2];b=priors[2];fold=z['fold'];scales=z['target_training_scales']
        source=np.mean([v/max(float(np.median(v[v>0])),1e-9) for v in priors[:2]],axis=0)
        source_qr=qr.ridge_qr(X,source);source_svd=svd_ridge(X,source)
        pair_qr=np.column_stack([X,source-X@source_qr]);pair_svd=np.column_stack([X,source-X@source_svd])
        saved={'geometry':np.zeros(len(b)),'pair':np.zeros(len(b))};independent={'geometry':np.zeros(len(b)),'pair':np.zeros(len(b))}
        for k in range(5):
            train=fold!=k;test=~train;y=b[train]/scales[k]
            for name,design,alternate in [('geometry',X,X),('pair',pair_qr,pair_svd)]:
                coefficient=z[name+'_coef'][k];svd=svd_ridge(alternate[train],y)
                saved[name][test]=np.maximum(design[test]@coefficient,0)*scales[k]
                independent[name][test]=np.maximum(alternate[test]@svd,0)*scales[k]
                # Compare each solver in its own design; source residuals are independently reconstructed.
                reference=svd_ridge(design[train],y)
                objective_errors.append(abs(objective(design[train],y,coefficient)-objective(design[train],y,reference))/(1+objective(design[train],y,reference)))
        for name in saved:
            prediction_errors.append(float(np.max(np.abs(saved[name]-independent[name]))/(1+np.max(b))))
        gmse=float(np.mean((saved['geometry']-b)**2));pmse=float(np.mean((saved['pair']-b)**2))
        gain_errors.append(abs((1-pmse/max(gmse,1e-12))-float(rows.pair_prior_prediction_MSE_reduction.iloc[0])))
        dest=paths.destination_zone.str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy();c=A@truth
        for name,(kind,alpha) in study.METHODS.items():
            prior=b if kind=='raw' else (1-alpha)*b+alpha*saved[kind]
            prediction,status=study.base.ridge(A,c,prior);assert status['success']
            scores=study.base.metrics(prediction,truth,b,A,c,dest)
            score_errors.extend(abs(float(rows.loc[name,key])-value) for key,value in scores.items())
    assert max(prediction_errors)<1e-8
    assert max(objective_errors)<1e-8
    assert max(score_errors)<1e-10 and max(gain_errors)<1e-10
    result=dict(cases=20,actions_recomputed=100,independent_solver='Column-equilibrated augmented SVD (gelsd), versus pivoted QR (gelsy)',maximum_relative_physical_prediction_difference=max(prediction_errors),maximum_relative_ridge_objective_difference=max(objective_errors),maximum_score_readback_error=max(score_errors),maximum_prior_prediction_gain_error=max(gain_errors),source_sha256=d.sha(__file__),scope='Numerical verification only, no hidden OD claim.')
    d.save(O/'independent_numerical_verification.json',result)
    print(result)


if __name__=='__main__':main()
