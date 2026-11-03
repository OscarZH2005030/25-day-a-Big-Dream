"""Fresh synthetic test of prior-prediction gains versus actual latent OD risk."""
import common as d
import task4_prior_structure as structure
import task4_span_shrink_stress as base
import numpy as np
import pandas as pd
import hashlib
import msvcrt
import traceback
from scipy.linalg import solve
from datetime import datetime, timezone

O=d.OUT/'task4_crossmonth_shared_noise_stress'
SEED=2026110317
TRUTHS=['short','origin_destination','sparse','piecewise_span']
NOISES=['independent_lognormal','shared_lognormal','independent_positive_background','shared_positive_background']
DRIFTS=['stable_pair','moderate_drift','replaced_pair']
METHODS={'baseline':('raw',0.),'geometry005':('geometry',.05),'pair005':('pair',.05),'pair025':('pair',.25),'pair100':('pair',1.)}


def forecasts(X, priors, rng):
    source=np.mean([v/max(float(np.median(v[v>0])),1e-9) for v in priors[:2]],axis=0)
    reg=np.eye(X.shape[1]);reg[0,0]=0
    residual=source-X@solve(X.T@X+reg,X.T@source,assume_a='pos')
    pair=np.column_stack([X,residual]);b=priors[2]
    order=rng.permutation(len(b));fold=np.empty(len(b),int);fold[order]=np.arange(len(b))%5
    predictions={'geometry':np.zeros(len(b)),'pair':np.zeros(len(b))};coefs={'geometry':[],'pair':[]};scales=[]
    for k in range(5):
        train=fold!=k;test=~train;scale=max(float(np.median(b[train&(b>0)])),1e-9);scales.append(scale)
        for name,design in [('geometry',X),('pair',pair)]:
            penalty=np.eye(design.shape[1]);penalty[0,0]=0
            coef=solve(design[train].T@design[train]+penalty,design[train].T@(b[train]/scale),assume_a='pos')
            predictions[name][test]=np.maximum(design[test]@coef,0)*scale
            coefs[name].append(coef)
    geometry_mse=float(np.mean((predictions['geometry']-b)**2));pair_mse=float(np.mean((predictions['pair']-b)**2))
    return predictions,coefs,np.array(scales),fold,1-pair_mse/max(geometry_mse,1e-12)


def main():
    O.mkdir(parents=True,exist_ok=True)
    protocol=dict(registered_utc=datetime.now(timezone.utc).isoformat(),seed=SEED,case_count=960,panels=10,truths=TRUTHS,noises=NOISES,drifts=DRIFTS,replicates=2,methods=METHODS,hypothesis='Held-out weak-prior predictability across months can reflect true common OD affinity or shared observation/prior noise; determine whether pair-based calibration risks erasing changing real demand while retaining common artifacts.',months=3,target_month=2,drift='Monthly scales0.85/1/1.15. Stable: same base latent OD; moderate: independent mean-one lognormal path perturbation sd0.35; replaced: new independent latent draw of same geometry family each month.',noise='Independent lognormal sd0.6; shared lognormal path error sd0.6 plus independent sd0.2 (mean-one factors); positive lognormal background sd0.7 amplitude0.8 times truth mean, either independent each month or shared path pattern plus independent amplitude0.1.',features='Span+origin/destination ridge1. Source residual from other two priors only; target prior predictions cross-fit over five fixed case-seeded OD folds. No latent truth enters forecasts or strength selection.',calibration='Freeze all alpha0/0.05/0.25/1 before draws; denoised prior=(1-alpha)*target_prior+alpha*crossfit prediction; all use baseline nonnegative lambda0.05 count solver. Geometry005 is an explicit non-pair control.',selection='None. Public-network shapes repeated but new artificial draws. Prediction MSE of supplied prior and known-latent complete OD components reported separately, including all harms and solver failures.',source_sha256=d.sha(__file__),feature_source_sha256=d.sha(d.WORK/'task4_prior_structure.py'),solver_source_sha256=d.sha(d.WORK/'task4_span_shrink_stress.py'),scope='Artificial assumption stress, not hidden OD validation. Official frozen reference unchanged, no actual candidate export or Kaggle upload.')
    d.save(O/'protocol.json',protocol)
    rows,hashes=[],{}
    for panel in d.night.predictor.load_routes():
        paths,A,real_c,_,input_hashes=base.load(panel,'train');hashes.update(input_hashes)
        X=structure.features(paths)['span_origin_destination'];level=float(np.median(real_c[real_c>0]))
        for truth_name in TRUTHS:
            for noise in NOISES:
                for drift in DRIFTS:
                    for rep in range(2):
                        case=f'{panel}/{truth_name}/{noise}/{drift}/{rep}'
                        seed=int.from_bytes(hashlib.sha256((str(SEED)+'/'+case).encode()).digest()[:8],'little')%(2**63);rng=np.random.default_rng(seed)
                        common,origin,dest=base.truth_draw(paths,A,level,truth_name,rng)
                        truth=[]
                        for month,multiplier in enumerate([.85,1.,1.15]):
                            if drift=='replaced_pair':
                                f,_,_=base.truth_draw(paths,A,level,truth_name,rng)
                            elif drift=='moderate_drift':f=common*rng.lognormal(-.35**2/2,.35,len(common))
                            else:f=common.copy()
                            truth.append(f*multiplier)
                        shared_log=rng.normal(-.6**2/2,.6,len(common));shared_background=rng.lognormal(-.7**2/2,.7,len(common))
                        priors=[]
                        for f in truth:
                            if noise=='independent_lognormal':b=f*rng.lognormal(-.6**2/2,.6,len(f))
                            elif noise=='shared_lognormal':b=f*np.exp(shared_log+rng.normal(-.2**2/2,.2,len(f)))
                            elif noise=='independent_positive_background':b=f+.8*f.mean()*rng.lognormal(-.7**2/2,.7,len(f))
                            else:b=f+f.mean()*(.8*shared_background+.1*rng.lognormal(-.7**2/2,.7,len(f)))
                            # Tiny floor is explicit for zero-flow artificial paths.
                            priors.append(np.maximum(b,1e-6))
                        predictions,coefs,scales,fold,predictive_gain=forecasts(X,priors,rng)
                        target_truth,b=truth[2],priors[2];c=A@target_truth
                        folder=d.WORK/'task4_crossmonth_shared_noise_cases';folder.mkdir(parents=True,exist_ok=True);artifact=folder/(hashlib.sha256(case.encode()).hexdigest()[:16]+'.npz')
                        np.savez_compressed(artifact,artificial_truth=np.array(truth),artificial_prior=np.array(priors),fold=fold,geometry_coef=np.array(coefs['geometry']),pair_coef=np.array(coefs['pair']),target_training_scales=scales,seed=seed)
                        digest=d.sha(artifact)
                        for name,(kind,alpha) in METHODS.items():
                            prior=b if kind=='raw' else (1-alpha)*b+alpha*predictions[kind]
                            prediction,status=base.ridge(A,c,prior)
                            row=dict(case=case,panel=panel,truth=truth_name,noise=noise,drift=drift,replicate=rep,seed=seed,method=name,solver_success=status['success'],pair_prior_prediction_MSE_reduction=predictive_gain,artifact_sha256=digest)
                            # Score deviation against ORIGINAL supplied prior, not the modified prior.
                            if status['success']:row.update(base.metrics(prediction,target_truth,b,A,c,dest))
                            rows.append(row)
        pd.DataFrame(rows).to_csv(O/'all_cases.csv',index=False)
        d.state(status='running',stage='task4_crossmonth_shared_noise_stress',panel=panel,completed_cases=len(rows)//len(METHODS),next='Compare cross-month prior prediction with true artificial OD risk under shared noise and changing demand; all fixed strengths, no actual submission.')
    df=pd.DataFrame(rows);d.save(O/'input_sha256.json',hashes)
    df.groupby(['truth','noise','drift','method'])[[c for c in df if c.startswith('synthetic_')]+['pair_prior_prediction_MSE_reduction']].mean().to_csv(O/'by_assumption.csv')
    baseline=df[df.method=='baseline'].set_index('case');paired=[]
    for name in METHODS:
        group=df[df.method==name].set_index('case');good=group.solver_success&baseline.solver_success
        delta=group.loc[good,'synthetic_S_ODME']-baseline.loc[good,'synthetic_S_ODME'];od=group.loc[good,'synthetic_S_od']-baseline.loc[good,'synthetic_S_od']
        paired.append(dict(method=name,paired_cases=len(delta),mean_score_delta=float(delta.mean()),worst_score_delta=float(delta.min()),score_harmed_cases=int((delta < -1e-12).sum()),mean_OD_delta=float(od.mean()),OD_harmed_cases=int((od < -1e-12).sum()),solver_failed_cases=int((~group.solver_success).sum())))
    pd.DataFrame(paired).to_csv(O/'paired_risk.csv',index=False)
    assert df.case.nunique()==protocol['case_count']
    d.state(status='stage_complete',stage='task4_crossmonth_shared_noise_stress_complete',next='Check whether prior-prediction gains coexist with actual OD harms, retain all noise/drift failures, then independently reproduce. No official-score or hidden-OD claim.')


if __name__=='__main__':
    lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:main()
    except Exception as error:
        d.state(status='failed',stage='task4_crossmonth_shared_noise_stress',error=repr(error),next='Preserve registered cases, folds and failures; no posthoc strength or road selection.');traceback.print_exc();raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
