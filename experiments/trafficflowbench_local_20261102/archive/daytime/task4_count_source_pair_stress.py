"""Fresh paired test: raw cross-month priors versus source-count-adjusted features."""
import common as d
import task4_crossmonth_shared_noise_stress as family
import task4_crossmonth_shared_noise_qr as qr
import numpy as np
import pandas as pd
import hashlib
import msvcrt
import traceback
from datetime import datetime,timezone

O=d.OUT/'task4_count_source_pair_stress'
SEED=2026110321
METHODS={'baseline':('raw',0.),'geometry005':('geometry',.05),'raw_pair005':('raw_pair',.05),'raw_pair025':('raw_pair',.25),'count_pair005':('count_pair',.05),'count_pair025':('count_pair',.25)}


def draw(paths,A,level,truth_name,noise,drift,rng):
    common,origin,dest=family.base.truth_draw(paths,A,level,truth_name,rng)
    truth=[]
    for multiplier in [.85,1.,1.15]:
        if drift=='replaced_pair':f,_,_=family.base.truth_draw(paths,A,level,truth_name,rng)
        elif drift=='moderate_drift':f=common*rng.lognormal(-.35**2/2,.35,len(common))
        else:f=common.copy()
        truth.append(f*multiplier)
    shared_log=rng.normal(-.6**2/2,.6,len(common));shared_background=rng.lognormal(-.7**2/2,.7,len(common));priors=[]
    for f in truth:
        if noise=='independent_lognormal':b=f*rng.lognormal(-.6**2/2,.6,len(f))
        elif noise=='shared_lognormal':b=f*np.exp(shared_log+rng.normal(-.2**2/2,.2,len(f)))
        elif noise=='independent_positive_background':b=f+.8*f.mean()*rng.lognormal(-.7**2/2,.7,len(f))
        else:b=f+f.mean()*(.8*shared_background+.1*rng.lognormal(-.7**2/2,.7,len(f)))
        priors.append(np.maximum(b,1e-6))
    # Observed link counts are generated here; the learner below never receives latent truth.
    return np.array(truth),np.array(priors),np.array([A@f for f in truth]),dest


def crossfit(X,A,counts,priors,fold):
    adjusted=[];source_ok=True
    for month in [0,1]:
        estimate,status=family.base.ridge(A,counts[month],priors[month]);source_ok &= bool(status['success']);adjusted.append(estimate)
    designs={'geometry':X}
    for kind,sources in [('raw_pair',priors[:2]),('count_pair',adjusted)]:
        if kind=='count_pair' and not source_ok:continue
        source=np.mean([v/max(float(np.median(v[v>0])),1e-9) for v in sources],axis=0)
        residual=source-X@qr.ridge_qr(X,source);designs[kind]=np.column_stack([X,residual])
    b=priors[2];predictions={name:np.zeros(len(b)) for name in designs};coefs={name:[] for name in designs};scales=[]
    for k in range(5):
        train=fold!=k;test=~train;scale=max(float(np.median(b[train&(b>0)])),1e-9);scales.append(scale)
        for name,design in designs.items():
            coef=qr.ridge_qr(design[train],b[train]/scale);coefs[name].append(coef);predictions[name][test]=np.maximum(design[test]@coef,0)*scale
    reference=float(np.mean((predictions['geometry']-b)**2))
    gains={name:1-float(np.mean((prediction-b)**2))/max(reference,1e-12) for name,prediction in predictions.items()}
    return predictions,coefs,np.array(scales),np.array(adjusted),source_ok,gains


def main():
    O.mkdir(parents=True,exist_ok=True)
    protocol=dict(registered_utc=datetime.now(timezone.utc).isoformat(),seed=SEED,case_count=480,replicates=1,truths=family.TRUTHS,noises=family.NOISES,drifts=family.DRIFTS,methods=METHODS,hypothesis='Shared source-prior artifacts can predict target priors without helping real OD. Source-month counts may remove some observable shared error before extracting pair affinity; compare raw and count-adjusted sources with identical target folds.',source_control='Run baseline nonnegative lambda0.05 solver separately on each source prior and its OWN generated observed link counts; normalize and residualize estimates with span+origin/destination ridge1. Never impose equality of flows or counts across months.',observation_boundary='Generator returns latent truth separately. Learner crossfit receives ONLY geometry, A, observed monthly counts, supplied priors and fixed fold. Source latent paths are never features.',fitting='Same five target OD-entry folds and training-only target median for both source variants; augmented column-equilibrated QR ridge1, no hidden truth or model selection.',scope='Fresh artificial seed in an adaptively chosen existing stress family; not system blind validation or real hidden OD scoring. No PM/ramp observations or unverified real-data constraints added. Official reference frozen; no Kaggle upload.',source_sha256=d.sha(__file__),generator_family_sha256=d.sha(family.__file__),QR_implementation_sha256=d.sha(qr.__file__))
    d.save(O/'protocol.json',protocol)
    rows,hashes=[],{}
    for panel in d.night.predictor.load_routes():
        paths,A,real_c,_,inputs=family.base.load(panel,'train');hashes.update(inputs)
        X=family.structure.features(paths)['span_origin_destination'];level=float(np.median(real_c[real_c>0]))
        for truth_name in family.TRUTHS:
            for noise in family.NOISES:
                for drift in family.DRIFTS:
                    case=f'{panel}/{truth_name}/{noise}/{drift}/0'
                    seed=int.from_bytes(hashlib.sha256((str(SEED)+'/'+case).encode()).digest()[:8],'little')%(2**63);rng=np.random.default_rng(seed)
                    truth,priors,counts,dest=draw(paths,A,level,truth_name,noise,drift,rng)
                    order=rng.permutation(len(paths));fold=np.empty(len(paths),int);fold[order]=np.arange(len(paths))%5
                    predictions,coefs,scales,source_flows,source_ok,gains=crossfit(X,A,counts,priors,fold)
                    folder=d.WORK/'task4_count_source_pair_cases';folder.mkdir(parents=True,exist_ok=True);artifact=folder/(hashlib.sha256(case.encode()).hexdigest()[:16]+'.npz')
                    np.savez_compressed(artifact,artificial_truth=truth,artificial_prior=priors,observed_counts=counts,source_count_estimates=source_flows,fold=fold,target_training_scales=scales,seed=seed,**{name+'_coef':np.array(value) for name,value in coefs.items()});digest=d.sha(artifact)
                    for name,(kind,alpha) in METHODS.items():
                        available=kind=='raw' or kind in predictions
                        row=dict(case=case,panel=panel,truth=truth_name,noise=noise,drift=drift,seed=seed,method=name,source_solver_success=source_ok,solver_success=False,raw_pair_prior_MSE_reduction=gains['raw_pair'],count_pair_prior_MSE_reduction=gains.get('count_pair'),artifact_sha256=digest)
                        if available:
                            prior=priors[2] if kind=='raw' else (1-alpha)*priors[2]+alpha*predictions[kind]
                            prediction,status=family.base.ridge(A,counts[2],prior);row['solver_success']=status['success']
                            if status['success']:row.update(family.base.metrics(prediction,truth[2],priors[2],A,counts[2],dest))
                        rows.append(row)
        pd.DataFrame(rows).to_csv(O/'all_cases.csv',index=False)
        d.state(status='running',stage='task4_count_source_pair_stress',panel=panel,completed_cases=len(rows)//len(METHODS),next='Same-fold fresh comparison of raw and source-count-adjusted affinities under shared artifacts/drift; retain failures and original-prior scoring, no real candidate export.')
    df=pd.DataFrame(rows);d.save(O/'input_sha256.json',hashes)
    baseline=df[df.method=='baseline'].set_index('case');risk=[]
    for name in METHODS:
        group=df[df.method==name].set_index('case');good=group.solver_success&baseline.solver_success
        delta=group.loc[good,'synthetic_S_ODME']-baseline.loc[good,'synthetic_S_ODME'];od=group.loc[good,'synthetic_S_od']-baseline.loc[good,'synthetic_S_od']
        risk.append(dict(method=name,paired_cases=len(delta),mean_score_delta=float(delta.mean()),worst_score_delta=float(delta.min()),score_harmed_cases=int((delta < -1e-12).sum()),mean_OD_delta=float(od.mean()),OD_harmed_cases=int((od < -1e-12).sum()),solver_failed_cases=int((~group.solver_success).sum())))
    pd.DataFrame(risk).to_csv(O/'paired_risk.csv',index=False)
    df.groupby(['noise','drift','method'])[[col for col in df if col.startswith('synthetic_')]+['raw_pair_prior_MSE_reduction','count_pair_prior_MSE_reduction']].mean().to_csv(O/'by_assumption.csv')
    assert df.case.nunique()==480
    d.state(status='stage_complete',stage='task4_count_source_pair_stress_complete',next='Compare source-control OD effects on identical cases, paired availability and all counterexamples; independently verify before publishing. No hidden-OD improvement or Kaggle upload.')


if __name__=='__main__':
    lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:main()
    except Exception as error:
        d.state(status='failed',stage='task4_count_source_pair_stress',error=repr(error),next='Preserve fixed seed, same-fold controls and all failures; repair numerical issue without posthoc selection.');traceback.print_exc();raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
