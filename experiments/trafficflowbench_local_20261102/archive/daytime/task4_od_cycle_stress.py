"""Fresh exact-pair signal test with feasible OD-only cycle refinements."""
import common as d
import task4_count_source_pair_stress as generator
import task4_exact_pair_audit as exact
import od_cycle_refinement as cycle
import numpy as np
import pandas as pd
import hashlib
import msvcrt
import traceback
from datetime import datetime,timezone

O=d.OUT/'task4_od_cycle_stress';SEED=2026110325


def forecasts(X,Q,priors,fold):
    source=np.mean([b/max(float(np.median(b[b>0])),1e-9) for b in priors[:2]],axis=0);z=source-Q@(Q.T@source)
    designs={'geometry':X,'pair':np.column_stack([X,z])};b=priors[2];predictions={name:np.zeros(len(b)) for name in designs};coefs={name:[] for name in designs};scales=[]
    for k in range(5):
        train=fold!=k;test=~train;scale=max(float(np.median(b[train&(b>0)])),1e-9);scales.append(scale)
        for name,design in designs.items():
            coef=generator.qr.ridge_qr(design[train],b[train]/scale);predictions[name][test]=np.maximum(design[test]@coef,0)*scale;coefs[name].append(coef)
    return predictions,coefs,np.array(scales)


def main():
    O.mkdir(parents=True,exist_ok=True)
    d.save(O/'protocol.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),seed=SEED,case_count=480,replicates=1,truths=generator.family.TRUTHS,noises=generator.family.NOISES,drifts=generator.family.DRIFTS,methods=['baseline','geometry005','direct005','direct025','cycle005','cycle025'],hypothesis='Exactly geometry-orthogonal cross-input signal predicts public priors. Test whether feasible OD cycle updates can retain its matching benefit while keeping all three non-OD metrics at baseline, instead of following marginal recalibration.',source_feature='Mean normalized source prior projected exactly out of span+origin/destination geometry; target cross-fit ridge1 with one common five-fold assignment.',cycle='Sample 4*path_count fixed random valid four-zone rectangles per case, shared between strengths. Cycle(+1,-1,-1,+1) preserves all origin/destination margins. Require original prior-sign derivative0 and retain original orthant/nonnegative bounds, preserving prior L1 deviation. Each accepted step decreases L1 distance to the estimated direct candidate; no latent truth enters direction, bound or acceptance.',target='Direct nonnegative lambda0.05 predictions at fixed5% and25% prior mixture. Cycle starts at original baseline and uses the same cycle sequence for both strengths.',guarantee='Release-specific count rows are marginal combinations, so feasible cycles preserve counts, S_link/S_attr/S_dev within numerical precision. S_od can still worsen. Heuristic coordinate updates are not a proven global projection or optimum.',numerical_controls='Separate constructive nonzero rectangle and 20 oracle L1 monotonicity controls must pass before use. Oracle truth is confined to those numerical checks, never supplied to this estimator.',evaluation='Report all known-latent clipped scores AND unclipped true relative OD L1, all zero movement, accepted updates and constraint drift. Fresh artificial draws in adaptively chosen family; not whole-system blind or official validation.',source_sha256=d.sha(__file__),cycle_source_sha256=d.sha(cycle.__file__),generator_sha256=d.sha(generator.__file__),exact_projection_sha256=d.sha(exact.__file__),scope='No real candidate or Kaggle upload; frozen official baseline retained.'))
    rows,hashes=[],{}
    for panel in d.night.predictor.load_routes():
        paths,A,real_c,_,inputs=generator.family.base.load(panel,'train');hashes.update(inputs)
        origin=paths.origin_zone.str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy();dest=paths.destination_zone.str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy();lookup=cycle.path_lookup(origin,dest)
        X=generator.family.structure.features(paths)['span_origin_destination'];Q,_=exact.basis(X);level=float(np.median(real_c[real_c>0]))
        for truth_name in generator.family.TRUTHS:
            for noise in generator.family.NOISES:
                for drift in generator.family.DRIFTS:
                    case=f'{panel}/{truth_name}/{noise}/{drift}/0';seed=int.from_bytes(hashlib.sha256((str(SEED)+'/'+case).encode()).digest()[:8],'little')%(2**63);rng=np.random.default_rng(seed)
                    truth,priors,counts,_=generator.draw(paths,A,level,truth_name,noise,drift,rng)
                    order=rng.permutation(len(paths));fold=np.empty(len(paths),int);fold[order]=np.arange(len(paths))%5
                    predictions,coefs,scales=forecasts(X,Q,priors,fold)
                    b,c=priors[2],counts[2];base,st=generator.family.base.ridge(A,c,b);assert st['success']
                    methods={'baseline':base};diagnostics={}
                    geometry,status=generator.family.base.ridge(A,c,.95*b+.05*predictions['geometry']);assert status['success'];methods['geometry005']=geometry
                    for name,alpha in [('005',.05),('025',.25)]:
                        direct,status=generator.family.base.ridge(A,c,(1-alpha)*b+alpha*predictions['pair']);assert status['success'];methods['direct'+name]=direct
                    folder=d.WORK/'task4_od_cycle_cases'/hashlib.sha256(case.encode()).hexdigest()[:16];folder.mkdir(parents=True,exist_ok=True)
                    inputs_file=folder/'inputs.npz';np.savez_compressed(inputs_file,artificial_truth=truth,artificial_prior=priors,observed_counts=counts,fold=fold,target_training_scales=scales,seed=seed,**{name+'_coef':np.array(value) for name,value in coefs.items()});input_digest=d.sha(inputs_file)
                    rectangles=cycle.draw_cycles(lookup,4*len(paths),rng)
                    # Validate every sampled direction against released incidence, using exact integer sums.
                    assert np.max(np.abs(A[:,rectangles[:,0]]-A[:,rectangles[:,1]]-A[:,rectangles[:,2]]+A[:,rectangles[:,3]]))==0
                    d.state(status='running',stage='task4_od_cycle_stress',case=case,next='Feasible matching-only updates; preserve non-OD metrics, score unclipped OD error too, no hidden truth used by learner.')
                    for name in ['005','025']:
                        methods['cycle'+name],diagnostics['cycle'+name]=cycle.refine_cycles(base,methods['direct'+name],b,rectangles)
                    predictions_file=folder/'predictions.npz';np.savez_compressed(predictions_file,**methods);prediction_digest=d.sha(predictions_file)
                    reference=generator.family.base.metrics(base,truth[2],b,A,c,dest)
                    for name,prediction in methods.items():
                        metrics=generator.family.base.metrics(prediction,truth[2],b,A,c,dest)
                        row=dict(case=case,panel=panel,truth=truth_name,noise=noise,drift=drift,seed=seed,method=name,input_artifact_sha256=input_digest,prediction_artifact_sha256=prediction_digest,true_relative_OD_L1=float(np.abs(prediction-truth[2]).sum()/max(truth[2].sum(),1e-12)),**metrics,**diagnostics.get(name,{}))
                        if name.startswith('cycle'):
                            count_drift=float(np.max(np.abs(A@(prediction-base))));origin_drift=float(np.max(np.abs(np.bincount(origin,weights=prediction-base))));dest_drift=float(np.max(np.abs(np.bincount(dest,weights=prediction-base))));prior_drift=abs(float(np.abs(prediction-b).sum()-np.abs(base-b).sum()));metric_drift=max(abs(metrics[k]-reference[k]) for k in ['synthetic_S_link','synthetic_S_dev','synthetic_S_attr'])
                            assert max(count_drift,origin_drift,dest_drift,prior_drift)<1e-7 and metric_drift<1e-9
                            row.update(max_constraint_drift=max(count_drift,origin_drift,dest_drift,prior_drift),max_non_OD_metric_drift=metric_drift)
                        rows.append(row)
        pd.DataFrame(rows).to_csv(O/'all_cases.csv',index=False)
        d.state(status='running',stage='task4_od_cycle_stress',panel=panel,completed_cases=len(rows)//6,next='Preserve all cycle moves and zero movement, compare actual OD risks with identical direct candidates; no official score claim.')
    df=pd.DataFrame(rows);d.save(O/'input_sha256.json',hashes);baseline=df[df.method=='baseline'].set_index('case');risk=[]
    for name in methods:
        group=df[df.method==name].set_index('case');delta=group.synthetic_S_ODME-baseline.synthetic_S_ODME;od=group.synthetic_S_od-baseline.synthetic_S_od;raw=group.true_relative_OD_L1-baseline.true_relative_OD_L1
        risk.append(dict(method=name,cases=len(group),mean_score_delta=float(delta.mean()),worst_score_delta=float(delta.min()),score_harmed_cases=int((delta < -1e-12).sum()),mean_OD_delta=float(od.mean()),OD_harmed_cases=int((od < -1e-12).sum()),mean_unclipped_OD_error_delta=float(raw.mean()),unclipped_OD_error_harmed_cases=int((raw>1e-12).sum()),zero_OD_score_cases=int((group.synthetic_S_od==0).sum()),moving_cases=int((group.relative_L1_movement.fillna(0)>1e-8).sum()) if 'relative_L1_movement' in group else None))
    pd.DataFrame(risk).to_csv(O/'paired_risk.csv',index=False)
    df.groupby(['noise','drift','method'])[['synthetic_S_ODME','synthetic_S_od','true_relative_OD_L1']].mean().to_csv(O/'by_assumption.csv');assert df.case.nunique()==480
    d.state(status='stage_complete',stage='task4_od_cycle_stress_complete',next='Audit actual OD effects, non-OD invariants, movement and score-floor masking; independently reproduce before publishing. No real candidate, official baseline unchanged.')


if __name__=='__main__':
    lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:main()
    except Exception as error:
        d.state(status='failed',stage='task4_od_cycle_stress',error=repr(error),next='Preserve fixed cycles, inputs, seeds and failed case; repair invariant/numerical implementation without posthoc selection.');traceback.print_exc();raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
