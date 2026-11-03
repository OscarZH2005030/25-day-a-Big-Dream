"""Use an uncertainty-based OD direction while preserving all three other metrics."""
import common as d
import task4_smallnet_posterior_stress as small
import task4_weighted_variance_stress as weighted
import numpy as np
import pandas as pd
import json
import hashlib
import msvcrt
import traceback
from datetime import datetime, timezone

O = d.OUT / 'task4_od_only_direction_stress'
SEED = 2026110313


def safeguard(f0, target, b, A, origin, dest):
    tolerance = 1e-10 * max(1., np.max(f0), np.max(b))
    free = (f0 > tolerance) & (np.abs(f0 - b) > tolerance)
    direction = np.zeros(len(b))
    if free.any():
        C = np.vstack([A, np.eye(origin.max()+1)[origin].T, np.eye(dest.max()+1)[dest].T, np.sign(f0-b)])[:, free]
        desired = (target-f0)[free]
        # Orthogonal projection onto the equality-constraint nullspace.
        direction[free] = desired - C.T @ np.linalg.lstsq(C.T, desired, rcond=1e-11)[0]
    sign = np.sign(f0-b)
    limits = [1.]
    negative = direction < -tolerance
    crosses = sign * direction < -tolerance
    if negative.any():
        limits.append(float(np.min(f0[negative] / -direction[negative])))
    if crosses.any():
        limits.append(float(np.min(np.abs(f0-b)[crosses] / np.abs(direction[crosses]))))
    step = max(0., min(limits))
    if step < 1.:
        step *= 1-1e-8
    result = f0 + step * direction
    equality_drift = max(float(np.max(np.abs(A@(result-f0)))), float(np.max(np.abs(np.bincount(dest, weights=result-f0)))), float(np.max(np.abs(np.bincount(origin, weights=result-f0)))), abs(float(np.abs(result-b).sum()-np.abs(f0-b).sum())))
    assert result.min() >= -1e-9 and equality_drift < 1e-8
    return result, dict(step=step, equality_max_abs_drift=equality_drift, relative_L1_movement=float(np.abs(result-f0).sum()/max(f0.sum(),1e-12)), movable_coordinates=int(free.sum()))


def main():
    O.mkdir(parents=True, exist_ok=True)
    model = d.OUT / 'task4_variance_controls_audit/full_train_models.json'
    eta = json.loads(model.read_text())['models']['extended_span']['parameter']
    protocol = dict(registered_utc=datetime.now(timezone.utc).isoformat(), seed=SEED, case_count=192, nodes=[6,9], scenarios=small.SCENARIOS, true_noise_factors=[.5,1.,2.], replicates=8, methods=['baseline','weighted010','weighted100','od_only010','od_only100'], hypothesis='Span uncertainty corrections may harm non-OD components; project their direction onto the count/margin/prior-distance invariant set to isolate actual OD-pairing benefit and risk.', algorithm='Freeze zero or prior-sign-boundary coordinates; orthogonally project target-minus-baseline direction onto nullspace of A, origin sums, destination sums and sign(f_base-b); truncate one step to retain nonnegativity and the baseline prior orthant. No truth enters direction or step.', guarantee='Within numerical precision, predicted counts, all origin/destination margins and prior L1 deviation equal baseline. Thus S_link/S_attr/S_dev are unchanged for every hidden reference; S_od may improve OR worsen.', eta_from_public_train=eta, generator='Reuse four explicitly artificial mechanisms from previous smallnet study with a new fixed seed. No MCMC here, no posterior claim. Prior studies informed this algorithm, so these are new artificial draws, not whole-system blind validation.', source_sha256=d.sha(__file__), generator_sha256=d.sha(d.WORK/'task4_smallnet_posterior_stress.py'), weighted_solver_sha256=d.sha(d.WORK/'task4_weighted_variance_stress.py'), model_sha256=d.sha(model), scope='Preserving other metrics cannot establish correct OD pairing. All outcomes retained; official Task4 reference unchanged, no Kaggle uploads.')
    d.save(O/'protocol.json', protocol)
    rows = []
    for nodes in protocol['nodes']:
        A, _, _, origin, dest = small.geometry(nodes)
        span = dest-origin
        weight = np.exp(eta*(span-span.mean())/span.max())
        weight *= np.sum(A*A)/np.sum(A*A*weight)
        for scenario in protocol['scenarios']:
            for factor in protocol['true_noise_factors']:
                for rep in range(protocol['replicates']):
                    case = f'{nodes}/{scenario}/{factor}/{rep}'
                    seed = int.from_bytes(hashlib.sha256((str(SEED)+'/'+case).encode()).digest()[:8], 'little') % (2**63)
                    truth, b, _ = small.latent_case(origin, dest, scenario, factor, np.random.default_rng(seed))
                    c = A@truth
                    f0, status = small.base.ridge(A,c,b)
                    assert status['success']
                    decisions = {'baseline': (f0, {})}
                    for name, alpha in [('010',.1),('100',1.)]:
                        target, success = weighted.ridge(A,c,b,(1-alpha)+alpha*weight)
                        assert success
                        decisions['weighted'+name] = target, {}
                        decisions['od_only'+name] = safeguard(f0,target,b,A,origin,dest)
                    reference = small.base.metrics(f0,truth,b,A,c,dest)
                    for name, (prediction, diagnostics) in decisions.items():
                        metrics = small.base.metrics(prediction,truth,b,A,c,dest)
                        if name.startswith('od_only'):
                            assert max(abs(metrics[k]-reference[k]) for k in ['synthetic_S_link','synthetic_S_dev','synthetic_S_attr']) < 1e-9
                        rows.append(dict(case=case,nodes=nodes,scenario=scenario,true_noise_factor=factor,replicate=rep,seed=seed,method=name,score_delta=metrics['synthetic_S_ODME']-reference['synthetic_S_ODME'],OD_delta=metrics['synthetic_S_od']-reference['synthetic_S_od'],**metrics,**diagnostics))
        pd.DataFrame(rows).to_csv(O/'all_cases.csv',index=False)
        d.state(status='running',stage='task4_od_only_direction_stress',nodes=nodes,completed_cases=len(rows)//5,next='Isolate OD effect while preserving other three metrics; keep all null movements and harmful OD cases.')
    df = pd.DataFrame(rows)
    df.groupby(['scenario','true_noise_factor','method'])[['score_delta','OD_delta']].mean().to_csv(O/'by_assumption.csv')
    risk = []
    for name, group in df.groupby('method'):
        risk.append(dict(method=name,cases=len(group),mean_score_delta=float(group.score_delta.mean()),worst_score_delta=float(group.score_delta.min()),score_harmed_cases=int((group.score_delta < -1e-12).sum()),mean_OD_delta=float(group.OD_delta.mean()),OD_harmed_cases=int((group.OD_delta < -1e-12).sum()),moving_cases=int((group.relative_L1_movement.fillna(0)>1e-8).sum()),max_constraint_drift=float(group.equality_max_abs_drift.fillna(0).max())))
    pd.DataFrame(risk).to_csv(O/'paired_risk.csv',index=False)
    assert df.case.nunique() == 192
    d.state(status='stage_complete',stage='task4_od_only_direction_stress_complete',next='Review all OD-only benefit/harm and degeneracy, verify KKT and independent implementation before real use. No official OD gain or Kaggle upload.')


if __name__ == '__main__':
    lock = (d.ROOT/'work/overnight/research.lock').open('r+b')
    lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:
        main()
    except Exception as error:
        d.state(status='failed',stage='task4_od_only_direction_stress',error=repr(error),next='Retain failure, repair invariant-direction implementation without changing cases or thresholds.')
        traceback.print_exc()
        raise
    finally:
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
        lock.close()
