"""Real-input research preview and conditional score-change bounds; no truth."""
import common as d
import task4_span_shrink_stress as base
import task4_observable_gate_stress as gate
import task4_prior_structure as structure
import numpy as np,pandas as pd,json,msvcrt,traceback
from scipy.linalg import solve
from datetime import datetime,timezone
O=d.OUT/'task4_candidate_sensitivity'

def terms(Q,c,b,f0,f1,dest,epsilon):
 K=float(np.sum(Q,axis=0).max());den=max(float(c.sum()),1e-12);Fmin=(1-epsilon)*den/K;Dmin=max((float(np.abs(c-Q@b).sum())-epsilon*den)/K,0);movement=float(np.abs(f1-f0).sum());D0=float(np.abs(f0-b).sum());D1=float(np.abs(f1-b).sum());od_radius=min(1,movement/max(Fmin,1e-12));dev_radius=min(1,abs(D1-D0)/max(Dmin,1e-12));a0=np.bincount(dest,weights=f0);a1=np.bincount(dest,weights=f1);attr_radius=float(.5*np.abs(a0/max(a0.sum(),1e-12)-a1/max(a1.sum(),1e-12)).sum());link0=max(0,1-float(np.abs(Q@f0-c).sum()/den));link1=max(0,1-float(np.abs(Q@f1-c).sum()/den));radius=.45*od_radius+.15*dev_radius+.15*attr_radius;known=.25*(link1-link0);return dict(truth_total_lower_bound=Fmin,truth_prior_L1_lower_bound=Dmin,OD_change_absolute_upper_bound=od_radius,dev_change_absolute_upper_bound=dev_radius,attraction_change_absolute_upper_bound=attr_radius,S_link_baseline=link0,S_link_preview=link1,known_weighted_link_delta=known,conditional_Task4_delta_lower=known-radius,conditional_Task4_delta_upper=known+radius,conditional_total_delta_lower=.2*(known-radius),conditional_total_delta_upper=.2*(known+radius),path_L1_relative_to_baseline=movement/max(float(f0.sum()),1e-12),prior_L1_relative_change=(D1-D0)/max(D0,1e-12))

def numeric_check():
 rng=np.random.default_rng(2026110308)
 for i in range(200):
  Q=(rng.random((7,20))<.4).astype(float);truth=rng.gamma(2,1,20);b=rng.gamma(2,1,20);f0=rng.gamma(2,1,20);f1=rng.gamma(2,1,20);dest=rng.integers(0,4,20);c=np.maximum(Q@truth*(1+rng.normal(0,.01,7)),1e-6);epsilon=float(np.abs(Q@truth-c).sum()/c.sum())+1e-8;bound=terms(Q,c,b,f0,f1,dest,epsilon);delta=base.metrics(f1,truth,b,Q,c,dest)['synthetic_S_ODME']-base.metrics(f0,truth,b,Q,c,dest)['synthetic_S_ODME'];assert bound['conditional_Task4_delta_lower']-1e-10<=delta<=bound['conditional_Task4_delta_upper']+1e-10
 d.save(O/'bound_numeric_check.json',dict(seed=2026110308,cases=200,all_enclosed=True,note='Checks implementation of inequalities, not hidden truth or statistical coverage.'))

def main():
 protocol=dict(registered_utc=datetime.now(timezone.utc).isoformat(),hypothesis='Finite artificial zero-harm counts do not establish real OD safety. Inspect a frozen observable-gated5% preview and bound possible score changes under explicit reference-count compatibility assumptions.',splits=['validation','private'],final_flow_blend=[0.,.1,.25,.5,1.],assumed_reference_count_NMAE=[0.,.01,.05],method='Same registered observable gate, span ridge1, input-prior5% blend, nonnegative lambda.05 solve; blends are research sensitivity grid, not selected submissions.',bounds='For nonnegative hidden f* with ||A f*-c||1<=eps*sum(c), K=max column sum(A), F*=sum f*>=(1-eps)sum(c)/K and D*=||f*-b||1>=max((||c-Ab||1-eps sum(c))/K,0). Score Lipschitz bounds follow; no hidden score is evaluated.',selection='None; every panel, blend and assumption reported. No count-compatibility assumption is claimed verified for withheld reference.',source_sha256=d.sha(__file__),gate_source_sha256=d.sha(d.WORK/'task4_observable_gate_stress.py'),scope='Conditional numerical sensitivity, not a guaranteed improvement or statistical confidence interval. Path previews stay in work, no Kaggle submission, official baseline unchanged.')
 d.save(O/'protocol.json',protocol);numeric_check();basefile=d.ROOT/'work/review-repo/experiments/task4/robust_odme_20261027/outputs/task4_path_flow.csv';frozen=pd.read_csv(basefile,dtype={'panel':str,'path_id':str,'departure_time':str});rows=[];checks=[];hashes={}
 for panel in d.night.predictor.load_routes():
  for split in protocol['splits']:
   d.check_deadline();d.state(status='running',stage='task4_real_preview_sensitivity',panel=panel,split=split,next='Research-only preview with conditional bounds; no hidden-score evaluation, no automatic promotion')
   paths,Q,c,b,inputs=base.load(panel,split);hashes.update(inputs);folder=d.DATA/'task4'/panel/split;departure=pd.read_csv(folder/'synthetic_weak_prior.csv',usecols=['departure_time']).departure_time.iloc[0];f0=frozen[(frozen.panel==panel)&(frozen.departure_time==departure)].set_index('path_id').reindex(paths.path_id).path_flow.to_numpy(float);repeat,status=base.ridge(Q,c,b);difference=float(np.abs(repeat-f0).max());assert status['success'] and difference<1e-7;X=structure.features(paths)['span'];observed=gate.signal(b,X,gate.key_seed('real/'+panel+'/'+split));scale=max(float(np.median(b[b>0])),1e-9);penalty=np.eye(X.shape[1]);penalty[0,0]=0;coef=solve(X.T@X+penalty,X.T@(b/scale),assume_a='pos');smooth=np.maximum(X@coef*scale,0);alpha=.05 if observed['gate_enabled'] else 0.;preview,st=base.ridge(Q,c,(1-alpha)*b+alpha*smooth);assert st['success'] and np.isfinite(preview).all() and (preview>=0).all();dest=pd.factorize(paths.destination_zone,sort=True)[0];path=d.WORK/'task4_real_preview_NOT_SUBMITTED'/panel/(split+'.npz');path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,baseline_prediction=f0,experimental_prediction=preview,span_coefficients=coef);checks.append(dict(panel=panel,split=split,baseline_solver_max_difference=difference,preview_sha256=d.sha(path),**observed,**st))
   for fraction in protocol['final_flow_blend']:
    f1=(1-fraction)*f0+fraction*preview
    for epsilon in protocol['assumed_reference_count_NMAE']:rows.append(dict(panel=panel,split=split,final_blend=fraction,assumed_reference_count_NMAE=epsilon,**observed,**terms(Q,c,b,f0,f1,dest,epsilon)))
   pd.DataFrame(rows).to_csv(O/'by_panel.csv',index=False);d.save(O/'preview_checks.json',checks);d.save(O/'input_sha256.json',hashes);print('REAL PREVIEW SENSITIVITY',panel,split,flush=True)
 df=pd.DataFrame(rows);metrics=['conditional_Task4_delta_lower','conditional_Task4_delta_upper','conditional_total_delta_lower','conditional_total_delta_upper','path_L1_relative_to_baseline','S_link_baseline','S_link_preview','prior_L1_relative_change'];df.groupby(['split','final_blend','assumed_reference_count_NMAE'])[metrics].mean().to_csv(O/'macro_summary.csv');d.save(O/'baseline_source.json',dict(sha256=d.sha(basefile),scope='Frozen predictions only, not reference truth.'));d.state(status='stage_complete',stage='task4_real_preview_sensitivity_complete',next='Interpret movement and explicit-assumption bounds, not hidden scores. Gate passing all real priors adds no information about unknown OD matching; keep official Task4 baseline unchanged.')

if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True);lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:main()
 except Exception as e:d.state(status='failed_or_cutoff',stage='task4_candidate_sensitivity',error=repr(e),next='Repair candidate/bound implementation without changing gate, blend grid or truth-count assumptions; no Kaggle upload');traceback.print_exc();raise
 finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
