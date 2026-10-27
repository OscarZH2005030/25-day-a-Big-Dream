"""JuneSep-only conditional first-arrival probe; no December/January/February data."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import sys,json,hashlib,traceback,time,copy,gc
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent;PROJECT=ROOT.parents[1]
sys.path[1:1]=[str(OUT/n) for n in ['task2-round10-phase-coverage','task2-round9-hierarchical','task2-round8-soft-targets','task2-round7-structured','task2-round6-ramp-physics','task2-round5-distribution','task2-round2']]
from run_round2 import sample,arrays,PANELS
from task2q.data import make_sample
from task2q.cache import atomic_json
from task2q.metrics import window_iou
from coverage_data import matrix
from timing_core import features,unseen_queue,adjust
TASK1=Path(r'C:\Users\18049\Documents\Codex\2026-10-24\trafficflowbench-186-global-road-adaptation\outputs\TrafficFlowBench-Best-20261026')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(n,v):atomic_json(ROOT/n,v)
def progress(phase,**kw):
    v=dict(phase=phase,pid=os.getpid(),utc=pd.Timestamp.now(tz='UTC').isoformat(),**kw);save('progress.json',v);print(json.dumps(v),flush=True)
def guard(stage):
    m=read(TASK1/'package-manifest.json');h={n:hashlib.sha256((TASK1/n).read_bytes()).hexdigest() for n in m['files']};assert all(h[n]==v['sha256'] for n,v in m['files'].items());save('task1-guard-'+stage+'.json',dict(passed=True,files=len(h),hashes=h));return h
def metric(ss,ps,t):return float(np.mean([window_iou(s['y'],p>=t,s['m']) for s,p in zip(ss,ps)]))
def neg(ss,ps):
    return dict(windows=len(ss),false_any=float(np.mean([((p>=.5)&s['m']).any() for s,p in zip(ss,ps)])),eligible_FP=float(np.mean([((p>=.5)&s['m']).sum()/s['m'].sum() for s,p in zip(ss,ps)])))
def main():
    started=time.time();before=guard('before');gates=read(OUT/'task2-round33-conditional-start/training-gates.json');assert read(OUT/'task2-round33-conditional-start/final-verification.json')['passed'] and gates['passed_roads']>0
    save('experiment.json',dict(hypothesis='retain existing strong future-link event support and learn conditional first-arrival timing; full head authorized by five R33 OctoberNov-only passed roads',stage='full shared head training only; December selection/January calibration/February reevaluation PENDING; no strategy replacement',fit='original R10 June-November Onset selected windows, complete-six-frame positive links with no observed eligible queue in released13frames; no December or later raw data',target='six-class first future observed queue step conditioned on event occurrence and complete labels; no event detection refit',features='same557 as R33: first-horizon555 minus6 horizon onehot +8 public panel onehot',weights='equal conditional-positive window mass then equal positive target-link mass; all targets retained, correlated views not independent episodes',head='shared eight-panel multiclass6, fixed600boosting rounds/31 leaves/.035/minleaf80/.85/L2=4/seed20261027; 3600 trees; no parameter search',selection_gate_source=gates,December_plan='only R33 passed roads, require >=4 per phase; parent R31 unchanged; normalized support-preserving temporal alpha.5/1, mean IoU gain>=.004/worstphase>=-.015',January_plan='only global Onsetalpha0/.5/1 with same normalized timing blend, thresholds .5/.25 and negative false-any+.03/eligibleFP+.005 fixed; Ongoing unchanged',February_plan='only after freezing, both already-used populations for paired report; never choose roads from February',normalization='R33 exact same per-link all-six peak and visible-queue link invariants; future-eligibility masked support may differ and needs negative guard; heuristic shape, not calibrated probability',Task1='92files protected; .942828633 historical not measured anew; R21/R23 saved archives read-only',CPU_threads=4,official_S_queue=None))

    entries=[e for e in read(OUT/'task2-round10-phase-coverage/expanded-fit.json')['samples'] if e['condition']==0];train=entries;chunks=[];labels=[];weights=[];panels=[];rows=[];cursor=0
    for i,e in enumerate(sorted(train,key=lambda e:(e['panel'],e['origin']))):
        s=sample(e);use=unseen_queue(s)&s['m'].all(0)&s['y'].any(0)
        if use.any():
            xx=features(s)[use];y=np.argmax(s['y'][:,use],axis=0).astype('int32');n=len(y);chunks.append(xx);labels.append(y);weights.append(np.full(n,1/n,dtype='float32'));panels.extend([s['panel']]*n);rows.append(dict(panel=s['panel'],origin=s['origin'],start=cursor,end=cursor+n));cursor+=n
        if (i+1)%512==0:progress('building_conditional_targets',windows=i+1,total=len(train),rows=cursor)
    X=np.concatenate(chunks);y=np.concatenate(labels);w=np.concatenate(weights);w/=w.sum();panels=np.asarray(panels);assert len(y)>=500
    data_audit=dict(rows=len(y),windows=len(rows),features=557,counts_by_panel={p:int((panels==p).sum()) for p in PANELS},future_first_steps=np.bincount(y,minlength=6).tolist(),weights_mass=float(w.sum()),positive_event_conditioning_only=True)
    save('training-data-audit.json',data_audit);save('conditional-target-index.json',rows)
    work=PROJECT/'work/round34-conditional-start-full';work.mkdir(parents=True,exist_ok=True)
    for name,v in [('X',X),('y',y),('w',w)]:np.save(work/(name+'.npy'),v)
    causal=[]
    for p in PANELS:
        for c in [0,1]:
            e=next(e for e in read(OUT/'task2-round10-phase-coverage/expanded-fit.json')['samples'] if e['panel']==p and e['condition']==c);s=sample(e);v=copy.deepcopy(arrays(e['day_key']));sl=e['origin_slot']
            for k in ['speed','flow','occupancy','quality','origin_speed','origin_flow','origin_occupancy','origin_quality']:v[k][sl+1:]=999
            v['eligible'][sl+1:]=False;s2=make_sample(v,s['network'],sl,p);expected=features(s);assert np.array_equal(expected,features(s2)) and np.array_equal(unseen_queue(s),unseen_queue(s2))
            import ramp_features as rf
            original_day=rf.day;origin_day=s['origin'][:10]
            def perturb_ramp_day(panel,split,date):
                a=original_day(panel,split,date).copy()
                if date>=origin_day:a[sl if date==origin_day else 0:]=999999
                return a
            try:
                rf.day=perturb_ramp_day;assert np.array_equal(expected,features(s))
            finally:rf.day=original_day
            causal.append(dict(panel=p,condition=c,passed=True,ramp_T_and_future_invariant=True))
    save('feature-causality-check.json',dict(passed=True,groups=causal))
    path=ROOT/'conditional-first-full.txt';checkpoint=ROOT/'conditional-first-full-checkpoint.txt'
    if path.exists():model=lgb.Booster(model_file=str(path))
    else:
        if pd.Timestamp.now(tz='UTC')>=pd.Timestamp('2026-10-27T15:00:00Z') or (PROJECT/'work/STOP_NIGHT').exists():raise RuntimeError('No new training after cutoff or STOP_NIGHT')
        initial=lgb.Booster(model_file=str(checkpoint)) if checkpoint.exists() else None;done=initial.current_iteration() if initial else 0
        def callback(env):
            if (env.iteration+1)%100==0:env.model.save_model(str(checkpoint));progress('training',step=env.iteration+1,total=600)
        model=lgb.train(dict(objective='multiclass',num_class=6,num_leaves=31,learning_rate=.035,min_data_in_leaf=80,feature_fraction=.85,lambda_l2=4,num_threads=4,verbosity=-1,seed=20261027,deterministic=True,force_col_wise=True),lgb.Dataset(X,label=y,weight=w/w.mean()),num_boost_round=600-done,init_model=initial,callbacks=[callback]) if done<600 else initial;model.save_model(str(path))
    assert model.num_feature()==557 and model.current_iteration()==600
    probe=model.predict(X[:min(len(X),1024)],num_threads=4);assert np.isfinite(probe).all() and np.allclose(probe.sum(1),1,atol=1e-5)
    assert guard('after')==before
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['conditional-first-full.txt','timing_core.py']}
    save('full-training-verification.json',dict(passed=True,trained_only_no_deployment=True,boosting_rounds=600,trees=model.num_trees(),features=557,rows=len(y),probability_simplex_passed=True,feature_mainline_future_and_ramp_T_future_groups=16,Task1_92files_unchanged=True,hashes=hashes,early_gate_sha256=hashlib.sha256((OUT/'task2-round33-conditional-start/training-gates.json').read_bytes()).hexdigest(),fit_index_sha256=hashlib.sha256((OUT/'task2-round10-phase-coverage/expanded-fit.json').read_bytes()).hexdigest(),pending='December selection, January calibration, frozen paired February reports and inference validation; current best remains unchanged',official_S_queue=None))
    progress('full_head_trained_pending_development',rows=len(y),seconds=time.time()-started)

if __name__=='__main__':
    import msvcrt
    lock=open(ROOT/'run.lock','a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise SystemExit('Probe already running')
    try:main()
    except Exception:save('failure.json',dict(traceback=traceback.format_exc()));progress('failed');raise
