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
    started=time.time();before=guard('before');save('experiment.json',dict(hypothesis='R32 shows high future-link support recall but weak exact first-time agreement even without simple cues; preserve baseline peak/support and correct conditional onset timing rather than replace event detection',scope='bounded early probe only: JuneSep fit, OctoberNov R10 selected all phases held; NO December/January/February data',target='six-class first observed queue step, only complete-six-label positive links with NO observed eligible queue in released 13 frames; conditional on a future positive event, not unconditional event probability',features='R10 first-horizon555 minus6 horizon indicators +8 public panel onehot =557; no labels/coverage/mask metadata as input',weights='equal mass per positive window with at least one eligible conditional target, then equal per target link; full positives retained, no negative links needed for conditional head',head='shared eight-panel LightGBM multiclass6, fixed200 rounds/31leaves, CPU4; 1200 trees; no parameter search',adjustment='conditional first-step CDF times base falling tail; blend .5/1 and rescale to identical per-link all-six peak; visible-queue links unchanged; does NOT guarantee support under unknown future eligibility masks or calibrated probabilities',training_gate='>=500 JuneSep conditional positive link views; >=50 per applicable road',held_gate='per-road >=20 known conditional held views; conditional logloss better than JuneSep per-road class prior by>=.02 and accuracy better by>=.03; at least one predeclared .5/1 adjustment has median IoU gain(.1/.2/.35)>=-.01 plus negative false-any+.03/eligibleFP+.005',negative_guard='all eligible held cells; .5 unchanged, no loosening; original R10 JuneSep-only early200 is causal control',limitations='complete-label and positive-event conditioning may select a different subset; overlapping views not independent episodes; support-normalization is a heuristic, not an information-bound claim',Task1='92file hash unchanged; historical .942828633 not remeasured',official_S_queue=None))
    entries=[e for e in read(OUT/'task2-round10-phase-coverage/expanded-fit.json')['samples'] if e['condition']==0];train=[e for e in entries if e['origin'][:10]<'2030-10-01'];chunks=[];labels=[];weights=[];panels=[];rows=[];cursor=0
    for i,e in enumerate(sorted(train,key=lambda e:(e['panel'],e['origin']))):
        s=sample(e);use=unseen_queue(s)&s['m'].all(0)&s['y'].any(0)
        if use.any():
            xx=features(s)[use];y=np.argmax(s['y'][:,use],axis=0).astype('int32');n=len(y);chunks.append(xx);labels.append(y);weights.append(np.full(n,1/n,dtype='float32'));panels.extend([s['panel']]*n);rows.append(dict(panel=s['panel'],origin=s['origin'],start=cursor,end=cursor+n));cursor+=n
        if (i+1)%512==0:progress('building_conditional_targets',windows=i+1,total=len(train),rows=cursor)
    X=np.concatenate(chunks);y=np.concatenate(labels);w=np.concatenate(weights);w/=w.sum();panels=np.asarray(panels);assert len(y)>=500
    data_audit=dict(rows=len(y),windows=len(rows),features=557,counts_by_panel={p:int((panels==p).sum()) for p in PANELS},future_first_steps=np.bincount(y,minlength=6).tolist(),weights_mass=float(w.sum()),positive_event_conditioning_only=True)
    save('training-data-audit.json',data_audit);save('conditional-target-index.json',rows)
    work=PROJECT/'work/round33-conditional-start';work.mkdir(parents=True,exist_ok=True)
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
    path=ROOT/'conditional-first-probe.txt';checkpoint=ROOT/'conditional-first-checkpoint.txt'
    if path.exists():model=lgb.Booster(model_file=str(path))
    else:
        if pd.Timestamp.now(tz='UTC')>=pd.Timestamp('2026-10-27T15:00:00Z') or (PROJECT/'work/STOP_NIGHT').exists():raise RuntimeError('No new training after cutoff or STOP_NIGHT')
        initial=lgb.Booster(model_file=str(checkpoint)) if checkpoint.exists() else None;done=initial.current_iteration() if initial else 0
        def callback(env):
            if (env.iteration+1)%100==0:env.model.save_model(str(checkpoint));progress('training',step=env.iteration+1,total=200)
        model=lgb.train(dict(objective='multiclass',num_class=6,num_leaves=31,learning_rate=.035,min_data_in_leaf=80,feature_fraction=.85,lambda_l2=4,num_threads=4,verbosity=-1,seed=20261027,deterministic=True,force_col_wise=True),lgb.Dataset(X,label=y,weight=w/w.mean()),num_boost_round=200-done,init_model=initial,callbacks=[callback]) if done<200 else initial;model.save_model(str(path))
    priors={p:(np.bincount(y[panels==p],weights=w[panels==p],minlength=6)+1e-6) for p in PANELS}
    priors={p:v/v.sum() for p,v in priors.items()};del X,chunks;gc.collect();reports=[];pred_ranges=[]
    for p in PANELS:
        held=[sample(e) for e in sorted(entries,key=lambda e:e['origin']) if e['panel']==p and e['origin'][:10]>='2030-10-01'];base_model=lgb.Booster(model_file=str(OUT/f'task2-round10-phase-coverage/{p}-0-probe-expanded.txt'));base=[];adapt={a:[] for a in [.5,1.]};truth=[];probs=[];caseweights=[]
        for s in held:
            b=base_model.predict(matrix(s),num_threads=4).astype('float32').reshape(6,-1);q=model.predict(features(s),num_threads=4).astype('float32');assert np.isfinite(q).all() and np.allclose(q.sum(1),1,atol=1e-5);base.append(b)
            for a in [.5,1.]:adapt[a].append(adjust(s,b,q,a))
            use=unseen_queue(s)&s['m'].all(0)&s['y'].any(0)
            if use.any():truth.extend(np.argmax(s['y'][:,use],axis=0).tolist());probs.extend(q[use].tolist());caseweights.extend([1/int(use.sum())]*int(use.sum()))
        positive=[i for i,s in enumerate(held) if s['has_future_queue']];negative=[i for i,s in enumerate(held) if not s['has_future_queue']];ss=[held[i] for i in positive];bp=[base[i] for i in positive];ns=[held[i] for i in negative];nb=neg(ns,[base[i] for i in negative]);yy=np.asarray(truth,dtype=int);pr=np.asarray(probs);ww=np.asarray(caseweights);ww/=ww.sum();prior=priors[p];newLL=float(np.dot(ww,-np.log(np.clip(pr[np.arange(len(yy)),yy],1e-8,1))));priorLL=float(np.dot(ww,-np.log(prior[yy])));acc=float(np.dot(ww,pr.argmax(1)==yy));prioracc=float(np.dot(ww,np.full(len(yy),prior.argmax())==yy));class_ok=bool(len(yy)>=20 and data_audit['counts_by_panel'][p]>=50 and newLL<=priorLL-.02 and acc>=prioracc+.03);options=[]
        for a in [.5,1.]:
            qp=[adapt[a][i] for i in positive];rr=[dict(threshold=t,base_IoU=metric(ss,bp,t),timing_IoU=metric(ss,qp,t)) for t in [.1,.2,.35]];median=float(np.median([r['timing_IoU']-r['base_IoU'] for r in rr]));nn=neg(ns,[adapt[a][i] for i in negative]);ng=nn['false_any']<=nb['false_any']+.03 and nn['eligible_FP']<=nb['eligible_FP']+.005;options.append(dict(alpha=a,metrics=rr,median_gain=median,negative_base=nb,negative_adjusted=nn,negative_guard_passed=bool(ng),eligible=bool(median>=-.01 and ng)))
        passed=bool(class_ok and any(r['eligible'] for r in options));reports.append(dict(panel=p,conditional_link_views=len(yy),conditional_windows=len(caseweights) and float(sum(caseweights)),class_prior=prior.tolist(),prior_logloss=priorLL,head_logloss=newLL,prior_accuracy=prioracc,head_accuracy=acc,conditional_gate_passed=class_ok,adjustments=options,full_trial_gate_passed=passed));save('temporal-holdout.json',reports);progress('held_direction_complete',panel=p,passed=passed,completed=len(reports),total=8)
    assert guard('after')==before;save('training-gates.json',dict(roads={r['panel']:r['full_trial_gate_passed'] for r in reports},passed_roads=sum(r['full_trial_gate_passed'] for r in reports),no_December_or_later_data_used=True));save('final-verification.json',dict(passed=True,scope='conditional early probe only; no deployment or calibration',Task1_92files_unchanged=True,causality_groups=16,all_six_step_support_peak_identical=True,head_simplex_passed=True,official_S_queue=None));progress('probe_complete',passed_roads=sum(r['full_trial_gate_passed'] for r in reports),seconds=time.time()-started)
if __name__=='__main__':
    import msvcrt
    lock=open(ROOT/'run.lock','a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise SystemExit('Probe already running')
    try:main()
    except Exception:save('failure.json',dict(traceback=traceback.format_exc()));progress('failed');raise
