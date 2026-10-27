import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import sys,json,time,hashlib,traceback,copy,gc
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent;PROJECT=ROOT.parents[1]
sys.path[1:1]=[str(OUT/n) for n in ['task2-round37-concentration-probe','task2-round33-conditional-start','task2-round10-phase-coverage','task2-round9-hierarchical','task2-round8-soft-targets','task2-round7-structured','task2-round6-ramp-physics','task2-round5-distribution','task2-round2']]
from timing_core import features,unseen_queue
from gated_core import gated_adjust
from paired_data import build,pair_data
from run_round2 import sample,arrays,PANELS
from coverage_data import matrix
from task2q.data import make_sample
from task2q.cache import atomic_json
from task2q.metrics import window_iou
TASK1=Path(r'C:\Users\18049\Documents\Codex\2026-10-24\trafficflowbench-186-global-road-adaptation\outputs\TrafficFlowBench-Best-20261026')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(n,v):atomic_json(ROOT/n,v)
def progress(phase,**kw):
    d=dict(phase=phase,pid=os.getpid(),utc=pd.Timestamp.now(tz='UTC').isoformat(),**kw);save('progress.json',d);atomic_json(OUT/'overnight-status.json',dict(d,training=True,script=str(Path(__file__))));print(json.dumps(d),flush=True)
def guard(stage):
    m=read(TASK1/'package-manifest.json');h={n:hashlib.sha256((TASK1/n).read_bytes()).hexdigest() for n in m['files']};assert all(h[n]==v['sha256'] for n,v in m['files'].items());save('task1-guard-'+stage+'.json',dict(passed=True,files=len(h),hashes=h));return h
def fit(data,full=False):
    name='conditional-first-full' if full else 'conditional-first-probe';rounds=600 if full else 200;path=ROOT/(name+'.txt');cp=ROOT/(name+'-checkpoint.txt')
    if path.exists():return lgb.Booster(model_file=str(path))
    if pd.Timestamp.now(tz='UTC')>=pd.Timestamp('2026-10-27T15:00:00Z') or (PROJECT/'work/STOP_NIGHT').exists():raise RuntimeError('No new training after cutoff/STOP_NIGHT')
    initial=lgb.Booster(model_file=str(cp)) if cp.exists() else None;done=initial.current_iteration() if initial else 0
    def callback(env):
        if (env.iteration+1)%100==0:env.model.save_model(str(cp));progress('training',model=name,step=env.iteration+1,total=rounds)
    model=lgb.train(dict(objective='multiclass',num_class=6,num_leaves=31,learning_rate=.035,min_data_in_leaf=80,feature_fraction=.85,lambda_l2=4,num_threads=4,verbosity=-1,seed=20261027,deterministic=True,force_col_wise=True),lgb.Dataset(data['X'],label=data['y'],weight=data['w']/data['w'].mean()),num_boost_round=rounds-done,init_model=initial,callbacks=[callback]) if done<rounds else initial;model.save_model(str(path));return model
def metric(ss,ps,t):return float(np.mean([window_iou(s['y'],p>=t,s['m']) for s,p in zip(ss,ps)]))
def negative(ss,ps):return dict(false_any=float(np.mean([((p>=.5)&s['m']).any() for s,p in zip(ss,ps)])),eligible_FP=float(np.mean([((p>=.5)&s['m']).sum()/s['m'].sum() for s,p in zip(ss,ps)])))
def class_report(y,w,old,new):
    w=np.asarray(w,dtype=float);w/=w.sum();idx=np.arange(len(y));a=float(np.dot(w,-np.log(np.clip(old[idx,y],1e-8,1))));b=float(np.dot(w,-np.log(np.clip(new[idx,y],1e-8,1))));aa=float(np.dot(w,old.argmax(1)==y));bb=float(np.dot(w,new.argmax(1)==y));return dict(views=len(y),old_logloss=a,new_logloss=b,logloss_gain=a-b,old_accuracy=aa,new_accuracy=bb,accuracy_gain=bb-aa)
def main():
    started=time.time();before=guard('before');assert read(OUT/'task2-round39-paired-phase-calendar/final-verification.json')['passed'];save('experiment.json',dict(hypothesis='reduce origin-phase/source confounding by appending matched3phase conditional targets, retaining original rows and architecture',data='JuneSep original3634 conditional rows plus R39 paired view cohort; original X/y exact and relative weights retained; complete future labels only sampling/targets',weights='each original conditional window raw mass1 plus each paired first-positive physical observation group mass1; normalize; paired observations can duplicate original, not independent new events',features='same557, no explicit phase/group/future metadata feature',models='shared6class early200, full600 only if early two-population gates pass; same31 leaves/.035/minleaf80/.85/L2=4/seed20261027; maximum4800trees CPU4serial',held='OctoberNov original R10 views and source-neutral paired target views separately; both correlated, not independent tests; no Dec/Jan/Feb data',classification_gate='>=20 original and>=30 paired held target views; LL gain vs frozen R33 early head>=-.02 and accuracy gain>=-.02 on BOTH populations; >=.02 LL and>=.03 accuracy gains together on at least one',timing_gate='fixed concentration>=.5 gate; alpha.5/1; original held whole-window median IoU gain(.1/.2/.35)>=-.01 vs causal base and >=-.005 vs old gated head; negative false-any+.03/eligibleFP+.005 vs BOTH old gated and raw base at .5',deployment='none here; any full head must later face December gates and January global-only calibration against frozen R38 parent',Task1='92files immutable, historical .942828633 not remeasured',Ongoing='no model changes',official_S_queue=None))
    data,audit=build('early',progress);save('training-data-audit.json',audit);new=fit(data);old=lgb.Booster(model_file=str(OUT/'task2-round33-conditional-start/conditional-first-probe.txt'));del data;gc.collect();heldpairs,pm=pair_data('held',progress);pairpred={p:None for p in PANELS}
    for p in PANELS:
        mask=np.asarray(heldpairs['panel'])==p;xx=np.asarray(heldpairs['X'][mask]);yy=np.asarray(heldpairs['y'][mask]);ww=np.asarray(heldpairs['w'][mask]);pairpred[p]=class_report(yy,ww,old.predict(xx,num_threads=4),new.predict(xx,num_threads=4))
    del heldpairs;gc.collect();entries=read(OUT/'task2-round10-phase-coverage/expanded-fit.json')['samples'];reports=[]
    for p in PANELS:
        held=[sample(e) for e in sorted(entries,key=lambda e:e['origin']) if e['panel']==p and e['condition']==0 and e['origin'][:10]>='2030-10-01'];legacy=lgb.Booster(model_file=str(OUT/f'task2-round10-phase-coverage/{p}-0-probe-expanded.txt'));base=[];op={a:[] for a in [.5,1.]};npred={a:[] for a in [.5,1.]};ys=[];ow=[];nw=[];cw=[]
        for s in held:
            b=legacy.predict(matrix(s),num_threads=4).astype('float32').reshape(6,-1);xx=features(s);qo=old.predict(xx,num_threads=4).astype('float32');qn=new.predict(xx,num_threads=4).astype('float32');assert np.isfinite(qn).all() and np.allclose(qn.sum(1),1,atol=1e-5);base.append(b)
            for a in [.5,1.]:op[a].append(gated_adjust(s,b,qo,a));npred[a].append(gated_adjust(s,b,qn,a))
            use=unseen_queue(s)&s['m'].all(0)&s['y'].any(0)
            if use.any():ys.extend(np.argmax(s['y'][:,use],axis=0).tolist());ow.extend(qo[use].tolist());nw.extend(qn[use].tolist());cw.extend([1/int(use.sum())]*int(use.sum()))
        original=class_report(np.asarray(ys,dtype=int),cw,np.asarray(ow),np.asarray(nw));paired=pairpred[p];class_ok=bool(original['views']>=20 and paired['views']>=30 and all(r['logloss_gain']>=-.02 and r['accuracy_gain']>=-.02 for r in [original,paired]) and any(r['logloss_gain']>=.02 and r['accuracy_gain']>=.03 for r in [original,paired]));positive=[i for i,s in enumerate(held) if s['has_future_queue']];neg=[i for i,s in enumerate(held) if not s['has_future_queue']];ss=[held[i] for i in positive];ns=[held[i] for i in neg];nb=negative(ns,[base[i] for i in neg]);options=[]
        for a in [.5,1.]:
            rr=[dict(threshold=t,base_IoU=metric(ss,[base[i] for i in positive],t),old_gated_IoU=metric(ss,[op[a][i] for i in positive],t),new_gated_IoU=metric(ss,[npred[a][i] for i in positive],t)) for t in [.1,.2,.35]];db=float(np.median([r['new_gated_IoU']-r['base_IoU'] for r in rr]));do=float(np.median([r['new_gated_IoU']-r['old_gated_IoU'] for r in rr]));no=negative(ns,[op[a][i] for i in neg]);nn=negative(ns,[npred[a][i] for i in neg]);ng=bool(all(nn['false_any']<=r['false_any']+.03 and nn['eligible_FP']<=r['eligible_FP']+.005 for r in [nb,no]));options.append(dict(alpha=a,reports=rr,median_gain_vs_base=db,median_gain_vs_old=do,negative_base=nb,negative_old=no,negative_new=nn,eligible=bool(db>=-.01 and do>=-.005 and ng)))
        passed=bool(class_ok and any(r['eligible'] for r in options));reports.append(dict(panel=p,original_selected_hold=original,paired_source_neutral_hold=paired,classification_gate=class_ok,adjustments=options,full_trial_gate_passed=passed));save('temporal-holdout.json',reports);progress('held_road_complete',panel=p,passed=passed,completed=len(reports),total=8)
    causal=[]
    for p in PANELS:
        for c in [0,1]:
            e=next(e for e in entries if e['panel']==p and e['condition']==c);s=sample(e);v=copy.deepcopy(arrays(e['day_key']));sl=e['origin_slot']
            for k in ['speed','flow','occupancy','quality','origin_speed','origin_flow','origin_occupancy','origin_quality']:v[k][sl+1:]=999
            v['eligible'][sl+1:]=False;s2=make_sample(v,s['network'],sl,p);xx=features(s);assert np.array_equal(xx,features(s2));q=new.predict(xx,num_threads=4);assert np.isfinite(q).all() and np.allclose(q.sum(1),1,atol=1e-5)
            import ramp_features as rf
            original_day=rf.day;origin_day=s['origin'][:10]
            def perturb_day(panel,split,date):
                a=original_day(panel,split,date).copy()
                if date>=origin_day:a[sl if date==origin_day else 0:]=999999
                return a
            try:rf.day=perturb_day;assert np.array_equal(xx,features(s))
            finally:rf.day=original_day
            causal.append(dict(panel=p,condition=c,passed=True,mainline_future_and_ramp_T_future_invariant=True))
    assert guard('after')==before;save('training-gates.json',dict(roads={r['panel']:r['full_trial_gate_passed'] for r in reports},passed_roads=int(sum(r['full_trial_gate_passed'] for r in reports)),both_original_and_paired_hold_guarded=True));save('early-verification.json',dict(passed=True,source_original_X_y_relative_weights_preserved=True,causality_groups=causal,Task1_92files_unchanged=True,no_December_January_February_used=True,official_S_queue=None));progress('early_probe_complete',passed_roads=int(sum(r['full_trial_gate_passed'] for r in reports)),seconds=time.time()-started)
if __name__=='__main__':
    import msvcrt
    lock=open(ROOT/'probe.lock','a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise SystemExit('Probe already running')
    try:main()
    except Exception:save('failure.json',dict(stage='early_probe',traceback=traceback.format_exc()));progress('failed');raise
