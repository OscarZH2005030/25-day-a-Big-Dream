"""Only change temporal shape where the frozen early head has a majority mode."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import sys,json,time,traceback,hashlib,copy
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent
sys.path[1:1]=[str(OUT/n) for n in ['task2-round33-conditional-start','task2-round10-phase-coverage','task2-round9-hierarchical','task2-round8-soft-targets','task2-round7-structured','task2-round6-ramp-physics','task2-round5-distribution','task2-round2']]
from run_round2 import sample,arrays,PANELS
from timing_core import features,adjust,unseen_queue
from gated_core import gated_adjust
from coverage_data import matrix
from task2q.data import make_sample
from task2q.cache import atomic_json
from task2q.metrics import window_iou
TASK1=Path(r'C:\Users\18049\Documents\Codex\2026-10-24\trafficflowbench-186-global-road-adaptation\outputs\TrafficFlowBench-Best-20261026')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(n,v):atomic_json(ROOT/n,v)
def progress(phase,**kw):
    d=dict(phase=phase,pid=os.getpid(),utc=pd.Timestamp.now(tz='UTC').isoformat(),**kw);save('progress.json',d);atomic_json(OUT/'overnight-status.json',dict(d,training=False,script=str(Path(__file__))));print(json.dumps(d),flush=True)
def guard(stage):
    m=read(TASK1/'package-manifest.json');h={n:hashlib.sha256((TASK1/n).read_bytes()).hexdigest() for n in m['files']};assert all(h[n]==v['sha256'] for n,v in m['files'].items());save('task1-guard-'+stage+'.json',dict(passed=True,files=len(h),hashes=h));return h
def metric(ss,ps,t):return float(np.mean([window_iou(s['y'],p>=t,s['m']) for s,p in zip(ss,ps)]))
def negative(ss,ps):return dict(windows=len(ss),false_any=float(np.mean([((p>=.5)&s['m']).any() for s,p in zip(ss,ps)])),eligible_FP=float(np.mean([((p>=.5)&s['m']).sum()/s['m'].sum() for s,p in zip(ss,ps)])))
def main():
    started=time.time();before=guard('before');save('experiment.json',dict(hypothesis='R34 conditional timing can worsen roads where the head is diffuse; keep base shape unless a six-class conditional first-step mode has >=.5 mass',scope='no new training; frozen R33 JuneSep200 head and R10 JuneSep200 base; OctoberNov all-selected-phase held only; no December/January/February raw data',concentration='fixed max conditional class output>=.5 majority mode; NOT a calibrated probability of correctness or an event probability; threshold not searched',safety='unchanged base on low-concentration/visible-queue links and Ongoing; identical unmasked six-step peak on all links, unknown future-eligibility support can change',conditional_gate='>=20 high-concentration complete-label positive link views, >=.20 weighted conditional-event coverage, accuracy>=.50 and improvement>=.03 over JuneSep per-road class prior, logloss improvement>=.02',timing_gate='at least one fixedalpha.5/1 has median IoU gain at .1/.2/.35>=-.01 vs causal base, median change vs ungated head>=0, negative false-any+.03 and eligibleFP+.005 at fixed .5',limits='conditional complete-label event-positive evaluation is selected and correlated; coverage is not general traffic coverage; no new policy is deployed here',new_models=0,CPU_threads=4,Task1='92files immutable; .942828633 historical only',official_S_queue=None))
    headpath=OUT/'task2-round33-conditional-start/conditional-first-probe.txt';head=lgb.Booster(model_file=str(headpath));priorrows=read(OUT/'task2-round33-conditional-start/temporal-holdout.json');priors={r['panel']:np.asarray(r['class_prior']) for r in priorrows};entries=read(OUT/'task2-round10-phase-coverage/expanded-fit.json')['samples'];reports=[]
    for p in PANELS:
        es=sorted([e for e in entries if e['panel']==p and e['condition']==0 and e['origin'][:10]>='2030-10-01'],key=lambda e:e['origin']);held=[];base=[];raw={a:[] for a in [.5,1.]};gated={a:[] for a in [.5,1.]};labels=[];probs=[];caseweights=[];legacy=lgb.Booster(model_file=str(OUT/f'task2-round10-phase-coverage/{p}-0-probe-expanded.txt'))
        for e in es:
            s=sample(e);held.append(s);b=legacy.predict(matrix(s),num_threads=4).astype('float32').reshape(6,-1);q=head.predict(features(s),num_threads=4).astype('float32');base.append(b)
            for a in [.5,1.]:raw[a].append(adjust(s,b,q,a));gated[a].append(gated_adjust(s,b,q,a))
            use=unseen_queue(s)&s['m'].all(0)&s['y'].any(0)
            if use.any():labels.extend(np.argmax(s['y'][:,use],axis=0).tolist());probs.extend(q[use].tolist());caseweights.extend([1/int(use.sum())]*int(use.sum()))
        yy=np.asarray(labels,dtype=int);qq=np.asarray(probs);w=np.asarray(caseweights);high=qq.max(1)>=.5;coverage=float(w[high].sum()/w.sum());nh=int(high.sum());prior=priors[p]
        if nh:
            wh=w[high]/w[high].sum();acc=float(np.dot(wh,qq[high].argmax(1)==yy[high]));pa=float(np.dot(wh,np.full(nh,prior.argmax())==yy[high]));ll=float(np.dot(wh,-np.log(np.clip(qq[high][np.arange(nh),yy[high]],1e-8,1))));pl=float(np.dot(wh,-np.log(prior[yy[high]])))
        else:acc=pa=0.;ll=pl=None
        classification=bool(nh>=20 and coverage>=.20 and acc>=.50 and acc>=pa+.03 and ll<=pl-.02);positive=[i for i,s in enumerate(held) if s['has_future_queue']];neg=[i for i,s in enumerate(held) if not s['has_future_queue']];ss=[held[i] for i in positive];ns=[held[i] for i in neg];nb=negative(ns,[base[i] for i in neg]);options=[]
        for a in [.5,1.]:
            rr=[]
            for t in [.1,.2,.35]:rr.append(dict(threshold=t,base_IoU=metric(ss,[base[i] for i in positive],t),ungated_IoU=metric(ss,[raw[a][i] for i in positive],t),gated_IoU=metric(ss,[gated[a][i] for i in positive],t)))
            delta=float(np.median([r['gated_IoU']-r['base_IoU'] for r in rr]));protect=float(np.median([r['gated_IoU']-r['ungated_IoU'] for r in rr]));nn=negative(ns,[gated[a][i] for i in neg]);ng=bool(nn['false_any']<=nb['false_any']+.03 and nn['eligible_FP']<=nb['eligible_FP']+.005);options.append(dict(alpha=a,reports=rr,median_gain=delta,median_change_vs_ungated=protect,negative_base=nb,negative_gated=nn,negative_guard_passed=ng,eligible=bool(delta>=-.01 and protect>=0 and ng)))
        row=dict(panel=p,conditional_link_views=len(yy),high_concentration_views=nh,weighted_conditional_event_coverage=coverage,high_group_accuracy=acc,prior_high_group_accuracy=pa,high_group_logloss=ll,prior_high_group_logloss=pl,classification_gate_passed=classification,adjustments=options,full_trial_gate_passed=bool(classification and any(r['eligible'] for r in options)));reports.append(row);save('concentration-holdout.json',reports);progress('held_road_complete',panel=p,passed=row['full_trial_gate_passed'],completed=len(reports),total=8)
    causal=[]
    for p in PANELS:
        legacy=lgb.Booster(model_file=str(OUT/f'task2-round10-phase-coverage/{p}-0-probe-expanded.txt'))
        for c in [0,1]:
            e=next(e for e in entries if e['panel']==p and e['condition']==c);s=sample(e);v=copy.deepcopy(arrays(e['day_key']));sl=e['origin_slot']
            for k in ['speed','flow','occupancy','quality','origin_speed','origin_flow','origin_occupancy','origin_quality']:v[k][sl+1:]=999
            v['eligible'][sl+1:]=False;s2=make_sample(v,s['network'],sl,p);xx=features(s);assert np.array_equal(xx,features(s2));b=legacy.predict(matrix(s),num_threads=4).astype('float32').reshape(6,-1);q=head.predict(xx,num_threads=4).astype('float32');expected=gated_adjust(s,b,q,1.);assert np.array_equal(expected,gated_adjust(s2,b,head.predict(features(s2),num_threads=4).astype('float32'),1.))
            import ramp_features as rf
            original_day=rf.day;origin_day=s['origin'][:10]
            def perturb_day(panel,split,date):
                a=original_day(panel,split,date).copy()
                if date>=origin_day:a[sl if date==origin_day else 0:]=999999
                return a
            try:rf.day=perturb_day;assert np.array_equal(xx,features(s))
            finally:rf.day=original_day
            causal.append(dict(panel=p,condition=c,passed=True,mainline_future_and_ramp_T_future_invariant=True))
    assert len(causal)==16;assert guard('after')==before;save('training-gates.json',dict(roads={r['panel']:r['full_trial_gate_passed'] for r in reports},passed_roads=int(sum(r['full_trial_gate_passed'] for r in reports)),new_training_required=False,full_head_available_readonly='R34; any actual candidate must be compared to frozen R34 and keep January global-only calibration'))
    save('final-verification.json',dict(passed=True,diagnostic_early_probe_only=True,new_models=0,causality_groups=causal,head_sha256=hashlib.sha256(headpath.read_bytes()).hexdigest(),Task1_92files_unchanged=True,no_December_January_February_data_used=True,official_S_queue=None));(ROOT/'第三十七轮诊断说明.txt').write_text('固定多数类集中度门控，未重新训练，也不宣称概率已校准；只在R33早期head和OctNov留出上检验。\n'+json.dumps(reports,ensure_ascii=False,indent=2)+'\n',encoding='utf8');progress('early_probe_complete',passed_roads=int(sum(r['full_trial_gate_passed'] for r in reports)),seconds=time.time()-started)
if __name__=='__main__':
    try:main()
    except Exception:save('failure.json',dict(traceback=traceback.format_exc()));progress('failed');raise
