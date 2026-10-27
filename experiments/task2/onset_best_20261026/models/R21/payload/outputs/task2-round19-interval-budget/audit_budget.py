"""Before extending R18, audit its already-trained iteration learning curve."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='4'
import sys,json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent;R18=OUT/'task2-round18-interval-likelihood'
sys.path[1:1]=[str(R18),str(OUT/'task2-round2')]
from interval_runtime import IntervalPredictor
from interval_data import features
from interval_core import softmax,PATTERNS,allowed,loss_grad
from run_round2 import sample,PANELS
from task2q.cache import atomic_json
from task2q.metrics import window_iou
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(n,v):atomic_json(ROOT/n,v)
def progress(phase,**kw):
    d=dict(phase=phase,pid=os.getpid(),utc=pd.Timestamp.now(tz='UTC').isoformat(),**kw);save('progress.json',d);print(json.dumps(d),flush=True)
def main():
    save('experiment.json',dict(phase='pre-training budget diagnosis, no new model yet',hypothesis='R18 rejected at60 conservative-curvature boost rounds; test whether held-time loss/IoU still improve before any continuation',selection='only June-September training and October-November held time, no February input',continuation_gate='at least one arm: median across panels IoU gain60minus40>=.01 over fixed thresholds .1/.2/.35, and held partial likelihood loss60<loss40',maximum_continuation='if gate passes, exactly200 boost rounds; no increased rounds/threshold search beyond fixed original full budget',CPU_threads=4,official_S_queue=None))
    entries=read(OUT/'task2-round10-phase-coverage/expanded-fit.json')['samples'];rng=np.random.default_rng(20261027);ss=[]
    for p in PANELS:
        group=[e for e in entries if e['panel']==p and e['condition']==0 and e['source']=='original' and e['origin'][:10]>='2030-10-01' and e['has_future_queue']];ss.extend(sample(e) for e in group)
    X=[features(s) for s in ss];records=[];summary={}
    # Holdout proper losses use the same public target-set construction and
    # sampled full training matrices restricted to October-November rows.
    work=ROOT.parents[1]/'work/round18-interval-likelihood';codes=np.load(work/'code.npy',mmap_mode='r');early=np.load(work/'early.npy',mmap_mode='r');w=np.load(work/'w.npy',mmap_mode='r');matrix=np.load(work/'X.npy',mmap_mode='r');indices=np.flatnonzero(early==0);indices=rng.choice(indices,min(60000,len(indices)),replace=False);A=allowed(codes[indices]);W=np.asarray(w[indices],dtype='float64');hx=np.asarray(matrix[indices])
    for kind in ['control','partial']:
        model=lgb.Booster(model_file=str(R18/('global-'+kind+'-probe.txt')));curve=[]
        for rounds in [20,40,60]:
            progress('audit_existing_curve',kind=kind,rounds=rounds);by_panel={p:[] for p in PANELS}
            for s,x in zip(ss,X):
                p=softmax(model.predict(x,raw_score=True,num_iteration=rounds,num_threads=4));q=(p@PATTERNS).astype('float32').T;values=[window_iou(s['y'],q>=t,s['m']) for t in [.1,.2,.35]];by_panel[s['panel']].append(values)
            row=dict(kind=kind,rounds=rounds,by_panel={p:np.mean(v,axis=0).tolist() for p,v in by_panel.items()},macro_threshold_IoU=np.mean([np.mean(v,axis=0) for v in by_panel.values()],axis=0).tolist());raw=model.predict(hx,raw_score=True,num_iteration=rounds,num_threads=4);row['held_partial_likelihood_loss']=loss_grad(raw,A,W)[0];curve.append(row);records.append(row);save('learning-curve.json',records)
        gains=[float(np.median(np.array(curve[2]['by_panel'][p])-np.array(curve[1]['by_panel'][p]))) for p in PANELS];median=float(np.median(gains));summary[kind]=dict(median_panel_IoU_gain60minus40=median,per_panel_gains=dict(zip(PANELS,gains)),loss40=curve[1]['held_partial_likelihood_loss'],loss60=curve[2]['held_partial_likelihood_loss'],continuation_gate_passed=bool(median>=.01 and curve[2]['held_partial_likelihood_loss']<curve[1]['held_partial_likelihood_loss']))
    save('budget-gate.json',dict(passed=any(r['continuation_gate_passed'] for r in summary.values()),arms=summary,no_February_used=True));progress('diagnostic_complete',continuation_gate_passed=any(r['continuation_gate_passed'] for r in summary.values()))
if __name__=='__main__':
    try:main()
    except Exception:save('failure.json',dict(traceback=traceback.format_exc()));progress('failed');raise
