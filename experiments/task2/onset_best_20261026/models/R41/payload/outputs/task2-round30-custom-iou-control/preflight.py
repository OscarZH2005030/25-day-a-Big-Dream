import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import sys,json,traceback,hashlib
from pathlib import Path
import numpy as np
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent;PROJECT=ROOT.parents[1]
sys.path[1:1]=[str(OUT/'task2-round2')]
from run_round2 import PANELS
from task2q.cache import atomic_json
from window_objective import WindowObjective,sigmoid
def main():
    y=np.array([1,0,0,1,0,0,0],float);w=np.array([.08,.16,.16,.1,.3,.08,.12]);wins=[dict(start=0,end=3,positive=True),dict(start=3,end=5,positive=True),dict(start=5,end=7,positive=False)]
    obj=WindowObjective(y,w,wins);z=np.array([-.8,.2,.5,.4,-.2,.8,-.5]);g,h=obj.gradient(z);eps=1e-5;fd=[]
    for i in range(len(z)):
        a=z.copy();b=z.copy();a[i]+=eps;b[i]-=eps;fd.append((obj.loss(a)-obj.loss(b))/(2*eps))
    error=float(np.max(np.abs(g-fd)));assert error<1e-7 and (h>0).all()
    # At coefficient zero, the offset custom objective must reproduce binary boosting.
    rng=np.random.default_rng(20261027);X=rng.normal(size=(4096,16));Y=(X[:,0]+.4*X[:,2]+rng.normal(size=4096)>.8).astype(float)
    W=rng.uniform(.5,2,len(Y));pos=Y>.5;W[pos]*=.8/W[pos].sum();W[~pos]*=.2/W[~pos].sum()
    Y[-512:]=0;W=np.ones(len(Y));W[:-512]*=.8/(len(Y)-512);W[-512:]*=.2/512
    zero=WindowObjective(Y,W,[dict(start=0,end=len(Y)-512,positive=True),dict(start=len(Y)-512,end=len(Y),positive=False)],coefficient=0)
    params=dict(num_leaves=31,learning_rate=.035,min_data_in_leaf=80,feature_fraction=.85,lambda_l2=4,num_threads=4,verbosity=-1,seed=20261027,deterministic=True,force_col_wise=True)
    ds=lambda:lgb.Dataset(X,label=Y,weight=zero.w)
    def direct(z,d):
        p=sigmoid(z);w=np.asarray(d.get_weight());return w*(p-d.get_label()),w*p*(1-p)
    a=lgb.train(dict(params,objective=direct),ds(),num_boost_round=7)
    b=lgb.train(dict(params,objective=zero),ds(),num_boost_round=7)
    equivalence=float(np.max(np.abs(sigmoid(a.predict(X,raw_score=True,num_threads=4))-sigmoid(b.predict(X,raw_score=True,num_threads=4)))))
    assert equivalence<1e-7,equivalence
    # Independent built-in binary sanity check with feature subsampling disabled.
    params['feature_fraction']=1.
    a1=lgb.train(dict(params,objective='binary',boost_from_average=False),ds(),num_boost_round=7)
    b1=lgb.train(dict(params,objective=zero),ds(),num_boost_round=7)
    binary_error=float(np.max(np.abs(a1.predict(X,num_threads=4)-sigmoid(b1.predict(X,raw_score=True,num_threads=4)))))
    assert binary_error<1e-7,binary_error
    summaries=[];gates={}
    for panel in PANELS:
        folder=PROJECT/f'work/round10-phase/{panel}-0';meta=json.loads((folder/'complete.json').read_text(encoding='utf8'));yy=np.load(folder/'y.npy',mmap_mode='r');ww=np.load(folder/'w.npy',mmap_mode='r');counts={b:sum(r['positive']==b for r in meta['windows']) for b in [False,True]};fractions=[];prior=[]
        for win in meta['windows']:
            if not win['early'] or not win['positive']:continue
            a,b=win['start'],win['end'];positives=yy[a:b]>.5;P=int(positives.sum());m=(.8/counts[True])/float(ww[a:b][positives][0]);fractions.append(P/m)
        median=float(np.median(fractions));gate=bool(len(fractions)>=50 and median<=.03 and panel!='D7_I10_W');gates[panel]=gate;summaries.append(dict(panel=panel,early_positive_windows=len(fractions),median_positive_fraction=median,training_gate_passed=gate))
    atomic_json(ROOT/'objective-preflight.json',dict(passed=True,finite_difference_max_error=error,zero_coefficient_direct_custom_probability_max_error=equivalence,unsubsampled_binary_probability_max_error=binary_error,initial_raw_score=0,reference='both arms same custom objective API, feature_fraction .85 RNG path, initial raw0; coefficient0 control exact BCE',curvature='positive diagonal surrogate, not true nonlocal Hessian; auxiliary IoU coefficient fixed .5; dataset weights applied exactly once',training_gate='June-September positive-window median eligible positive fraction<=.03 and >=50 positive windows; I10W insufficient development phase coverage excluded',by_panel=summaries))
    atomic_json(ROOT/'training-gates.json',dict(strategies=gates));print(json.dumps(dict(passed=True,gradient_error=error,zero_coefficient_equivalence=equivalence,gates=gates)))
if __name__=='__main__':
    try:main()
    except Exception:atomic_json(ROOT/'failure.json',dict(stage='preflight',traceback=traceback.format_exc()));raise
