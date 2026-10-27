import os,sys,json,traceback,hashlib
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
from pathlib import Path
import numpy as np
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent;PARENT=ROOT.parent/'task2-round30-custom-iou-control';sys.path.insert(1,str(PARENT));sys.path.insert(2,str(ROOT.parent/'task2-round2'))
from task2q.cache import atomic_json
from balanced_objective import BalancedObjective,sigmoid
from window_objective import WindowObjective
def main():
    assert json.loads((PARENT/'inference-reproduction.json').read_text(encoding='utf8'))['passed']
    y=np.array([1,0,0,1,0,0,0],float);w=np.array([.08,.16,.16,.1,.3,.08,.12]);windows=[dict(start=0,end=3,positive=True),dict(start=3,end=5,positive=True),dict(start=5,end=7,positive=False)];z=np.array([-.8,.2,.5,.4,-.2,.8,-.5]);obj=BalancedObjective(y,w,windows)
    g,h=obj.gradient(z);eps=1e-5;fd=[]
    for i in range(len(z)):
        a=z.copy();b=z.copy();a[i]+=eps;b[i]-=eps;fd.append((obj.loss(a)-obj.loss(b))/(2*eps))
    err=float(np.max(np.abs(g-fd)));assert err<1e-7 and (h>0).all()
    exact=BalancedObjective(y,w,windows,coefficient=0);g,h=exact.gradient(z);hfd=[]
    for i in range(len(z)):
        a=z.copy();b=z.copy();a[i]+=eps;b[i]-=eps;hfd.append((exact.gradient(a)[0][i]-exact.gradient(b)[0][i])/(2*eps))
    herr=float(np.max(np.abs(h-hfd)));assert herr<1e-7
    rng=np.random.default_rng(20261027);X=rng.normal(size=(4096,16));Y=(X[:,0]+.4*X[:,2]+rng.normal(size=4096)>.8).astype(float);Y[-512:]=0;W=np.ones(len(Y));W[:-512]*=.8/(len(Y)-512);W[-512:]*=.2/512;ww=[dict(start=0,end=len(Y)-512,positive=True),dict(start=len(Y)-512,end=len(Y),positive=False)]
    comparisons={}
    for coeff in [0.,.5]:
        old=WindowObjective(Y,W,ww,coefficient=coeff);new=BalancedObjective(Y,W,ww,coefficient=coeff,negative_coefficient=0)
        params=dict(num_leaves=31,learning_rate=.035,min_data_in_leaf=80,feature_fraction=.85,lambda_l2=4,num_threads=4,verbosity=-1,seed=20261027,deterministic=True,force_col_wise=True,boost_from_average=False)
        ds=lambda:lgb.Dataset(X,label=Y,weight=old.w)
        a=lgb.train(dict(params,objective=old),ds(),num_boost_round=7);b=lgb.train(dict(params,objective=new),ds(),num_boost_round=7)
        delta=float(np.max(np.abs(a.predict(X,raw_score=True)-b.predict(X,raw_score=True))));assert delta<1e-12;comparisons[str(coeff)]=delta
    gates=json.loads((PARENT/'training-gates.json').read_text(encoding='utf8'));atomic_json(ROOT/'training-gates.json',gates)
    report=dict(passed=True,gradient_finite_difference_max_error=err,negative_term_diagonal_hessian_max_error=herr,negative_coefficient_zero_parent_model_raw_score_max_errors=comparisons,positive_coefficient=.5,negative_coefficient=.5,tau=.25,initial_raw_score=0,no_coefficient_search=True,scope='sampled-negative IPW-normalized log-mean-exp margin tail, not full-cell max or calibrated event probability',parent_frozen_sha256=hashlib.sha256((PARENT/'frozen-calibration.json').read_bytes()).hexdigest())
    atomic_json(ROOT/'objective-preflight.json',report);print(json.dumps(report))
if __name__=='__main__':
    try:main()
    except Exception:atomic_json(ROOT/'failure.json',dict(stage='preflight',traceback=traceback.format_exc()));raise
