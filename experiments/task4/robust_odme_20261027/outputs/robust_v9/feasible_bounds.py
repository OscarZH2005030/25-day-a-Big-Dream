# 中文说明：只用公开计数给出条件性偏离范围；不把LP可行解当作隐藏真值。
import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import sys,time,warnings
from pathlib import Path
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O.parent));from run_task4 import REL,load,load_operator
import numpy as np,pandas as pd
from scipy import sparse
from scipy.optimize import linprog,OptimizeWarning

def lp(c,Aub,bub):
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',OptimizeWarning)
  r=linprog(c,A_ub=Aub,b_ub=bub,bounds=(0,None),method='highs',options={'threads':4,'time_limit':60.})
 if not r.success:raise RuntimeError(r.message)
 return r

def bounds(A,c,b,tol):
 n=len(b);I=sparse.eye(n,format='csr');Z=sparse.csr_matrix(A.shape);AB=sparse.vstack([A,-A],format='csr');cb=np.r_[c*(1+tol),-c*(1-tol)]
 M=sparse.vstack([sparse.hstack([I,-I]),sparse.hstack([-I,-I]),sparse.hstack([AB,sparse.csr_matrix((2*A.shape[0],n))])],format='csr')
 lo=lp(np.r_[np.zeros(n),np.ones(n)],M,np.r_[b,-b,cb]);high=lp(-np.ones(n),AB,cb)
 return float(lo.fun),float(b.sum()-high.fun),float(-high.fun)
if __name__=='__main__':
 p='D12_I405_S';ids,ls,D,paths=load_operator(REL/'corridors'/p/'network');A,c,b,t,_=load(p,'train',ids,ls,D);start=time.time();print(bounds(A,c,b,.01),time.time()-start)
