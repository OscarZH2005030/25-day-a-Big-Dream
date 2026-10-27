"""Project an informed path estimate onto fixed link and destination totals."""
import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import numpy as np,pandas as pd
from scipy import sparse
from scipy.linalg import eigh,solve
from scipy.optimize import minimize

class EqualProjector:
 def __init__(self,A,dest):
  di=pd.factorize(dest)[0];n=A.shape[1];B=sparse.csr_matrix((np.ones(n),(di,np.arange(n))),shape=(di.max()+1,n));self.M=sparse.vstack([A,B],format='csr');M=self.M.toarray();v,U=eigh(M@M.T);keep=v>max(v.max()*1e-11,1e-10);self.Q=(U[:,keep]/np.sqrt(v[keep])).T@M
 def project(self,q,ref):
  scale=max(float(np.mean(ref)),1.);p=q/scale;rhs=self.Q@ref/scale;Q=self.Q
  def fg(y):
   x=np.maximum(p-Q.T@y,0);return .5*x@x+rhs@y,rhs-Q@x,x
  r=minimize(lambda y:fg(y)[:2],np.zeros(len(rhs)),jac=True,method='L-BFGS-B',options={'maxiter':1500,'ftol':1e-15,'gtol':1e-11,'maxls':60});y=r.x
  for it in range(100):
   val,g,x=fg(y)
   if abs(g).max()<2e-12:break
   H=(Q*(x>0))@Q.T+np.eye(len(rhs))*1e-10;step=solve(H,-g,assume_a='pos');rate=1.
   for _ in range(50):
    if np.linalg.norm(fg(y+rate*step)[1])<np.linalg.norm(g):break
    rate*=.5
   y+=rate*step
  _,g,x=fg(y);x=x*scale;err=float(abs(self.M@(x-ref)).max());assert err<2e-6,(err,r.message,float(abs(g).max()))
  return x,err

if __name__=='__main__':
 from pathlib import Path
 import json
 out=[]
 for seed in [1501,1502,1503]:
  rng=np.random.default_rng(seed);A=sparse.csr_matrix(rng.integers(0,2,(3,12)));dest=np.repeat(np.arange(3),4);ref=rng.uniform(1,20,12);q=rng.uniform(0,40,12);p=EqualProjector(A,dest);x,err=p.project(q,ref);M=p.M.toarray()
  r=minimize(lambda z:(.5*np.sum((z-q)**2),z-q),ref,jac=True,bounds=[(0,None)]*12,constraints={'type':'eq','fun':lambda z:M@(z-ref),'jac':lambda z:M},method='SLSQP',options={'maxiter':1000,'ftol':1e-10});assert r.success,r.message;diff=float(abs(x-r.x).max());assert diff<1e-4,(diff,err);out.append({'seed':seed,'constraint_error':err,'independent_SLSQP_max_difference':diff})
 (Path(__file__).resolve().parent/'solver_validation.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(out)
