"""Check the new nonsmooth dual against an independent epigraph QP."""
from run_bias import elastic,np,O,json
from scipy import sparse
from scipy.optimize import minimize,LinearConstraint,Bounds
checks=[]
for seed in [12,13,14]:
 rng=np.random.default_rng(seed);Q=rng.integers(0,2,(5,12)).astype(float);b=rng.uniform(2,15,12);c=Q@rng.uniform(0,20,12)*1.3;lam=.05;strength=1.;eta=lam*np.median(b)*strength
 x=elastic(sparse.csr_matrix(Q),c,b,strength);n=len(b)
 def fg(v):
  z=v[:n];r=Q@z-c
  return .5*r@r+.5*lam*np.dot(z-b,z-b)+eta*v[n:].sum(),np.r_[Q.T@r+lam*(z-b),np.full(n,eta)]
 M=np.r_[np.c_[np.eye(n),-np.eye(n)],np.c_[-np.eye(n),-np.eye(n)]];upper=np.r_[b,-b]
 r=minimize(fg,np.r_[b,np.zeros(n)],jac=True,method='SLSQP',bounds=Bounds(0,np.inf),constraints=LinearConstraint(M,-np.inf,upper),options={'maxiter':2000,'ftol':1e-10});assert r.success,r.message
 err=float(abs(x-r.x[:n]).max());objdiff=float(abs(fg(np.r_[x,abs(x-b)])[0]-r.fun));assert err<1e-4 and objdiff<1e-7,(err,objdiff)
 checks.append({'seed':seed,'max_flow_difference':err,'objective_difference':objdiff})
(O/'solver_validation.json').write_text(json.dumps({'independent_solver':'SLSQP quadratic epigraph with linear absolute-value constraints','checks':checks},indent=2),encoding='utf-8');print(checks)
