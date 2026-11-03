"""Independent primal LP check of every conditional demand certificate."""
import common as d
import task4_span_shrink_stress as base
import numpy as np,pandas as pd,msvcrt
from scipy import sparse
from scipy.optimize import linprog

def main():
    out=d.OUT/'task4_budget_dual_bounds';df=pd.read_csv(out/'by_panel.csv');checks=[]
    for (panel,split),group in df.groupby(['panel','split']):
        _,A,c,_,_=base.load(panel,split);m,n=A.shape;Q=sparse.csr_matrix(A);I=sparse.eye(m,format='csr')
        constraints=sparse.vstack([sparse.hstack([Q,-I]),sparse.hstack([-Q,-I]),sparse.hstack([sparse.csr_matrix((1,n)),sparse.csr_matrix(np.ones((1,m)))])],format='csr')
        for row in group.itertuples():
            file=d.WORK/'task4_budget_dual_bounds'/panel/split/f'eps{row.epsilon}.npz';assert d.sha(file)==row.certificate_sha256;y=np.load(file)['y']
            lhs=A.T@y;dual=float(c@y-row.epsilon*c.sum()*np.abs(y).max());assert lhs.max()<=1+1e-12 and abs(dual-row.dual_lower)<1e-7
            rhs=np.r_[c,-c,row.epsilon*c.sum()];primal=linprog(np.r_[np.ones(n),np.zeros(m)],A_ub=constraints,b_ub=rhs,bounds=(0,None),method='highs');assert primal.success,primal.message
            f=primal.x[:n];err=float(np.abs(A@f-c).sum()/c.sum());gap=float(f.sum()-dual);relative_gap=gap/max(1,float(f.sum()))
            assert err<=row.epsilon+1e-7 and f.min()>=-1e-8 and relative_gap>=-1e-7 and relative_gap<1e-6
            assert row.usable_lower<=float(f.sum())+1e-6*max(1,float(f.sum()))
            checks.append(dict(panel=panel,split=split,epsilon=row.epsilon,primal_min_total=float(f.sum()),certificate_total=dual,relative_primal_dual_gap=relative_gap,primal_count_NMAE=err,max_dual_constraint=float(lhs.max())))
    pd.DataFrame(checks).to_csv(out/'independent_primal_checks.csv',index=False)
    d.save(out/'independent_verification.json',dict(cases=len(checks),all_passed=True,max_relative_primal_dual_gap=max(x['relative_primal_dual_gap'] for x in checks),source_sha256=d.sha(__file__),scope='Independent original-variable LP and weak-duality numerical audit, not verified hidden-count compatibility or empirical OD improvement.'))
    d.state(status='stage_complete',stage='budget_dual_primal_verified',next='Reproduce standalone audit before publication; preserve explicit assumptions and numerical, not formal exact-arithmetic, certificate scope.')

if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:main()
        finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
