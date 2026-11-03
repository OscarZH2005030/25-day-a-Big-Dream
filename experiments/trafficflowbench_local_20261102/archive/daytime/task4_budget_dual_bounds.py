"""Count-compatible demand lower certificates for conditional OD movement budgets."""
import common as d
import task4_span_shrink_stress as base
import numpy as np,pandas as pd,msvcrt
from scipy import sparse
from scipy.optimize import linprog
from datetime import datetime,timezone
O=d.OUT/'task4_budget_dual_bounds'

def main():
    O.mkdir(parents=True,exist_ok=True)
    d.save(O/'protocol.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),seed=None,deterministic=True,assumed_count_NMAE=[0.,.01,.05],splits=['train','validation','private'],hypothesis='Exploit released count geometry to tighten the crude total-demand lower bound used in conditional movement budgets.',certificate='For any y with A.T@y <= 1 and nonnegative reference f with ||Af-c||1 <= eps*sum(c), sum(f) >= c@y-eps*sum(c)*max(abs(y)). Optimize y then shrink for numerical feasibility, retaining crude bound as fallback.',scope='Conditional deterministic geometry audit, not OD validation, statistical coverage, official safety or new candidate.',source_sha256=d.sha(__file__)))
    rows=[];hashes={}
    for panel in d.night.predictor.load_routes():
        for split in ['train','validation','private']:
            paths,A,c,_,inputs=base.load(panel,split);hashes.update(inputs);m,n=A.shape;At=sparse.csr_matrix(A.T);I=sparse.eye(m,format='csr');ones=sparse.csr_matrix(-np.ones((m,1)))
            constraints=sparse.vstack([sparse.hstack([At,-At,sparse.csr_matrix((n,1))]),sparse.hstack([I,-I,ones]),sparse.hstack([-I,I,ones])],format='csr');rhs=np.r_[np.ones(n),np.zeros(2*m)]
            for eps in [0.,.01,.05]:
                cost=np.r_[-c,c,eps*c.sum()];solution=linprog(cost,A_ub=constraints,b_ub=rhs,bounds=(0,None),method='highs');assert solution.success,solution.message
                y=solution.x[:m]-solution.x[m:2*m];y/=max(1.,float(np.max(A.T@y)))*(1+1e-10)
                certified=float(c@y-eps*c.sum()*np.abs(y).max());crude=(1-eps)*float(c.sum())/float(A.sum(axis=0).max());lower=max(crude,certified)
                assert float(np.max(A.T@y))<=1+1e-12
                folder=d.WORK/'task4_budget_dual_bounds'/panel/split;folder.mkdir(parents=True,exist_ok=True);file=folder/f'eps{eps}.npz';np.savez_compressed(file,y=y,epsilon=eps)
                rows.append(dict(panel=panel,split=split,epsilon=eps,crude_lower=crude,dual_lower=certified,usable_lower=lower,capacity_ratio=lower/crude,max_dual_constraint=float(np.max(A.T@y)),optimizer_objective=float(-solution.fun),certificate_sha256=d.sha(file)))
            pd.DataFrame(rows).to_csv(O/'by_panel.csv',index=False)
    frame=pd.DataFrame(rows);frame.groupby('epsilon').capacity_ratio.agg(['min','mean','max']).to_csv(O/'summary.csv');d.save(O/'input_sha256.json',hashes)
    d.state(status='stage_complete',stage='budget_dual_bounds_complete',next='Independently audit dual feasibility and weak duality against primal references before publication. Tighter conditional capacity is not actual OD improvement; no hidden count compatibility verified.')

if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:main()
        finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
