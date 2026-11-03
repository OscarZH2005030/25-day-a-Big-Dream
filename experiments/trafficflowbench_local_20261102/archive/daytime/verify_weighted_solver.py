"""Independent primal BVLS checks of the weighted dual solver on small networks."""
import common as d
import task4_smallnet_posterior_stress as small
import task4_weighted_variance_stress as weighted
import task4_od_only_direction_stress as safeguarded
import numpy as np
from scipy.optimize import lsq_linear

rng = np.random.default_rng(2026110314)
errors, gradients, invariants = [], [], []
for nodes in [6,9]:
    A, _, _, origin, dest = small.geometry(nodes)
    for rep in range(10):
        truth = rng.gamma(1.2, 5., A.shape[1]) * (rng.random(A.shape[1]) > .35)
        b = np.maximum(truth + rng.normal(0, 5, len(truth)), 1e-6)
        c = A @ truth
        weight = np.exp(rng.uniform(-4,4,len(b)))
        prediction, success = weighted.ridge(A,c,b,weight)
        assert success
        penalty = np.sqrt(.05/weight)
        matrix = np.vstack([A,np.diag(penalty)])
        target = np.r_[c,penalty*b]
        independent = lsq_linear(matrix,target,bounds=(0,np.inf),method='bvls',tol=1e-12,max_iter=500)
        assert independent.success
        error = float(np.max(np.abs(prediction-independent.x)))
        assert error < 1e-4
        errors.append(error)
        g = A.T@(A@prediction-c)+.05*(prediction-b)/weight
        kkt = float(max(np.max(np.abs(g[prediction>1e-8]),initial=0),np.max(np.maximum(-g[prediction<=1e-8],0),initial=0)))
        assert kkt < 1e-5
        gradients.append(kkt)
        f0, status = small.base.ridge(A,c,b)
        assert status['success']
        safe, diagnostics = safeguarded.safeguard(f0,prediction,b,A,origin,dest)
        for _ in range(10):
            alternate_reference = rng.gamma(1.2,5.,len(b))
            a = small.base.metrics(f0,alternate_reference,b,A,c,dest)
            z = small.base.metrics(safe,alternate_reference,b,A,c,dest)
            drift = max(abs(a[k]-z[k]) for k in ['synthetic_S_link','synthetic_S_dev','synthetic_S_attr'])
            assert drift < 1e-10
            invariants.append(drift)
d.save(d.OUT/'task4_weighted_variance_stress/independent_solver_verification.json',dict(seed=2026110314,cases=20,independent_solver='SciPy primal augmented bounded least squares, BVLS',maximum_primal_solution_difference=max(errors),maximum_KKT_violation=max(gradients),arbitrary_reference_invariant_checks=len(invariants),maximum_non_OD_metric_drift=max(invariants),source_sha256=d.sha(__file__),scope='Numerical correctness only; no OD improvement guarantee.'))
print('20 independent solver checks; 200 arbitrary-reference invariant checks:',max(errors),max(gradients),max(invariants))
