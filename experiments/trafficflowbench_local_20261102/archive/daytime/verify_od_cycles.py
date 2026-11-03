"""Exercise nontrivial cycle moves and oracle L1 monotonicity/invariants."""
import common as d
import od_cycle_refinement as cycle
import task4_smallnet_posterior_stress as small
import numpy as np


def main():
    rng=np.random.default_rng(2026110326);drifts=[];improvements=[]
    A,_,_,origin,dest=small.geometry(6);lookup=cycle.path_lookup(origin,dest)
    ids=np.array([[lookup[0,2],lookup[0,3],lookup[1,2],lookup[1,3]]])
    f0=np.full(len(origin),5.);b=np.ones(len(origin));truth=f0.copy();truth[ids[0]]+=np.array([1,-1,-1,1])
    f,diag=cycle.refine_cycles(f0,truth,b,ids)
    assert np.array_equal(f,truth) and diag['accepted_updates']==1
    for nodes in [6,9]:
        A,_,_,origin,dest=small.geometry(nodes);lookup=cycle.path_lookup(origin,dest)
        for rep in range(10):
            truth=rng.gamma(2.,5.,len(origin));b=np.maximum(truth+rng.normal(0,4.,len(origin)),1e-6);c=A@truth
            f0,status=small.base.ridge(A,c,b);assert status['success']
            f,diag=cycle.refine_cycles(f0,truth,b,cycle.draw_cycles(lookup,2000,rng))
            assert np.abs(f-truth).sum()<=np.abs(f0-truth).sum()+1e-9
            improvements.append(float(np.abs(f0-truth).sum()-np.abs(f-truth).sum()))
            for _ in range(3):
                reference=rng.gamma(2.,5.,len(origin))
                before=small.base.metrics(f0,reference,b,A,c,dest);after=small.base.metrics(f,reference,b,A,c,dest)
                drift=max(abs(before[k]-after[k]) for k in ['synthetic_S_link','synthetic_S_dev','synthetic_S_attr'])
                assert drift<1e-10;drifts.append(drift)
    result=dict(seed=2026110326,constructive_nonzero_cycle_exact=True,oracle_checks=20,arbitrary_reference_invariant_checks=len(drifts),maximum_non_OD_metric_drift=max(drifts),minimum_oracle_L1_improvement=min(improvements),source_sha256=d.sha(cycle.__file__),verifier_sha256=d.sha(__file__),scope='Oracle truth is used only in these numerical controls, never by research estimators. This proves feasible updates/numerics, not real OD gain or global optimality.')
    d.save(d.OUT/'od_cycle_numerical_verification.json',result);print(result)


if __name__=='__main__':main()
