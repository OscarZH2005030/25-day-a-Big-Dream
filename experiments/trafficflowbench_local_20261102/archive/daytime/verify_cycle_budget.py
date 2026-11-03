"""Independent conditional-bound readback and tightness/necessity controls."""
import common as d
import task4_span_shrink_stress as base
import pandas as pd,numpy as np,hashlib,msvcrt

def score(f,t,b,A,c,dest):
    od=max(0,1-float(np.abs(f-t).sum()/t.sum()));link=max(0,1-float(np.abs(A@f-c).sum()/c.sum()))
    dev=float(np.exp(-abs(float(np.abs(f-b).sum()/max(np.abs(t-b).sum(),1e-12))-1)))
    a=np.bincount(dest,weights=f);a/=a.sum();target=np.bincount(dest,weights=t);target/=target.sum();attr=1-.5*float(np.abs(a-target).sum())
    return od,.45*od+.25*link+.15*dev+.15*attr

def main():
    out=d.OUT/'task4_cycle_budget_check';df=pd.read_csv(out/'all_budget_checks.csv');errors=[];bounds=[];artifacts={}
    for panel,group in df.groupby('panel'):
        paths,A,_,_,_=base.load(panel,'train');dest=pd.factorize(paths.destination_zone,sort=True)[0];K=float(A.sum(axis=0).max())
        for case,rows in group.groupby('case'):
            folder=d.WORK/'task4_od_cycle_cases'/hashlib.sha256(case.encode()).hexdigest()[:16]
            z=np.load(folder/'inputs.npz');pred=np.load(folder/'predictions.npz');t=z['artificial_truth'][2];b=z['artificial_prior'][2];c=z['observed_counts'][2];f0=pred['baseline'];s0=score(f0,t,b,A,c,dest)
            for name in ['inputs.npz','predictions.npz']:artifacts[str(folder.relative_to(d.ROOT))+'/'+name]=d.sha(folder/name)
            for row in rows.itertuples():
                delta=pred[row.method]-f0;lower=(1-row.epsilon)*float(c.sum())/K;blend=min(1.,row.rho*lower/max(float(np.abs(delta).sum()),1e-12));f=f0+blend*delta;s1=score(f,t,b,A,c,dest)
                error=max(abs(blend-row.blend),abs(s1[0]-s0[0]-row.OD_delta),abs(s1[1]-s0[1]-row.Task4_delta));errors.append(error)
                compatible=float(np.abs(A@t-c).sum()/c.sum())<=row.epsilon+1e-12
                assert compatible==row.count_compatible
                if compatible:assert abs(s1[0]-s0[0])<=row.rho+1e-9 and abs(s1[1]-s0[1])<=.45*row.rho+1e-9
                bounds.append(abs(s1[1]-s0[1])/(.45*row.rho))
    assert len(df)==3840 and max(errors)<1e-12
    # Tight compatible control: every column contributes once, reference total equals lower bound.
    A=np.ones((1,4));dest=np.array([0,1,0,1]);truth=np.array([2.,1.,1.,2.]);b=np.zeros(4);c=np.array([6.]);rho=.001;v=np.array([-1.,1.,1.,-1.]);theta=rho*6/4;f=truth+theta*v
    q0=score(truth,truth,b,A,c,dest);q1=score(f,truth,b,A,c,dest);loss=q0[1]-q1[1];assert abs(loss-.45*rho)<1e-12
    # Same geometry/prior deviation and counts, opposite unknown matching reference preferences.
    q2=score(truth,f,b,A,c,dest);q3=score(f,f,b,A,c,dest);assert abs((q3[1]-q2[1])-loss)<1e-12
    # Without compatible counts, the same observable-count budget is not an OD bound.
    tiny=.01*truth;f_bad=tiny+theta*v;assert f_bad.min()>0
    bad0=score(tiny,tiny,b,A,c,dest);bad1=score(f_bad,tiny,b,A,c,dest);od_loss=bad0[0]-bad1[0];assert od_loss>rho
    d.save(out/'independent_verification.json',dict(rows=len(df),maximum_readback_error=max(errors),maximum_observed_loss_to_conditional_radius_ratio=max(bounds),input_prediction_hashes=artifacts,compatible_tight_control_Task4_loss=loss,promised_radius=.45*rho,opposite_reference_preference_equal_magnitude=True,incompatible_count_control_OD_loss=od_loss,incompatible_count_actual_NMAE=float(np.abs(A@tiny-c).sum()/c.sum()),source_sha256=d.sha(__file__),scope='Mathematical sensitivity and artificial numerical controls. Bound can be tight; no unverified hidden-count assumption, empirical improvement or confidence interval is claimed.'))
    d.state(status='stage_complete',stage='cycle_budget_independently_verified',next='Review conditional bound, sharp compatible control and incompatible counterexample before publication. Continue justified OD structure research; official baseline remains frozen.')

if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:main()
        finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
