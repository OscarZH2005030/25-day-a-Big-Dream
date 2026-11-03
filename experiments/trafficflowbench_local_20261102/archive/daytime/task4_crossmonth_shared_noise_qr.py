"""Numerically equivalent augmented-QR rerun; preserve the ill-conditioned attempt."""
import common as d
import task4_crossmonth_shared_noise_stress as original
import numpy as np
import pandas as pd
import types
import shutil
import msvcrt
import traceback
import os
import time
from scipy.linalg import lstsq, solve


def ridge_qr(X,y):
    # Minimize ||X*coef-y||^2 + sum(coef[1:]^2), exactly as before.
    penalty=np.eye(X.shape[1]);penalty[0,0]=0
    augmented=np.vstack([X,penalty]);target=np.r_[y,np.zeros(X.shape[1])]
    scales=np.maximum(np.linalg.norm(augmented,axis=0),1e-12)
    solution,_,rank,_=lstsq(augmented/scales,target,cond=1e-12,lapack_driver='gelsy')
    assert rank==X.shape[1]
    return solution/scales


def forecasts(X,priors,rng):
    source=np.mean([v/max(float(np.median(v[v>0])),1e-9) for v in priors[:2]],axis=0)
    residual=source-X@ridge_qr(X,source)
    pair=np.column_stack([X,residual]);b=priors[2]
    order=rng.permutation(len(b));fold=np.empty(len(b),int);fold[order]=np.arange(len(b))%5
    predictions={'geometry':np.zeros(len(b)),'pair':np.zeros(len(b))};coefs={'geometry':[],'pair':[]};scales=[]
    for k in range(5):
        train=fold!=k;test=~train;scale=max(float(np.median(b[train&(b>0)])),1e-9);scales.append(scale)
        for name,design in [('geometry',X),('pair',pair)]:
            coef=ridge_qr(design[train],b[train]/scale)
            predictions[name][test]=np.maximum(design[test]@coef,0)*scale;coefs[name].append(coef)
    gmse=float(np.mean((predictions['geometry']-b)**2));pmse=float(np.mean((predictions['pair']-b)**2))
    return predictions,coefs,np.array(scales),fold,1-pmse/max(gmse,1e-12)


def main():
    lock=(d.ROOT/'work/overnight/research.lock').open('r+b')
    queue=d.WORK/'shared_noise_qr_retry_queue.json'
    d.save(queue,dict(pid=os.getpid(),status='waiting_for_shared_lock',seed=original.SEED,source_sha256=d.sha(__file__),no_duplicate_research=True))
    start=time.monotonic()
    while True:
        try:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);break
        except OSError:
            if time.monotonic()-start>1800:raise TimeoutError('Known first attempt still holds lock; retain queue for review')
            time.sleep(1)
    try:
        d.save(queue,dict(pid=os.getpid(),status='running',seed=original.SEED))
        out=d.OUT/'task4_crossmonth_shared_noise_stress_qr';work=d.WORK/'shared_noise_qr_retry';work.mkdir(parents=True,exist_ok=True)
        rng=np.random.default_rng(2026110318);errors=[]
        for _ in range(20):
            X=np.column_stack([np.ones(80),rng.normal(size=(80,7))]);y=rng.normal(size=80);penalty=np.eye(8);penalty[0,0]=0
            errors.append(float(np.max(np.abs(ridge_qr(X,y)-solve(X.T@X+penalty,X.T@y,assume_a='pos')))))
        assert max(errors)<1e-12
        d.save(out/'numerical_repair.json',dict(reason='First fixed-seed attempt produced normal-equation LinAlgWarning near rcond1e-19 in sparse/artificial floor cases; its source, results and log remain unchanged and cannot be ranked before this check.',scientific_change=False,implementation='Same ridge1 objective, unpenalized intercept, same five folds/source scales/clipping/alphas/seed; solve augmented least squares with column equilibration and pivoted QR instead of squaring condition number.',new_implementation_sha256=d.sha(__file__),original_source_sha256=d.sha(original.__file__),well_conditioned_comparison_cases=20,maximum_coefficient_error=max(errors),scope='Numerical repair is not a new positive scientific result or hidden OD validation.'))
        for name in ['task4_prior_structure.py','task4_span_shrink_stress.py']:shutil.copy2(d.WORK/name,work/name)
        stub=types.SimpleNamespace(**{key:value for key,value in vars(d).items() if not key.startswith('__')});stub.WORK=work
        stub.state=lambda **kw:d.state(**kw,numerical_repair='augmented_qr',retry_work=str(work))
        original.d=stub;original.O=out;original.forecasts=forecasts
        d.state(status='running',stage='shared_noise_qr_retry',next='Rerun same 960 cases with equivalent QR ridge; preserve first ill-conditioned attempt. No official score or candidate claim.')
        original.main()
        # Prove the generation and evaluation folds were unchanged by the repair.
        old=d.WORK/'task4_crossmonth_shared_noise_cases';new=work/'task4_crossmonth_shared_noise_cases'
        count=0
        for file in new.glob('*.npz'):
            a,b=np.load(old/file.name),np.load(file)
            for key in ['artificial_truth','artificial_prior','fold','seed']:
                assert np.array_equal(a[key],b[key]),(file.name,key)
            count+=1
        assert count==960
        d.save(out/'generation_identity.json',dict(cases=count,all_truth_prior_fold_seed_arrays_identical=True,original_attempt_preserved=True,source_sha256=d.sha(__file__)))
        d.save(queue,dict(pid=os.getpid(),status='complete',seed=original.SEED,cases=count))
        d.state(status='stage_complete',stage='shared_noise_qr_retry_complete',next='Evaluate corrected results and compare with original numerical attempt; all 960 latent/prior/fold/seed arrays identical. Then verify portability and retain shared-noise/drift harms, no Kaggle upload.')
    except Exception as error:
        d.state(status='failed',stage='shared_noise_qr_retry',error=repr(error),next='Preserve original and repaired attempts; inspect rank/scaling without changing synthetic cases or strengths.');traceback.print_exc();raise
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':main()
