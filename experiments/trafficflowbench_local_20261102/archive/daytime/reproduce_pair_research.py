"""Coordinate standalone full pair-prior studies and verify weights and tables."""
import common as d
import subprocess
import sys
import json
import msvcrt
import traceback
import pandas as pd
import numpy as np

lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
try:
    out=d.WORK/'publication_smoke/crossmonth_pairs'
    d.state(status='running',stage='crossmonth_pairs_full_portable_reproduction',output=str(out),next='Verify all public-prior folds, coefficients, model hashes and shuffled controls before publishing pairing evidence. Predicting priors is not true OD validation.')
    source=d.WORK/'publish-repo/experiments/trafficflowbench_local_20261102/task4/audit_crossmonth_pairs.py'
    subprocess.run([sys.executable,'-X','utf8',str(source),'--release-root',str(d.DATA),'--output',str(out)],check=True)
    for name,modeldir in [('task4_crossmonth_pair_audit','task4_crossmonth_pair_models'),('task4_pair_marginal_controls','task4_pair_marginal_models')]:
        errors={}
        for file in (d.OUT/name).glob('*.csv'):
            a,b=pd.read_csv(file),pd.read_csv(out/name/file.name)
            assert a.shape==b.shape and a.columns.equals(b.columns),(name,file.name)
            numeric=a.select_dtypes(include='number').columns
            for column in a.columns.difference(numeric):assert a[column].equals(b[column]),(name,file.name,column)
            assert np.array_equal(a[numeric].isna(),b[numeric].isna())
            error=float(np.nanmax(np.abs(a[numeric].to_numpy()-b[numeric].to_numpy())))
            assert error<1e-12,(name,file.name,error)
            errors[file.name]=error
        original=json.loads((d.OUT/name/'model_sha256.json').read_text());rerun=json.loads((out/name/'model_sha256.json').read_text())
        normalize=lambda values:{k.replace('\\','/').split(modeldir+'/',1)[1]:v for k,v in values.items()}
        assert normalize(original)==normalize(rerun),(name,'model hashes')
        d.save(d.OUT/name/'portable_reproduction.json',dict(all_CSV_maximum_metric_differences=errors,all_model_file_hashes_identical=True,model_files=len(original),runner_sha256=d.sha(__file__),scope='Public weak-prior prediction and negative controls only; hidden OD truth unobserved.'))
    d.state(status='stage_complete',stage='crossmonth_pairs_full_portable_reproduction_complete',next='Publish verified pair-prior predictability and marginal/shuffled controls with all negative roads; then stress shared-noise and changing-demand alternatives before any OD candidate.')
    print('PAIR STUDIES FULLY REPRODUCED',flush=True)
except Exception as error:
    d.state(status='failed',stage='crossmonth_pairs_full_portable_reproduction',error=repr(error),next='Preserve fixed folds, all failures and original results; repair portability without selecting roads.')
    traceback.print_exc();raise
finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
