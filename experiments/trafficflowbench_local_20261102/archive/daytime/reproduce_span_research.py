"""Full isolated reproduction of four registered span uncertainty studies."""
import common as d
import subprocess
import sys
import msvcrt
import traceback
import pandas as pd
import numpy as np
import json

lock = (d.ROOT/'work/overnight/research.lock').open('r+b')
lock.seek(0)
msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
try:
    out = d.WORK/'publication_smoke/span_uncertainty'
    d.state(status='running',stage='span_uncertainty_full_portable_reproduction',output=str(out),next='Recompute all 30 public QR inputs, both likelihood audits, 1200 weighted cases and 192 OD-only cases; verify before publishing, no Kaggle upload.')
    source = d.WORK/'publish-repo/experiments/trafficflowbench_local_20261102/task4/audit_span_uncertainty.py'
    subprocess.run([sys.executable,'-X','utf8',str(source),'--release-root',str(d.DATA),'--output',str(out),'--with-stress','--with-od-directions'],check=True)
    names = ['task4_span_variance_audit','task4_variance_controls_audit','task4_weighted_variance_stress','task4_od_only_direction_stress']
    verification = {}
    for name in names:
        errors = {}
        for original in (d.OUT/name).glob('*.csv'):
            rerun = out/name/original.name
            a,b = pd.read_csv(original),pd.read_csv(rerun)
            assert a.shape == b.shape and a.columns.equals(b.columns),(name,original.name)
            numeric = a.select_dtypes(include='number').columns
            for col in a.columns.difference(numeric):
                assert a[col].equals(b[col]),(name,original.name,col)
            assert np.array_equal(a[numeric].isna(),b[numeric].isna())
            error = float(np.nanmax(np.abs(a[numeric].to_numpy()-b[numeric].to_numpy()))) if len(numeric) else 0.
            assert error < 1e-12,(name,original.name,error)
            errors[original.name] = error
        if (d.OUT/name/'full_train_models.json').exists():
            a=json.loads((d.OUT/name/'full_train_models.json').read_text())['models']
            b=json.loads((out/name/'full_train_models.json').read_text())['models']
            assert a==b,(name,'full_train_models')
        verification[name]=errors
        d.save(d.OUT/name/'portable_reproduction.json',dict(all_CSV_maximum_metric_differences=errors,all_rows_and_missingness_identical=True,full_train_models_identical=True if (d.OUT/name/'full_train_models.json').exists() else None,runner_sha256=d.sha(__file__),scope='Reproduction of public-input diagnostics/artificial scores, not official OD validation.'))
    d.save(d.OUT/'span_uncertainty_reproduction.json',verification)
    d.state(status='stage_complete',stage='span_uncertainty_full_reproduction_complete',next='Publish verified span likelihood support alongside extensive weighted OD and invariant-direction harms. Continue researching identifiable OD structure; official Task4 baseline unchanged.')
    print('FOUR STUDIES REPRODUCED',verification,flush=True)
except Exception as error:
    d.state(status='failed',stage='span_uncertainty_full_portable_reproduction',error=repr(error),next='Preserve failure; repair portability without changing fixed models or case recipes.')
    traceback.print_exc()
    raise
finally:
    lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
    lock.close()
