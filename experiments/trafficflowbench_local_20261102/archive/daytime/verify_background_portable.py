"""Compare all portable background-mean table rows with the original audit."""
import common as d
import pandas as pd
import numpy as np
import json

source = d.OUT / 'task4_background_mean_audit'
rerun = d.WORK / 'publication_smoke/background_mean/audit'
errors = {}
for name in ['family_cv_by_panel.csv', 'family_cv_summary.csv', 'transfer_by_panel.csv', 'transfer_summary.csv']:
    a, b = pd.read_csv(source / name), pd.read_csv(rerun / name)
    assert a.shape == b.shape and a.columns.equals(b.columns)
    columns = a.select_dtypes(include='number').columns
    for column in a.columns.difference(columns):
        assert a[column].equals(b[column]), (name, column)
    error = float(np.max(np.abs(a[columns].to_numpy() - b[columns].to_numpy())))
    assert error < 1e-12, (name, error)
    errors[name] = error
for name in ['family_cv_models.json', 'full_train_models.json']:
    a = json.loads((source / name).read_text())
    b = json.loads((rerun / name).read_text())
    if name == 'full_train_models.json':
        a, b = a['models'], b['models']
    assert a == b, name
d.save(source / 'portable_reproduction.json', dict(maximum_metric_absolute_errors=errors, all_rows_and_models_compared=True, verifier_sha256=d.sha(__file__), scope='Public count residual model checks only; no hidden OD scoring.'))
print(errors)
