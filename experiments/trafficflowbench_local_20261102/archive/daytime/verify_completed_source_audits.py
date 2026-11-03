"""Verify completed audit outputs while the disjoint synthetic rerun continues."""
import common as d
import pandas as pd
import numpy as np
import json

for name in ['task4_count_residual_alignment','task4_exact_pair_audit']:
    errors={};rerun=d.WORK/'publication_smoke/source_count_studies'/name
    for file in (d.OUT/name).glob('*.csv'):
        a,b=pd.read_csv(file),pd.read_csv(rerun/file.name)
        assert a.shape==b.shape and a.columns.equals(b.columns),(name,file.name)
        numeric=a.select_dtypes(include='number').columns
        for col in a.columns.difference(numeric):assert a[col].equals(b[col]),(name,file.name,col)
        assert np.array_equal(a[numeric].isna(),b[numeric].isna())
        error=float(np.nanmax(np.abs(a[numeric].to_numpy()-b[numeric].to_numpy())))
        assert error<1e-12,(name,file.name,error)
        errors[file.name]=error
    if name=='task4_exact_pair_audit':
        normalize=lambda values:{key.replace('\\','/').split('task4_exact_pair_models/',1)[1]:value for key,value in values.items()}
        original=normalize(json.loads((d.OUT/name/'model_sha256.json').read_text()));new=normalize(json.loads((rerun/'model_sha256.json').read_text()));assert original==new and len(new)==600
    d.save(d.OUT/name/'completed_audit_portable_verification.json',dict(maximum_CSV_metric_differences=errors,exact_models_hashes_identical=600 if name=='task4_exact_pair_audit' else None,source_sha256=d.sha(__file__),scope='These two completed portable audits verified while a separate subsequent synthetic study continues; no unfinished outputs used. No hidden OD validation.'))
    print(name,errors)
f=pd.read_csv(d.OUT/'task4_exact_pair_audit/paired_comparison.csv')
g=f.groupby(['panel','target_split']).exact_pair_MSE_reduction_vs_geometry.mean()
print('Exact-pair positive road-input groups:',int((g>0).sum()),'of',len(g),'minimum:',float(g.min()))
