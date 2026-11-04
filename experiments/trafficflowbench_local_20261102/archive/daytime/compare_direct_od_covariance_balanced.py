from pathlib import Path
import hashlib,json
import pandas as pd
root=Path('outputs/daytime')
out=root/'task4_direct_od_covariance_balanced'
suffixes={'balanced':'covariance_balanced','joint':'covariance_transfer','diagonal':'reverse_covariance_transfer'}
inputs={name:root/('task4_direct_od_'+suffix)/'by_panel.csv' for name,suffix in suffixes.items()}
table=pd.concat([pd.read_csv(path).assign(selection=name) for name,path in inputs.items()])
keys=['panel','design','law','exponent']
w=table.pivot(index=keys,columns='selection',values='total_reduction')
assert len(w)==200 and not w.isna().any().any()
w['balanced_minus_best_single']=w.balanced-w[['joint','diagonal']].max(axis=1)
w.to_csv(out/'paired_total.csv')
s=w.groupby(['design','law','exponent']).mean()
s.to_csv(out/'paired_total_summary.csv')
(out/'paired_hashes.json').write_text(json.dumps({name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in inputs.items()},indent=2))
print(s.groupby('design').balanced_minus_best_single.agg(['min','max']))
