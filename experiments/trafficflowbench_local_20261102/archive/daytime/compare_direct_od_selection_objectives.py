from pathlib import Path
import hashlib
import json
import pandas as pd

root=Path('outputs/daytime')
paths=[root/f'task4_direct_od_{name}'/'by_panel.csv'
       for name in ['variance_interpolation','unselected_selection']]
keys=['panel','design','exponent']
base,change=map(pd.read_csv,paths)
paired=base.merge(change,on=keys,validate='one_to_one',suffixes=('_total_policy','_other_policy'))
assert len(base)==len(change)==len(paired)==200
for metric in ['total_reduction','unselected_reduction']:
    paired[metric+'_delta']=paired[metric+'_other_policy']-paired[metric+'_total_policy']
out=paths[1].parent
paired.to_csv(out/'paired_by_panel.csv',index=False)
summary=paired.groupby(['design','exponent'])[['total_reduction_delta','unselected_reduction_delta']].mean()
summary.to_csv(out/'paired_summary.csv')
(out/'paired_input_sha256.json').write_text(json.dumps({p.parent.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))
print(summary.to_string())
