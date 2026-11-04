import pandas as pd
from pathlib import Path
o=Path('outputs/daytime')
b=o/'task4_direct_od_variance_interpolation'
a=pd.read_csv(b/'by_panel.csv')
old=pd.read_csv(o/'task4_direct_od_variance_transfer/by_panel.csv')
keys=['panel','design','exponent']
assert len(a)==200 and not a.duplicated(keys).any()
pd.testing.assert_frame_equal(a[a.exponent.isin(old.exponent)].sort_values(keys).reset_index(drop=True),old.sort_values(keys).reset_index(drop=True),atol=1e-12,rtol=0)
w=a[a.exponent.isin([.25,.75])].pivot(index=['panel','exponent'],columns='design',values='total_reduction')
r=w.max(axis=1).to_numpy()[:,None]-w
r.groupby('panel').max().to_csv(b/'interpolation_regret_by_panel.csv')
print(r.groupby('panel').max().mean())
