import pandas as pd
from pathlib import Path
r=Path('outputs/daytime')
o=r/'task4_direct_od_marginal_measurement_gain'
a=pd.read_csv(o/'by_panel.csv')
b=pd.read_csv(r/'task4_direct_od_covariance_shared_mask/by_panel.csv')
pd.testing.assert_frame_equal(a[b.columns],b,atol=1e-12,rtol=0)
cols=['origin_reduction','destination_reduction']
assert ((a[cols]>=-1e-12)&(a[cols]<=1+1e-12)).all().all()
p=a.pivot(index=['panel','selection','law','exponent'],columns='design',values=cols)
delta=p.xs('other',axis=1,level=1)-p.xs('total',axis=1,level=1)
delta.to_csv(o/'paired_marginal.csv')
print(delta.groupby(['selection','law']).mean())
