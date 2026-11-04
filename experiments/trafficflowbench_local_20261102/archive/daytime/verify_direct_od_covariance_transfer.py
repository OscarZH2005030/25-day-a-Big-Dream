from pathlib import Path
import numpy as np
import pandas as pd
root=Path('outputs/daytime')
out=root/'task4_direct_od_covariance_transfer'
a=pd.read_csv(out/'by_panel.csv')
b=pd.read_csv(root/'task4_direct_od_dual_covariance/by_panel.csv')
keys=['panel','design','exponent']
assert len(a)==200 and not a.duplicated(keys+['law']).any()
m=a.query("law=='joint'").merge(b,on=keys,validate='one_to_one')
assert len(m)==100
for metric in ['total_reduction','common_reduction']:
    np.testing.assert_allclose(m[metric+'_x'],m[metric+'_y'],atol=1e-12,rtol=0)
p=a.pivot(index=['panel','exponent','law'],columns='design',values='common_reduction')
p['delta']=p.other-p.total
p.to_csv(out/'paired_common.csv')
print(p.groupby(['exponent','law']).delta.mean().unstack())
