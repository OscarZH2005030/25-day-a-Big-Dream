from pathlib import Path
import numpy as np
import pandas as pd
o=Path('outputs/daytime')
out=o/'task4_direct_od_fixed_noise'
a=pd.read_csv(out/'by_panel.csv')
b=pd.read_csv(o/'task4_direct_od_common_support/by_panel.csv')
keys=['panel','design','exponent']
assert len(a)==300 and not a.duplicated(keys+['noise']).any()
m=a.query('noise==0').merge(b,on=keys,validate='one_to_one')
assert len(m)==100
for col in ['total_reduction','common_reduction']:
    np.testing.assert_allclose(m[col+'_x'],m[col+'_y'],rtol=0,atol=1e-12)
    assert (np.diff(a.pivot(index=keys,columns='noise',values=col).to_numpy(),axis=1)<=1e-12).all()
p=a.pivot(index=['panel','exponent','noise'],columns='design',values='common_reduction')
p['delta']=p.other-p.total
p.to_csv(out/'paired_common.csv')
print(p.groupby(['exponent','noise']).delta.mean().unstack())
