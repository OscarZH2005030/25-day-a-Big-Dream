from pathlib import Path
import numpy as np
import pandas as pd
o=Path('outputs/daytime')
a=pd.read_csv(o/'task4_direct_od_common_support/by_panel.csv')
assert len(a)==100
for policy,source in [('total','variance_interpolation'),('other','unselected_selection')]:
    b=pd.read_csv(o/f'task4_direct_od_{source}/by_panel.csv').query("design=='balanced'")
    m=a.query('design==@policy').merge(b,on=['panel','exponent'],validate='one_to_one')
    assert len(m)==50
    for metric in ['total_reduction','unselected_reduction']:
        np.testing.assert_allclose(m[metric+'_x'],m[metric+'_y'],atol=1e-12,rtol=0)
w=a.pivot(index=['panel','exponent'],columns='design',values='common_reduction')
w['delta']=w.other-w.total
w.to_csv(o/'task4_direct_od_common_support/paired_common.csv')
print(w.groupby('exponent').delta.agg(['mean','min','max']))
print(a.overlap.mean())
