from pathlib import Path
import pandas as pd
import numpy as np
o=Path('outputs/daytime')
a=o/'task4_direct_od_noise_adaptation'
d=pd.read_csv(a/'by_panel.csv')
assert len(d)==200
keys=['panel','design','exponent']
for noise,reference,filter_expr in [(0,'fixed_noise','noise==.25'),(.25,'noisy_support','index==index')]:
    b=pd.read_csv(o/f'task4_direct_od_{reference}/by_panel.csv').query(filter_expr)
    m=d.query('assumed_noise==@noise').merge(b,on=keys,validate='one_to_one')
    assert len(m)==100
    np.testing.assert_allclose(m.total_reduction_x,m.total_reduction_y,atol=1e-12,rtol=0)
    assert (m.common_reduction_x<=m.common_reduction_y+1e-12).all()
w=d.pivot(index=keys,columns='assumed_noise',values=['total_reduction','common_reduction'])
delta=w.xs(.25,axis=1,level=1)-w.xs(0.,axis=1,level=1)
delta.to_csv(a/'paired_adaptation.csv')
print(delta.groupby(['design','exponent']).mean())
