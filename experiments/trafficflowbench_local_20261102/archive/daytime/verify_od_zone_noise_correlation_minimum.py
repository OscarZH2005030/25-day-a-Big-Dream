import pandas as pd
from pathlib import Path
o=Path('outputs/daytime/task4_od_zone_noise_correlation_minimum')
a=pd.read_csv(o/'noisy.csv')
b=pd.read_csv(o.parent/'task4_od_zone_noise_correlation_grid/noisy.csv')
k=['panel','selected_power','evaluated_power','noise_ratio']
c=a.merge(b.groupby(k).gain.min().rename('grid_min'),on=k,validate='one_to_one')
assert len(c)==240 and (c.gain<=c.grid_min+1e-12).all()
assert c.slope.abs().max()<1e-10
c['grid_minus_continuous']=c.grid_min-c.gain
s=c.groupby('noise_ratio')[['rho','grid_minus_continuous']].agg(['min','mean','max'])
s.to_csv(o/'minimum_summary.csv')
print(s.to_string())
