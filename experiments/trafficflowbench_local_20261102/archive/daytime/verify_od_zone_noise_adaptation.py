import pandas as pd
from pathlib import Path
o=Path('outputs/daytime/task4_od_zone_noise_adaptation')
a=pd.read_csv(o/'adapted.csv')
b=pd.read_csv(o.parent/'task4_od_zone_measurement_noise/noisy.csv')
pd.testing.assert_frame_equal(pd.read_csv(o/'noisy.csv'),b,atol=1e-10,rtol=0)
assert len(a)==60 and not a.duplicated(['panel','power','noise_ratio']).any()
a['delta']=a.adaptive_gain-a.fixed_gain
s=a.groupby(['power','noise_ratio'])[['delta','overlap']].agg(['mean','min','max'])
s.to_csv(o/'adaptation_summary.csv')
print(s)
