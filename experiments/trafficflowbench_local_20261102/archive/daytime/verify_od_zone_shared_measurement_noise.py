from pathlib import Path
import pandas as pd
p=Path('outputs/daytime')
a=pd.read_csv(p/'task4_od_zone_shared_measurement_noise/noisy.csv')
b=pd.read_csv(p/'task4_od_zone_measurement_noise/noisy.csv')
keys=['panel','selected_power','evaluated_power','noise_ratio']
c=a.merge(b,on=keys,validate='one_to_one',suffixes=('_shared','_independent'))
assert len(c)==240
c['delta']=c.gain_shared-c.gain_independent
c.to_csv(p/'task4_od_zone_shared_measurement_noise/paired.csv',index=False)
print(c.groupby('noise_ratio').delta.agg(['mean','min','max']))
