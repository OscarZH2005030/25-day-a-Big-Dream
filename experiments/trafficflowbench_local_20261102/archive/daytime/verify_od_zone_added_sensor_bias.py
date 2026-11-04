import pandas as pd
from pathlib import Path
p=Path('outputs/daytime')
o=p/'task4_od_zone_added_sensor_bias'
a=pd.read_csv(o/'noisy.csv')
b=pd.read_csv(p/'task4_od_zone_measurement_noise/noisy.csv')
k=['panel','selected_power','evaluated_power','noise_ratio']
c=a.merge(b,on=k,validate='one_to_one',suffixes=('_bias','_independent'))
assert len(c)==240
c['loss']=c.gain_independent-c.gain_bias
assert c.loss.min()>=-1e-12
c.to_csv(o/'paired.csv',index=False)
print(c.groupby('noise_ratio').loss.agg(['mean','min','max']))
