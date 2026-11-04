from pathlib import Path
import pandas as pd
import numpy as np
root=Path('outputs/daytime')
out=root/'task4_od_zone_balanced_selection'
a=pd.read_csv(out/'transfer.csv')
b=pd.read_csv(root/'task4_od_zone_variance_transfer/transfer.csv')
assert len(a)==120 and not a.duplicated(['panel','selected_power','evaluated_power']).any()
reference=a[a.selected_power!='balanced'].copy()
reference.selected_power=reference.selected_power.astype(float)
pd.testing.assert_frame_equal(reference.reset_index(drop=True),b,atol=1e-10,rtol=0)
w=a.pivot(index=['panel','evaluated_power'],columns='selected_power',values='gain')
r=(w.max(axis=1).to_numpy()[:,None]-w).groupby('panel').max()
r.to_csv(out/'worst_gap_by_panel.csv')
print(r.agg(['mean','max']))
