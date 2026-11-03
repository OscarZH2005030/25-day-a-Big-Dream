import itertools
import numpy as np
import pandas as pd
from pathlib import Path
root=Path('outputs/daytime')
p=pd.read_csv(root/'task4_source_error_amplitude_panel_check/by_panel.csv')
q=pd.read_csv(root/'task4_source_error_rule_interaction/summary.csv').set_index('noise')
x=p.query("rule=='agreement_source'").pivot(index='panel',columns='noise',values='delta')-p.query("rule=='mean_source'").pivot(index='panel',columns='noise',values='delta')
def stat(s):
    sums=np.asarray(s)@x.to_numpy()
    return abs(sums)*np.sqrt(9/(10*(x*x).sum().to_numpy()-sums*sums))
a=stat(np.ones(10))
b=np.array([stat(s) for s in itertools.product([-1,1],repeat=10)])
np.testing.assert_allclose(q.statistic,a,atol=1e-12)
np.testing.assert_array_equal(q.raw_reference_fraction,(b>=a-1e-12).mean(axis=0))
np.testing.assert_array_equal(q.joint_reference_fraction,(b.max(axis=1)[:,None]>=a-1e-12).mean(axis=0))
print('Verified all 1024 assignments using sums and sums of squares')
