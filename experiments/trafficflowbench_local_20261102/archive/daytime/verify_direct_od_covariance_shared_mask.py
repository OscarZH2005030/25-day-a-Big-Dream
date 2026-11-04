from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
root=Path('outputs/daytime')
out=root/'task4_direct_od_covariance_shared_mask'
data=pd.read_csv(out/'by_panel.csv')
keys=['panel','design','exponent','law']
assert len(data)==600 and not data.duplicated(keys+['selection']).any()
hashes={}
for selection,suffix in [('joint','covariance_transfer'),('diagonal','reverse_covariance_transfer'),('balanced','covariance_balanced')]:
    path=root/('task4_direct_od_'+suffix)/'by_panel.csv'
    hashes[selection]=hashlib.sha256(path.read_bytes()).hexdigest()
    paired=data.query('selection==@selection').merge(pd.read_csv(path),on=keys,validate='one_to_one')
    assert len(paired)==200
    np.testing.assert_allclose(paired.total_reduction_x,paired.total_reduction_y,atol=1e-12,rtol=0)
    assert (paired.common_reduction_x<=paired.common_reduction_y+1e-12).all()
p=data.pivot(index=['panel','selection','exponent','law'],columns='design',values='common_reduction')
p['delta']=p.other-p.total
p.to_csv(out/'paired_common.csv')
(out/'verification.json').write_text(json.dumps(dict(input_sha256=hashes,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),failures=[]),indent=2))
print(p.groupby(['selection','law']).delta.mean())
