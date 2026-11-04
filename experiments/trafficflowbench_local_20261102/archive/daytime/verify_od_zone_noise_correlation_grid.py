from pathlib import Path
import hashlib,json
import pandas as pd
p=Path('outputs/daytime')
o=p/'task4_od_zone_noise_correlation_grid'
a=pd.read_csv(o/'noisy.csv')
keys=['panel','selected_power','evaluated_power','noise_ratio']
assert len(a)==1920 and not a.duplicated(keys+['rho']).any()
hashes={}
for rho,name in [(-.05,'negative_measurement_noise'),(0.,'measurement_noise'),(.5,'shared_measurement_noise')]:
    source=p/('task4_od_zone_'+name)/'noisy.csv'
    hashes[name]=hashlib.sha256(source.read_bytes()).hexdigest()
    pd.testing.assert_frame_equal(a[a.rho==rho].drop(columns='rho').reset_index(drop=True),pd.read_csv(source),atol=1e-10,rtol=0)
s=a.groupby(['noise_ratio','rho']).gain.mean().unstack()
s.to_csv(o/'correlation_summary.csv')
(o/'verification.json').write_text(json.dumps(dict(reference_sha256=hashes,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),failures=[]),indent=2))
print(s)
print(s.idxmin(axis=1))
