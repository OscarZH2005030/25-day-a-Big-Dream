import os
# 中文说明：严格验收自然键、区域、period token 及统一提交键；不写入其他任务。
from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parent
REL=Path(os.environ['TFB_RELEASE_ROOT']).expanduser().resolve()
x=pd.read_csv(ROOT/'task4_path_flow.csv',dtype=str)
keys=['panel','departure_time','path_id','origin_zone','destination_zone']
t=[]
for s in ['validation','private']:
 for p in sorted((REL/'task4').iterdir()):
  t.append(pd.read_csv(p/s/'sample_submission_path_flow.csv',dtype=str))
t=pd.concat(t,ignore_index=True)
assert x[keys].equals(t[keys])
f=x.path_flow.astype(float); assert np.isfinite(f).all() and (f>=0).all()
assert not x.duplicated(keys[:3]).any()
parts=[]
for chunk in pd.read_csv(REL/'submission_key.csv',dtype=str,chunksize=250000):
 parts.append(chunk[chunk.task=='odme'])
k=pd.concat(parts,ignore_index=True)
join=k.merge(x,on=keys[:3],how='outer',indicator=True,suffixes=('_key','_pred'),validate='one_to_one')
assert (join._merge=='both').all()
for c in ['origin_zone','destination_zone']:
 assert (join[c+'_key']==join[c+'_pred']).all()
join[['submission_id','path_flow']].to_csv(ROOT/'task4_submission_id_patch.csv',index=False)
checks={'rows':len(x),'panels':x.panel.nunique(),'periods':sorted(x.departure_time.unique()),'unified_key_rows':len(k),'missing':0,'duplicates':0,'nonfinite':0,'negative':0,'zones_match':True,'template_keys_and_order_match':True}
(ROOT/'acceptance.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
manifest={}
for p in sorted((REL/'task4').rglob('*.csv')): manifest[str(p.relative_to(REL))]=hashlib.sha256(p.read_bytes()).hexdigest()
for p in sorted((REL/'config').glob('*.json')): manifest[str(p.relative_to(REL))]=hashlib.sha256(p.read_bytes()).hexdigest()
(ROOT/'input_hashes.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
