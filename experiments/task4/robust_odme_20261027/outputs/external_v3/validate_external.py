# 中文说明：严格对照官方模板，验收所有外部数据候选，不更改已冻结主预测。
from pathlib import Path
import sys,json
import pandas as pd,numpy as np
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O.parent));from run_task4 import REL
keys=['panel','departure_time','path_id','origin_zone','destination_zone']
t=pd.concat([pd.read_csv(p/s/'sample_submission_path_flow.csv',dtype=str) for s in ['validation','private'] for p in sorted((REL/'task4').iterdir())],ignore_index=True)
checks=[]
for p in sorted(O.glob('task4_*.csv')):
 x=pd.read_csv(p,dtype=str);v=x.path_flow.to_numpy(float)
 assert x[keys].equals(t[keys]) and np.isfinite(v).all() and (v>=0).all() and not x.duplicated(keys[:3]).any()
 checks.append({'file':p.name,'rows':len(x),'all_keys_zones_tokens_match':True,'finite_nonnegative':True})
(O/'acceptance.json').write_text(json.dumps(checks,indent=2),encoding='utf-8');print(json.dumps(checks,indent=2))
