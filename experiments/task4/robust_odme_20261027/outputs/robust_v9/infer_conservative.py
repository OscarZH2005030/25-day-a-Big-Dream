# 中文说明：只部署已冻结的保守方向策略；不重做人工选参，不覆盖官方基准。
import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
from pathlib import Path
import sys,json
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O.parent));sys.path.insert(0,str(O.parent/'improvement_v2'))
from run_task4 import REL,load,load_operator,solve as ridge
from run_v2 import kl
import pandas as pd,numpy as np
policy=json.loads((O/'conservative_policy.json').read_text())['policy'];base=pd.read_csv(O.parent/'task4_path_flow.csv',dtype=str);out=base.copy()
for panel,name in policy.items():
 if name=='official':continue
 weight={'mix25':.25,'mix50':.5,'mix75':.75,'scaled_kl':1}[name];ids,ls,D,paths=load_operator(REL/'corridors'/panel/'network')
 for split in ['validation','private']:
  A,c,b,t,_=load(panel,split,ids,ls,D);x,st=ridge(A,c,b,'ridge',.05);assert st['success'];q,st=kl(A,c,b*c.sum()/(A@b).sum(),.05);assert st['success'];pred=(1-weight)*x+weight*q;mask=(out.panel==panel)&(out.departure_time==t.departure_time.iloc[0]);assert out.loc[mask,'path_id'].tolist()==ids;out.loc[mask,'path_flow']=pred.astype(str)
keys=['panel','departure_time','path_id','origin_zone','destination_zone'];v=out.path_flow.to_numpy(float);assert out[keys].equals(base[keys]) and np.isfinite(v).all() and (v>=0).all() and not out.duplicated(keys[:3]).any();out.to_csv(O/'task4_conservative_candidate.csv',index=False)
print('Conservative inference validated:',len(out),'rows')
