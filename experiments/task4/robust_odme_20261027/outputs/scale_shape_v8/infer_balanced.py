# 中文说明：独立推理，不重做人工场景训练；四方向尺度校正KL，其余方向保持冻结基准。
import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import sys,json
from pathlib import Path
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O.parent));sys.path.insert(0,str(O.parent/'improvement_v2'))
from run_task4 import REL,load,load_operator
from run_v2 import kl
import pandas as pd,numpy as np
base=pd.read_csv(O.parent/'task4_path_flow.csv',dtype=str);out=base.copy();panels=sorted(pd.read_csv(O/'balanced_deployment.csv').panel.unique())
for panel in panels:
 ids,links,D,paths=load_operator(REL/'corridors'/panel/'network')
 for split in ['validation','private']:
  A,c,b,t,_=load(panel,split,ids,links,D);s=float(c.sum()/(A@b).sum());x,st=kl(A,c,b*s,.05);assert st['success'],st;mask=(out.panel==panel)&(out.departure_time==t.departure_time.iloc[0]);assert out.loc[mask,'path_id'].tolist()==ids;out.loc[mask,'path_flow']=x.astype(str)
keys=['panel','departure_time','path_id','origin_zone','destination_zone'];v=out.path_flow.to_numpy(float);assert out[keys].equals(base[keys]) and np.isfinite(v).all() and (v>=0).all() and not out.duplicated(keys[:3]).any();out.to_csv(O/'task4_balanced_scale_candidate.csv',index=False)
(O/'balanced_acceptance.json').write_text(json.dumps({'rows':len(out),'keys_zones_periods_match':True,'finite_nonnegative':True,'duplicates':0,'baseline_preserved':True,'updated_panels':panels},indent=2),encoding='utf-8');print('Inference and validation completed:',len(out),'rows')
