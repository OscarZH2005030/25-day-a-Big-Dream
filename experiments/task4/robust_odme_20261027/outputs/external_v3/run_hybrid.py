# 中文说明：探索按方向启用外部先验，并以新的随机划分复核；非独立路径盲测。
import sys,json
from pathlib import Path
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O));from run_external import *
cv=pd.read_csv(O/'train_selection.csv').groupby(['panel','candidate']).NMAE.mean().unstack();ext='external_attr10_recent';policy={p:(ext if cv.loc[p,ext]<cv.loc[p,'official_005'] else 'official_005') for p in cv.index}
rows=[]
for panel,name in policy.items():
 ids,links,D,paths=load_operator(REL/'corridors'/panel/'network');A,c,b,t,_=load(panel,'train',ids,links,D)
 for seed in [20261031,20261101]:
  fd=folds(A,seed)
  for cand in ['official_005','direction_hybrid']:
   model=name if cand=='direction_hybrid' else cand
   for k in range(3):
    x=pred(panel,A[fd!=k],c[fd!=k],b,paths,model);rows.append({'panel':panel,'seed':seed,'candidate':cand,'fold':k,'NMAE':float(abs(A[fd==k]@x-c[fd==k]).sum()/c[fd==k].sum())})
pd.DataFrame(rows).to_csv(O/'hybrid_fresh_resampling.csv',index=False);agg=pd.DataFrame(rows).groupby(['seed','candidate','panel']).NMAE.mean().groupby(['seed','candidate']).mean();agg.to_csv(O/'hybrid_resampling_overall.csv')
base=pd.read_csv(O/'task4_official_005.csv',dtype=str);alt=pd.read_csv(O/f'task4_{ext}.csv',dtype=str);keys=['panel','departure_time','path_id','origin_zone','destination_zone'];assert base[keys].equals(alt[keys]);mask=base.panel.isin([p for p,c in policy.items() if c==ext]);base.loc[mask,'path_flow']=alt.loc[mask,'path_flow'];base.to_csv(O/'task4_direction_hybrid.csv',index=False)
(O/'hybrid_policy.json').write_text(json.dumps({'policy':policy,'status':'exploratory; original baseline unchanged','selection':'panelwise improvement on seed20261029; developed after viewing earlier audits','new_resampling_seeds':[20261031,20261101],'independent_path_truth_test':False},indent=2),encoding='utf-8')
print(policy);print(agg)
