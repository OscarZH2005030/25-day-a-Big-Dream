# 中文说明：固定已选1.25幅度，额外检验更强先验误差；此结果不再反向调参。
import sys
from pathlib import Path
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O));from run_dev import *
real=pd.read_csv(O.parent/'fivemin_v4/real_daily_PM_flows.csv');rows=[]
for panel,d in real.groupby('panel'):
 ids,links,D,paths=load_operator(REL/'corridors'/panel/'network');A,c,b,t,_=load(panel,'train',ids,links,D);table=d.pivot(index='day_index',columns='station_id',values='PM_mean_flow_per_lane');ratio=np.clip(table.to_numpy()/table.iloc[:15].mean().to_numpy(),.25,4);oi=pd.factorize(paths.origin_zone)[0];di=pd.factorize(paths.destination_zone)[0]
 for gain in [2.,3.]:
  for day in [17,23,29]:
   z=np.clip(ratio[day]**gain,.1,8);rng=np.random.default_rng(29100+day)
   for scenario in ['destination_shock','origin_destination_shock','additive_clipped']:
    if scenario=='destination_shock':f=b*rng.choice(z,di.max()+1)[di]
    elif scenario=='origin_destination_shock':f=b*np.sqrt(rng.choice(z,oi.max()+1)[oi]*rng.choice(z,di.max()+1)[di])
    else:f=np.maximum(0,b+np.mean(b)*(rng.choice(z,len(b))-1))
    cc=A@f;x,st=kl(A,cc,b,.05);assert st['success'];y,st=ridge(A,cc,np.maximum(b+1.25*(x-b),0),'ridge',.05);assert st['success']
    for name,p in [('kl',x),('kl_amp125',y)]:rows.append({'panel':panel,'gain':gain,'day':day,'scenario':scenario,'candidate':name,**metric(p,f,b,A,cc,paths.destination_zone.to_numpy())})
pd.DataFrame(rows).to_csv(O/'stress_trials.csv',index=False);g=pd.DataFrame(rows).groupby(['gain','candidate'])[['S_od','S_dev','S_attr','synthetic_S_ODME']].mean();g.to_csv(O/'stress_overall.csv');print(g.to_string())
