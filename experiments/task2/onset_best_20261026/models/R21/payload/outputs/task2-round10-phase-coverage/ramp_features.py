"""真实公开历史匝道与拓扑特征。匝道严格截止T-5；不读取未来质量/coverage。"""
import json,hashlib
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
DATA=Path(r'C:\Users\18049\Documents\Codex\2026-10-22\trafficflowbench-kaggle-rtx-5080-16gb-1\work\competition-data\kaggle_public')
WORK=Path(__file__).resolve().parents[2]/'work/round6-ramp-days';WORK.mkdir(parents=True,exist_ok=True)
@lru_cache(maxsize=8)
def topology(panel):
 directory=DATA/'corridors'/panel/'network';links=pd.read_csv(directory/'links.csv').sort_values('link_id');ids=links.link_id.astype(str).tolist();lookup={v:i for i,v in enumerate(ids)};top=pd.read_csv(directory/'lwr_mainline_topology.csv').fillna('');mapping=[];counts=np.zeros((len(ids),2));order=np.zeros(len(ids));external=np.zeros(len(ids));branches=np.zeros(len(ids))
 for row in top.to_dict('records'):
  link=str(row['mainline_link_id'])
  if link not in lookup:continue
  i=lookup[link];order[i]=float(row.get('order_index',0));external[i]=float(row.get('n_incoming_no_sensor',0));branches[i]=float(row.get('n_outgoing',0))
  for side,key in enumerate(['on_ramp_link_ids','off_ramp_link_ids']):
   for ramp in str(row.get(key,'')).split(';'):
    if ramp:mapping.append(dict(ramp_link_id=ramp,node=i,side=side));counts[i,side]+=1
 frame=pd.DataFrame(mapping);assert not frame.ramp_link_id.duplicated().any(),'匝道重复挂接，需要明确分流比例后才能使用';digest=hashlib.sha256((directory/'lwr_mainline_topology.csv').read_bytes()).hexdigest();return dict(ids=ids,map=frame,counts=counts,order=order/max(order.max(),1),external=external,branches=branches,hash=digest)
@lru_cache(maxsize=8)
def day(panel,split,date):
 net=topology(panel);directory=DATA/'corridors'/panel/split/'ramp_states';name='synthetic_ramp_'+date.replace('-','_')+'.parquet';matches=list(directory.rglob(name));assert len(matches)<=1;path=matches[0] if matches else None;source_hash=hashlib.sha256(path.read_bytes()).hexdigest() if path else 'missing';key=hashlib.sha256((panel+split+date+source_hash+net['hash']).encode()).hexdigest();cached=WORK/(key+'.npz')
 if cached.exists():
  with np.load(cached) as f:return f['a']
 a=np.zeros((288,len(net['ids']),4),dtype='float32')
 if path is not None:
  f=pd.read_parquet(path,columns=['timestamp','ramp_link_id','flow_vph','pct_observed','is_score_eligible']);assert not f.duplicated(['timestamp','ramp_link_id']).any();f['timestamp']=pd.to_datetime(f.timestamp,utc=True);assert (f.timestamp.dt.strftime('%Y-%m-%d')==date).all();f=f.merge(net['map'],on='ramp_link_id',how='inner',validate='many_to_one');good=f.is_score_eligible.astype(bool)&(f.pct_observed>=75)&np.isfinite(f.flow_vph);f=f[good];slots=(f.timestamp.dt.hour*12+f.timestamp.dt.minute//5).to_numpy(int);nodes=f.node.to_numpy(int);side=f.side.to_numpy(int);np.add.at(a,(slots,nodes,side),f.flow_vph.to_numpy('float32'));np.add.at(a,(slots,nodes,side+2),1)
 np.savez_compressed(cached,a=a);return a
def ramp_history(s,override=None):
 origin=pd.Timestamp(s['origin']);times=pd.date_range(end=origin-pd.Timedelta(minutes=5),periods=12,freq='5min');split=s.get('split','train');parts=[]
 for t in times:
  date=t.strftime('%Y-%m-%d');a=override[date] if override and date in override else day(s['panel'],split,date);parts.append(a[t.hour*12+t.minute//5])
 return np.stack(parts)
def extras(s,override=None):
 h=ramp_history(s,override);net=topology(s['panel']);n=s['x'].shape[1];assert net['ids']==list(s['network']['link_ids']);cap=s['network']['capacity'];ramp=[];carried=[]
 for side in [0,1]:
  count=net['counts'][:,side];coverage=np.divide(h[:,:,side+2],count[None],out=np.ones((12,n)),where=count[None]>0);raw=h[:,:,side]/cap[None];carry=np.zeros_like(raw);last=np.zeros(n);age=np.zeros(n)
  for t in range(12):
   valid=(coverage[t]>=.75)|(count==0);last=np.where(valid,raw[t],last);age=np.where(valid,0,age+1);carry[t]=last
  carried.append(carry)
  for w in [3,6,12]:
   z=carry[-w:];ramp.extend([z.mean(0),z.std(0),z.min(0),z.max(0)])
  ramp.extend([carry[-1],carry[-1]-carry[-4],carry[-1]-carry[-7],coverage[-1],coverage.mean(0),count/10,age/12])
 on,off=carried;netflow=on-off;ramp.extend([netflow[-1],netflow[-1]-netflow[-4],netflow.mean(0),netflow.std(0)])
 rampbase=np.column_stack(ramp);rampH=[]
 for horizon in range(1,7):rampH.append(np.concatenate([rampbase,np.column_stack([netflow[-1]+(netflow[-1]-netflow[-4])/3*horizon,on[-1]+(on[-1]-on[-4])/3*horizon])],axis=1))
 # 内部主线输入流量与可见匝道流量之差仅为部分平衡代理，外部CONN未知显式标记。
 x=s['x'][:12];own=x[:,:,1]*cap[None];observed=(x[:,:,5]<.5)&(x[:,:,7]>.5)&(x[:,:,3]>=.75);own=np.where(observed,own,0);incoming=np.zeros_like(own);incoming_known=np.zeros_like(own);edges=s['network']['edges'];a,b=edges
 if len(a):np.add.at(incoming,(np.arange(12)[:,None],b[None,:]),own[:,a]);np.add.at(incoming_known,(np.arange(12)[:,None],b[None,:]),observed[:,a])
 counts=np.bincount(b,minlength=n);known=np.divide(incoming_known,counts[None],out=np.ones_like(own),where=counts[None]>0);balance=(incoming-own)/cap[None]+netflow;physics=[]
 for w in [3,6,12]:
  z=balance[-w:];physics.extend([z.mean(0),z.std(0),z.min(0),z.max(0)])
 physics.extend([balance[-1],balance[-1]-balance[-4],known[-1],known.mean(0),net['external'],net['branches'],net['order'],counts/4,observed[-1].astype(float)])
 return np.concatenate(rampH).astype('float32'),np.tile(np.column_stack(physics),(6,1)).astype('float32')
