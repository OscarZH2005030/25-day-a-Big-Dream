"""分布驱动局部特征：只依赖过去12帧与合法masked T。"""
import numpy as np
from causal_features import enhanced
def latest_state(s):
 x=s['x'];n=x.shape[1];last=np.ones(n,dtype='float32')
 for t in range(13):last=np.where(x[t,:,4]<.5,x[t,:,0],last)
 return last,last<=s['network']['threshold']/s['network']['free_speed']
def features(s):
 x=s['x'];net=s['network'];n=x.shape[1];extra=[];carried=[]
 for ch,missing,default in [(0,4,1.),(1,5,0.),(2,6,0.)]:
  a=x[:,:,ch];valid=x[:,:,missing]<.5;z=np.empty_like(a);last=np.full(n,default)
  for t in range(13):last=np.where(valid[t],a[t],last);z[t]=last
  carried.append(z)
  for w in [3,6,12]:
   tail=z[-w:];extra.extend([np.quantile(tail,.1,axis=0),np.quantile(tail,.9,axis=0),tail[-1]-tail.mean(0),np.mean(np.diff(tail,axis=0)<0,axis=0),np.mean(np.diff(tail,axis=0)>0,axis=0)])
  extra.extend([z[-1]-z[-2],z[-2]-z[-3],z[-1]-2*z[-2]+z[-3]])
 speed,flow,occ=carried;cut=net['threshold']/net['free_speed'];v=speed[-1];q=flow[-1];o=occ[-1];dv=(v-speed[-4])/3;dq=(q-flow[-4])/3;do=(o-occ[-4])/3;density=q/np.maximum(v,.15)
 for level in [.95,.85,.75,.65]:extra.append((speed[-6:]<level).mean(0))
 extra.extend([density,dv,dq,do,q*o,q*np.maximum(1-v,0),np.maximum(cut-v,0),np.maximum(v-cut,0),density-(flow[-4]/np.maximum(speed[-4],.15))])
 # 多跳局部瓶颈和速度/需求梯度，保留方向与距离。
 edges=net['edges'];source=np.column_stack([v,q,o,dv,dq,do,density,(v<=cut).astype(float)])
 downstream=None
 for reverse in [False,True]:
  a,b=edges if reverse else edges[::-1];z=source.copy()
  for hop in range(3):
   total=np.zeros_like(z);count=np.zeros(n);minimum=np.full_like(z,np.inf);maximum=np.full_like(z,-np.inf);np.add.at(total,b,z[a]);np.add.at(count,b,1);np.minimum.at(minimum,b,z[a]);np.maximum.at(maximum,b,z[a]);mean=np.divide(total,count[:,None],out=np.zeros_like(total),where=count[:,None]>0);minimum=np.where(np.isfinite(minimum),minimum,0);maximum=np.where(np.isfinite(maximum),maximum,0)
   extra.extend([mean[:,0]-v,mean[:,1]-q,mean[:,2]-o,minimum[:,0],maximum[:,3],minimum[:,3],maximum[:,4],maximum[:,6],maximum[:,7]])
   if not reverse and hop==0:downstream=mean[:,0]
   z=mean
 base=np.column_stack(extra).astype('float32');out=[]
 for h in range(1,7):
  # 这些为历史斜率外推量，不是真实未来观测。
  projected=v+np.clip(dv,-.12,.12)*h;dyn=np.column_stack([projected,projected-cut,np.maximum(cut-projected,0),q+dq*h,o+do*h,v-downstream,(v-downstream)*h,net['length']*12/max(h,1)])
  out.append(np.concatenate([base,dyn],axis=1))
 result=np.concatenate([enhanced(s,net),np.concatenate(out)],axis=1).astype('float32');assert np.isfinite(result).all();return result
def event_features(s):
 n=s['x'].shape[1];x=features(s)[:n];return np.concatenate([x[:,:185],x[:,191:]],axis=1)
