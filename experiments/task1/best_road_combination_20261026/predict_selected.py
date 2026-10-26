"""按已冻结的逐道路赢家推理。只读掩码输入，不训练、不读真值、不提交Kaggle。"""
import os
os.environ['OMP_NUM_THREADS']='4'
import argparse,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
HERE=Path(__file__).resolve().parent
import feature_runtime as base
from github_graph import StateTransformer

def load_routes():
 routes=json.loads((HERE/'selected-road-models.json').read_text(encoding='utf8'))['routes']
 for route in routes.values():
  for name,path in route['paths'].items():route['paths'][name]=str((HERE/path).resolve())
  if 'history' in route:route['history']=str((HERE/route['history']).resolve())
 return routes

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def graph(e,device):
 adj=np.eye(e.n,dtype=bool)
 for groups in e.neighbors.values():
  for i,ids in enumerate(groups):adj[i,ids]=True
 adj|=adj.T.copy();iso=adj.sum(1)==1;adj=adj|((adj.astype('int16')@adj.astype('int16'))>0)
 topology=pd.read_csv(e.panel/'network/lwr_mainline_topology.csv').set_index('mainline_link_id');order=np.array([topology.loc[l,'order_index'] for l in e.links]);direction=np.sign(order[None,:]-order[:,None]).astype('int64');width=int(adj.sum(1).max());idx=np.zeros((e.n,width),dtype='int64');valid=np.zeros_like(idx,dtype=bool);direct=np.zeros_like(idx)
 for i in range(e.n):
  ns=np.flatnonzero(adj[i]);idx[i,:len(ns)]=ns;valid[i,:len(ns)]=True;direct[i,:len(ns)]=direction[i,ns]+1
 return {k:torch.from_numpy(v).to(device) for k,v in [('neighbors',idx),('neighbor_valid',valid),('direction',direct)]},iso
def keys_for(data,panel,split,date):
 path=data/'task1'/panel/split/'sample_submission_state.csv';parts=[]
 for chunk in pd.read_csv(path,dtype=str,usecols=base.KEYS,chunksize=250000):parts.append(chunk[chunk.timestamp.str[:10]==date])
 keys=pd.concat(parts,ignore_index=True);assert len(keys)>0 and not keys.duplicated(base.KEYS).any();return keys
@torch.inference_mode()
def predict(panel,date,route,data,output,device='cpu'):
 split='train' if date<='2031-02-28' else 'validation' if date<='2031-03-31' else 'private'
 e=base.Experiment(SimpleNamespace(release_root=data,panel=panel,output=output/'metadata'/panel,seed=20261024));keys=keys_for(data,panel,split,date)
 if split!='train':e.masked_files={p.stem[-10:].replace('_','-'):p for p in (e.panel/split/'mainline_states_masked').rglob('*.parquet')}
 # 明确移除真值文件映射，保证推理不能意外读取标签。
 e.truth_files={}
 for n,p in route['paths'].items():assert sha(p)==route['sha256'][n],f'模型文件变化：{p}'
 if route.get('history_sha256'):assert sha(route['history'])==route['history_sha256'],'历史特征文件已变化'
 ii=keys.station_id.map(e.si).to_numpy(dtype=int);ts=pd.to_datetime(keys.timestamp,utc=True);jj=(ts.dt.hour*12+ts.dt.minute//5).to_numpy(dtype=int);assert np.array_equal(keys.link_id.to_numpy(),np.array(e.links)[ii])
 if route['model']=='jayden_186':
  from jayden_model import PriorityTransformer
  from observation_features import feature_grid_from_observations
  h=np.load(route['paths']['history']);assert list(h['stations'])==e.stations;e.history=h['mean']
  obs=pd.read_parquet(e.masked_files[date]);values,bases,_,_=feature_grid_from_observations(e,date,obs);base.append_features(e,date,values,obs)
  raw=np.stack(list(values.values()),axis=-1).transpose(1,0,2);full,names,_=base.generate(raw,list(values));z=full[...,[names.index(n) for n in base.BASE['all_numeric_features']]].astype('float32');flat=z.reshape(-1,164);X,names=base.expand_fast(np.concatenate([flat,np.isfinite(flat)],axis=1));ni=[i for i,n in enumerate(names) if not n.endswith('__finite')];numeric=X[:,ni].reshape(288,e.n,186)
  packed=(np.sign(numeric)*np.log1p(np.abs(numeric))/5).astype('float16').astype('float32');norm=np.load(route['paths']['normalization']);finite=np.isfinite(packed);features=np.concatenate([np.where(finite,((packed-norm['mean'])/norm['scale']).clip(-8,8),0),finite],axis=-1).astype('float32');initial=np.stack([bases[0]/25,bases[1]/600],axis=-1).transpose(1,0,2).astype('float32')
  model=PriorityTransformer(route['config'],'T1_balanced').to(device);model.load_state_dict(torch.load(route['paths']['transformer'],map_location=device,weights_only=False)['model']);model.eval();g,_=graph(e,device);starts=list(range(0,265,12));total=np.zeros((288,e.n,2));count=np.zeros((288,1,1))
  for offset in range(0,len(starts),4):
   ss=starts[offset:offset+4];batch=dict(features=torch.from_numpy(np.stack([features[s:s+24] for s in ss])).to(device),base=torch.from_numpy(np.stack([initial[s:s+24] for s in ss])).to(device),time_valid=torch.ones((len(ss),24),dtype=torch.bool,device=device));predictions=model(batch,g).cpu().numpy()
   for s,p in zip(ss,predictions):total[s:s+24]+=p;count[s:s+24]+=1
  pred=(total/count)[jj,ii]*np.array([25,600])
 elif route['model']=='github_original_hybrid':
  ck=torch.load(route['paths']['transformer'],map_location='cpu',weights_only=False);assert e.stations==ck['stations'] and e.links==ck['links'];h=np.load(route['paths']['history']);assert list(h['stations'])==e.stations;e.history=h['mean'];cfg=ck['args'];model=StateTransformer(102,cfg['width'],cfg['heads'],cfg['layers'],cfg['window'],cfg['dropout']).to(device);model.load_state_dict(ck['state_dict']);model.eval()
  values,bases,_,obs=e.feature_grid(date);raw=np.stack(list(values.values()),axis=-1).transpose(1,0,2).astype('float32');names=list(values);assert ck['input_names']==names+[n+'__finite' for n in names]
  finite=np.isfinite(raw);features=np.concatenate([np.where(finite,(raw-ck['feature_mean'])/ck['feature_scale'],0),finite],axis=-1).astype('float32');initial=np.stack([bases[0]/25,bases[1]/600],axis=-1).transpose(1,0,2).astype('float32');g,iso=graph(e,device)
  total=np.zeros(initial.shape);count=np.zeros((288,1,1));window=cfg['window'];starts=list(range(0,289-window,max(1,window//2)))
  if starts[-1]!=288-window:starts.append(288-window)
  for s in starts:
   b={'features':torch.from_numpy(features[None,s:s+window]).to(device),'base':torch.from_numpy(initial[None,s:s+window]).to(device)};v=model(b,g)[0].cpu().numpy();total[s:s+window]+=v;count[s:s+window]+=1
  normalized=total/count;pred=normalized[jj,ii]*np.array([25,600]);chosen=iso[ii]
  if chosen.any():
   x=raw[jj[chosen],ii[chosen]];base_physical=initial[jj[chosen],ii[chosen]].astype('float64')*np.array([25,600]);res=np.column_stack([lgb.Booster(model_file=route['paths'][c]).predict(x,num_threads=4) for c in ['speed','flow']]);pred[chosen]=np.maximum(base_physical+res,0)
 else:
  assert route['model']=='adapted_lightgbm';e.history=np.load(route['history']);values,bases,_,obs=e.feature_grid(date);base.append_features(e,date,values,obs);raw=np.stack(list(values.values()),axis=-1).transpose(1,0,2);full,names,_=base.generate(raw,list(values));numeric=full[...,[names.index(n) for n in base.BASE['all_numeric_features']]].astype('float32');mat=numeric.reshape(-1,164);X,names=base.expand_fast(np.concatenate([mat,np.isfinite(mat)],axis=1));assert names==base.SCHEMA['input_names'];X=X.reshape(288,e.n,372)[jj,ii]
  positions=e.grid(obs,'milepost')[:,0].astype('float32');positions=(positions-positions.min())/max(float(np.ptp(positions)),1);ri=base.PANELS.index(panel);X=np.column_stack([X,np.tile(np.eye(10)[ri],(len(ii),1)),positions[ii]]).astype('float32');initial=np.stack([bases[0]/25,bases[1]/600],axis=-1).transpose(1,0,2).astype('float32')[jj,ii];res=np.column_stack([lgb.Booster(model_file=route['paths'][c]).predict(X,num_threads=4) for c in ['speed','flow']]);pred=np.maximum(initial+res,0)*np.array([25,600])
 for c in ['speed_kmh','flow_vph','occupancy']:assert not np.isfinite(e.grid(obs,c)[ii,jj]).any()
 assert np.isfinite(pred).all() and (pred>=0).all();result=keys.copy();result['speed_kmh']=pred[:,0];result['flow_vph']=pred[:,1]*e.lanes[ii];result['selected_model']=route['model'];path=output/(panel+'_'+date+'.parquet');path.parent.mkdir(parents=True,exist_ok=True);result.to_parquet(path,index=False);return result
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--date',required=True);parser.add_argument('--panel',action='append');parser.add_argument('--data-root',type=Path,required=True);parser.add_argument('--output',type=Path,default=HERE/'predictions');parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=parser.parse_args();torch.set_num_threads(4);routes=load_routes()
 for panel in args.panel or list(routes):
  result=predict(panel,args.date,routes[panel],args.data_root,args.output,args.device);print(panel,args.date,routes[panel]['model'],len(result),flush=True)
