"""新实验：Jayden T1均衡边界采样 + 186特征 + 训练遮挡视图库。

不是原队列复现：按原配方补跑剩余8路；已完成2路保持只读，不运行T0/融合或别人的断点。
每日固定一份合法增强视图，训练窗口随机抽取原/增强视图；增强后全部186特征重新计算。
"""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import sys,json,time,datetime,hashlib,argparse,random,gc,math,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch import nn
import feature_runtime as feature
from observation_features import feature_grid_from_observations
from jayden_model import PriorityTransformer,objective
from jayden_sampling import graph_tensors,sampling_spec,additional_mask,choose_supervision

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[1]/'work';CACHE=WORK/'jayden-t1-186-cache'
DATA=Path(r'C:\Users\18049\Documents\Codex\2026-10-22\trafficflowbench-kaggle-rtx-5080-16gb-1\work\competition-data\kaggle_public')
ROADS=['D12_I405_N', 'D12_I405_S', 'D12_I5_S', 'D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_S'];PLAN=json.loads((ROOT/'reference-split.json').read_text());TRAIN=PLAN['training_dates'];INNER=PLAN['internal_validation_dates'];HOLD=PLAN['holdout_dates']
CONFIG=dict(window=24,patch_nodes=64,width=64,heads=4,layers=3,dropout=.1,batch_size=2,seed=2026,threads=4,epochs=60,steps_per_epoch=1024,patience=8,learning_rate=.0003,weight_decay=.01,clip=1.,eval_batch=4,
            input_numeric=186,input_columns=372,augmented_view_probability=.5,checkpoint_every_epochs=5,training_dtype='float32',feature_transform='signedlog/5 float16 cache then training-only zscore clipped8',
            mask_library='one deterministic daily mask, randomly sampled alongside static view; not original per-step dynamic recomputation')
AUGMENT=dict(hidden_fraction=.10,extra_supervision_fraction=.20,patterns=['point','temporal','spatial'],temporal_lengths=[6,12],spatial_links=4,spatial_ticks=6)
NAMES=[n for n in feature.SCHEMA['input_names'] if not n.endswith('__finite')];assert len(NAMES)==186
COLUMNS=['speed_kmh','flow_vph','occupancy','pct_observed','is_imputed']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def replace(tmp,p):
 for i in range(40):
  try:os.replace(tmp,p);return
  except PermissionError:
   if i==39:raise
   time.sleep(.05)
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf8');replace(t,p)
def array(p,v):
 t=p.with_suffix('.tmp')
 with t.open('wb') as f:np.save(f,v,allow_pickle=False)
 replace(t,p)
def checkpoint(p,v):
 t=p.with_suffix('.tmp');torch.save(v,t);replace(t,p)
def progress(phase,**kw):
 v=dict(phase=phase,pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw)
 try:save(ROOT/'progress.json',v)
 except PermissionError:pass
 print(json.dumps(v,ensure_ascii=False),flush=True)
def preflight(lock=False):
 files=[p for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts]
 signature=dict(config=CONFIG,augmentation=AUGMENT,source={str(p.relative_to(ROOT)):sha(p) for p in files},history={r:sha(ROOT/'history'/r/'history.npz') for r in ROADS},split=PLAN,environment=dict(torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__))
 if (ROOT/'run-signature.json').exists():assert json.loads((ROOT/'run-signature.json').read_text())==signature,'源码/输入签名改变，不能混跑旧断点'
 elif lock:save(ROOT/'run-signature.json',signature)
 save(ROOT/'training-config.json',signature)

class Data:
 def __init__(self,road):
  self.road=road;self.folder=CACHE/road;self.folder.mkdir(parents=True,exist_ok=True);self.exp=feature.Experiment(SimpleNamespace(release_root=DATA,panel=road,output=self.folder/'metadata',seed=2026));e=self.exp;self.n=e.n
  h=np.load(ROOT/'history'/road/'history.npz');assert list(h['stations'])==e.stations;e.history=h['mean']
  # 历史仅为原协议首28日，排除训练期增强值通过画像返回的可能。
  topo=pd.read_csv(e.panel/'network/lwr_mainline_topology.csv').set_index('mainline_link_id');order=np.array([topo.loc[l,'order_index'] for l in e.links]);adj=np.eye(e.n,dtype=bool)
  for groups in e.neighbors.values():
   for i,ids in enumerate(groups):adj[i,ids]=True
  adj=adj|adj.T;self.graph=dict(graph=adj|((adj.astype('int16')@adj.astype('int16'))>0),direction=np.sign(order[None,:]-order[:,None]).astype('int64'),order=np.argsort(order),isolated=adj.sum(1)==1)
  self.adjacency=[set(np.flatnonzero(adj[i]&~np.eye(e.n,dtype=bool)[i]).tolist()) for i in range(e.n)];self.units=np.arange(e.n)
  keys=pd.read_parquet(WORK/'cache'/road/'keys.parquet');self.keys={d:k.reset_index(drop=True) for d,k in keys.groupby(keys.timestamp.str[:10])};self.arrays={};self.graph_cache={};self.allow_holdout=False

 def features(self,date,obs):
  e=self.exp;values,bases,_,_=feature_grid_from_observations(e,date,obs);assert list(values)==feature.BASE['original_features'];feature.append_features(e,date,values,obs);raw=np.stack(list(values.values()),axis=-1).transpose(1,0,2);full,names,_=feature.generate(raw,list(values));z=full[...,[names.index(n) for n in feature.BASE['all_numeric_features']]].astype('float32');flat=z.reshape(-1,164);X,names=feature.expand_fast(np.concatenate([flat,np.isfinite(flat)],axis=1));assert names==feature.SCHEMA['input_names'];ni=[i for i,n in enumerate(names) if not n.endswith('__finite')];numeric=X[:,ni].reshape(288,self.n,186);packed=(np.sign(numeric)*np.log1p(np.abs(numeric))/5).astype('float16');assert not np.isinf(packed).any();base=np.stack([bases[0]/25,bases[1]/600],axis=-1).transpose(1,0,2).astype('float32');assert np.isfinite(base).all();return packed,base

 def prepare(self,date):
  if date in HOLD and not self.allow_holdout:raise ValueError('模型冻结前禁止准备留出标签')
  folder=self.folder/date;folder.mkdir(exist_ok=True)
  if (folder/'ready.json').exists():return
  e=self.exp;obs=pd.read_parquet(e.masked_files[date]);keys=self.keys[date];ii=keys.station_id.map(e.si).to_numpy(dtype=int);ts=pd.to_datetime(keys.timestamp,utc=True);jj=(ts.dt.hour*12+ts.dt.minute//5).to_numpy(dtype=int);assert np.array_equal(keys.link_id.to_numpy(),np.array(e.links)[ii]);assert len(np.unique(jj*e.n+ii))==len(ii)
  for col in ['speed_kmh','flow_vph','occupancy']:assert not np.isfinite(e.grid(obs,col)[ii,jj]).any()
  mask=np.zeros((288,e.n),bool);mask[jj,ii]=True;packed,base=self.features(date,obs);array(folder/'static_x.npy',packed);array(folder/'static_base.npy',base);info={}
  if date in TRAIN:
   observation=np.stack([e.grid(obs,c) for c in COLUMNS],axis=-1);rng=np.random.default_rng(CONFIG['seed']+(ROADS.index(self.road)+2)*10000+TRAIN.index(date));hidden,eligible,info=additional_mask(observation,self.units,self.adjacency,rng,AUGMENT)
   assert not (hidden.T&mask).any();row_i=obs.station_id.map(e.si).to_numpy(dtype=int);times=pd.to_datetime(obs.timestamp,utc=True);row_t=(times.dt.hour*12+times.dt.minute//5).to_numpy(dtype=int);aug=obs.copy();aug[COLUMNS]=aug[COLUMNS].astype('float64');aug.loc[hidden[row_i,row_t],COLUMNS]=np.nan
   # 关键：端点、邻居、稳健统计、finite标记和插值基底全都重算。
   augmented,newbase=self.features(date,aug);array(folder/'aug_x.npy',augmented);array(folder/'aug_base.npy',newbase);extra=hidden.T&eligible.T;extra_y=np.zeros((288,e.n,2),dtype='float32');v=np.swapaxes(observation[:,:,:2],0,1).copy();v[:,:,1]/=e.lanes[None,:];v/=np.array([25,600]);extra_y[extra]=v[extra];assert np.isfinite(extra_y).all();array(folder/'extra_mask.npy',extra);array(folder/'extra_y.npy',extra_y)
  # 标签最后读，只写官方目标；增强额外监督来自被遮挡的已发布合格观测。
  truth=pd.read_parquet(e.truth_files[date]);y=np.zeros((288,e.n,2),dtype='float32')
  for c,col in enumerate(['speed_kmh','flow_vph']):
   a=e.grid(truth,col)/(25 if c==0 else e.lanes[:,None]*600);y[jj,ii,c]=a[ii,jj]
  assert np.isfinite(y).all() and (e.grid(truth,'pct_observed')[ii,jj]>=75).all();array(folder/'y.npy',y);array(folder/'mask.npy',mask);r=keys.mask_regime.unique();assert len(r)==1;save(folder/'ready.json',dict(date=date,rows=len(keys),regime=['R1','R2','R3'].index(r[0]),augmentation=info,feature_columns=372,history_dates=PLAN['history_dates']))

 def get(self,date,name):
  key=(date,name)
  if key not in self.arrays:self.arrays[key]=np.load(self.folder/date/(name+'.npy'),mmap_mode='r',allow_pickle=False)
  return self.arrays[key]
 def normalization(self):
  path=self.folder/'normalization.npz'
  if path.exists():
   with np.load(path) as a:self.mean=a['mean'];self.scale=a['scale']
   return
  sums=np.zeros(186);sq=np.zeros(186);count=np.zeros(186)
  for date in TRAIN:
   x=np.asarray(self.get(date,'static_x'),dtype='float64');ok=np.isfinite(x);v=np.where(ok,x,0);sums+=v.sum((0,1));sq+=(v*v).sum((0,1));count+=ok.sum((0,1))
  self.mean=(sums/np.maximum(count,1)).astype('float32');self.scale=np.sqrt(np.maximum(sq/np.maximum(count,1)-self.mean**2,0)).astype('float32');self.scale[self.scale<1e-3]=1.;tmp=path.with_suffix('.tmp')
  with tmp.open('wb') as f:np.savez(f,mean=self.mean,scale=self.scale,fit_dates=np.array(TRAIN))
  replace(tmp,path)
 def graph_for(self,nodes):
  key=tuple(nodes.tolist())
  if key not in self.graph_cache:self.graph_cache[key]={k:v.cuda() for k,v in graph_tensors(self.graph,nodes).items()}
  return self.graph_cache[key]
 def block(self,date,start,nodes,aux=None):
  ticks=np.arange(start,start+24);valid=(ticks>=0)&(ticks<288);ticks=ticks.clip(0,287);aug=aux is not None and float(aux.random())<CONFIG['augmented_view_probability'];prefix='aug' if aug else 'static';raw=np.array(self.get(date,prefix+'_x')[np.ix_(ticks,nodes)],dtype='float32');raw[~valid]=np.nan;finite=np.isfinite(raw);x=np.concatenate([np.where(finite,((raw-self.mean)/self.scale).clip(-8,8),0),finite],axis=-1).astype('float32');base=np.array(self.get(date,prefix+'_base')[np.ix_(ticks,nodes)]);base[~valid]=0;original=np.array(self.get(date,'mask')[np.ix_(ticks,nodes)]);original[~valid]=False;y=np.array(self.get(date,'y')[np.ix_(ticks,nodes)]);y[~valid]=0;mask=original.copy();extras=0
  if aug:
   extra=np.array(self.get(date,'extra_mask')[np.ix_(ticks,nodes)]);extra[~valid]=False;values=np.array(self.get(date,'extra_y')[np.ix_(ticks,nodes)]);mask,y=choose_supervision(original,extra,y,values,aux,.2);extras=int((mask&~original).sum());assert int(mask.sum())==int(original.sum())
  return dict(features=x,base=base,time_valid=valid,target=mask,label=y),dict(augmented=aug,extra_targets=extras,original_targets=int(original.sum()),retained_targets=int((mask&original).sum()))
 def batch(self,rng,aux):
  nodes,samples=sampling_spec(rng,self.n,self.graph['order'],TRAIN,24,64,2,True);parts=[];infos=[]
  for date,start in samples:
   p,info=self.block(date,start,nodes,aux);parts.append(p);infos.append(info)
  batch={k:torch.from_numpy(np.stack([p[k] for p in parts])).cuda() for k in parts[0]};return batch,self.graph_for(nodes),infos

def metric(acc):
 rows=[]
 for r,(n,sv,sq) in enumerate(acc):
  assert n>0;v,q=np.sqrt(np.array([sv,sq])/n);rows.append(dict(regime='R'+str(r+1),rows=int(n),speed_rmse_kmh=float(v*25),flow_rmse_per_lane_vph=float(q*600),S_state=float(.54*max(0,1-v)+.46*max(0,1-q))))
 return dict(S_state=float(np.mean([r['S_state'] for r in rows])),rows=sum(r['rows'] for r in rows),regimes=rows)
@torch.inference_mode()
def evaluate(model,data,dates,save_predictions=False):
 model.eval();acc=np.zeros((3,3));daily=[];nodes=np.arange(data.n);graph=data.graph_for(nodes);starts=list(range(0,265,12));frames=[]
 for date in dates:
  total=np.zeros((288,data.n,2));count=np.zeros((288,1,1))
  for offset in range(0,len(starts),4):
   ss=starts[offset:offset+4];parts=[data.block(date,s,nodes)[0] for s in ss];batch={k:torch.from_numpy(np.stack([p[k] for p in parts])).cuda() for k in parts[0]};pred=model(batch,graph).cpu().numpy()
   for s,p in zip(ss,pred):total[s:s+24]+=p;count[s:s+24]+=1
  pred=total/count;mask=np.array(data.get(date,'mask'));y=np.array(data.get(date,'y'));error=pred[mask]-y[mask];r=json.loads((data.folder/date/'ready.json').read_text())['regime'];v=[len(error),float(np.square(error[:,0]).sum()),float(np.square(error[:,1]).sum())];acc[r]+=v;daily.append(dict(date=date,regime=r,rows=v[0],speed_sse=v[1],flow_sse=v[2]))
  if save_predictions:
   keys=data.keys[date].copy();ii=keys.station_id.map(data.exp.si).to_numpy(dtype=int);ts=pd.to_datetime(keys.timestamp,utc=True);jj=(ts.dt.hour*12+ts.dt.minute//5).to_numpy(dtype=int);values=pred[jj,ii];keys['speed_kmh']=values[:,0]*25;keys['flow_vph']=values[:,1]*600*data.exp.lanes[ii];frames.append(keys)
 result=metric(acc);result['days']=daily
 if frames:pd.concat(frames,ignore_index=True).to_parquet(ROOT/'results'/data.road/'holdout-predictions.parquet',index=False)
 return result

def fit(data):
 folder=ROOT/'results'/data.road;folder.mkdir(parents=True,exist_ok=True);torch.manual_seed(CONFIG['seed']);model=PriorityTransformer(CONFIG,'T1_balanced').cuda()
 if (folder/'fit.json').exists():model.load_state_dict(torch.load(folder/'best.pt',map_location='cuda',weights_only=False)['model']);return model
 optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,60*1024,eta_min=.00003);rng=np.random.default_rng(2026);aux=np.random.default_rng(2837);trace=[];best=-1.;best_epoch=0;stale=0;first=0;steps=0;old_seconds=0.;start=time.time();counts=dict(augmented_windows=0,static_windows=0,extra_targets=0,original_targets=0,retained_targets=0)
 if (folder/'last.pt').exists():
  c=torch.load(folder/'last.pt',map_location='cpu',weights_only=False);model.load_state_dict(c['model']);optimizer.load_state_dict(c['optimizer']);scheduler.load_state_dict(c['scheduler']);rng.bit_generator.state=c['rng'];aux.bit_generator.state=c['aux_rng'];torch.set_rng_state(c['torch_rng']);torch.cuda.set_rng_state_all(c['cuda_rng']);first=c['epoch'];steps=c['steps'];best=c['best'];stale=c['stale'];best_epoch=c['best_epoch'];trace=c['trace'];counts=c['exposure'];old_seconds=c.get('seconds',0);del c
 for epoch in range(first,60):
  if stale>=8:break
  model.train();losses=[];tick=time.time()
  for step in range(1024):
   batch,graph,info=data.batch(rng,aux);optimizer.zero_grad(set_to_none=True);pred=model(batch,graph);loss=objective(pred,batch);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();scheduler.step();steps+=1;losses.append(float(loss.detach()))
   for i in info:
    counts['augmented_windows' if i['augmented'] else 'static_windows']+=1
    for k in ['extra_targets','original_targets','retained_targets']:counts[k]+=i[k]
   if step%128==0:progress('training_jayden_186',panel=data.road,epoch=epoch+1,epochs=60,step=steps,batch=step+1,batches=1024,loss=float(loss.detach()),learning_rate=scheduler.get_last_lr()[0])
  progress('inner_validation',panel=data.road,epoch=epoch+1,step=steps);score=evaluate(model,data,INNER)['S_state'];improved=score>best+1e-7;stale=0 if improved else stale+1
  if improved:best=score;best_epoch=epoch+1;checkpoint(folder/'best.pt',dict(model=model.state_dict(),epoch=epoch+1,steps=steps,inner_S_state=score))
  trace.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),inner_S_state=score,steps=steps,seconds=time.time()-tick,learning_rate=scheduler.get_last_lr()[0]));save(folder/'trace.json',trace)
  # 模型很小；每5轮/早停/结束保存恢复状态，不重复写大型全模型状态。
  if (epoch+1)%5==0 or stale>=8 or epoch+1==60:checkpoint(folder/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=rng.bit_generator.state,aux_rng=aux.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),epoch=epoch+1,steps=steps,best=best,best_epoch=best_epoch,stale=stale,trace=trace,exposure=counts,seconds=old_seconds+time.time()-start))
  progress('jayden_epoch_complete',panel=data.road,epoch=epoch+1,step=steps,inner_S_state=score,best_epoch=best_epoch,best_score=best,stale=stale,seconds=time.time()-tick)
 model.load_state_dict(torch.load(folder/'best.pt',map_location='cuda',weights_only=False)['model']);save(folder/'fit.json',dict(parameters=sum(p.numel() for p in model.parameters()),epochs=len(trace),actual_optimizer_updates=steps,best_epoch=best_epoch,best_inner_S_state=best,seconds=old_seconds+time.time()-start,exposure=counts,initialization='new random372input T1; no old checkpoint',config=CONFIG));return model

def run():
 preflight(True);assert json.loads((ROOT/'smoke.json').read_text())['passed'];all_results=[]
 for road in ROADS:
  folder=ROOT/'results'/road;folder.mkdir(parents=True,exist_ok=True)
  if (folder/'complete.json').exists():all_results.append(json.loads((folder/'complete.json').read_text()));continue
  progress('preparing_road',panel=road);data=Data(road)
  for i,d in enumerate(TRAIN+INNER):
   data.prepare(d)
   if i%10==0:progress('preparing_186_views',panel=road,day=d,days_done=i+1,days_total=190)
  progress('fitting_training_normalization',panel=road);data.normalization();model=fit(data);save(folder/'frozen-before-holdout.json',dict(checkpoint_sha256=sha(folder/'best.pt'),selection_dates=INNER,training_dates=TRAIN));data.allow_holdout=True
  for i,d in enumerate(HOLD):
   data.prepare(d)
   if i%15==0:progress('preparing_holdout',panel=road,day=d,days_done=i+1,days_total=55)
  progress('holdout_evaluation',panel=road);result=evaluate(model,data,HOLD,True);ref=pd.read_csv(ROOT/'reference-transformer-metrics.csv');ref=ref[(ref.panel==road)&(ref.regime=='macro_mean')];result.update(panel=road,reference_original_transformer=float(ref.iloc[0].S_state),fit=json.loads((folder/'fit.json').read_text()),protocol='28history/168train/22selection/55holdout; holdoutpreviouslyviewed; newmethodnotreplication');result['gain_vs_original']=result['S_state']-result['reference_original_transformer'];save(folder/'complete.json',result);all_results.append(result);progress('road_complete',panel=road,S_state=result['S_state'],gain_vs_original=result['gain_vs_original']);del model,data;gc.collect();torch.cuda.empty_cache()
 save(ROOT/'results.json',dict(roads=all_results,mean_two_roads=float(np.mean([r['S_state'] for r in all_results])),reference_mean_two_roads=float(np.mean([r['reference_original_transformer'] for r in all_results])),scope='eight newly trained roads; full ten-road summary is generated separately'));progress('complete')

def smoke():
 preflight();records=[]
 for road in ROADS:
  data=Data(road);date=TRAIN[0];data.prepare(date);a=np.asarray(data.get(date,'static_x'),dtype='float64');data.mean=np.nan_to_num(np.nanmean(a,(0,1))).astype('float32');data.scale=np.nanstd(a,(0,1)).astype('float32');data.scale[~np.isfinite(data.scale)|(data.scale<1e-3)]=1.;torch.manual_seed(2026);model=PriorityTransformer(CONFIG,'T1_balanced').cuda();rng=np.random.default_rng(2026);aux=np.random.default_rng(1)
  # 只用一个真实训练日做前反向；不进入正式epoch。
  nodes=data.graph['order'][:64];parts=[]
  for start in [-23,276]:parts.append(data.block(date,start,nodes,aux)[0])
  batch={k:torch.from_numpy(np.stack([p[k] for p in parts])).cuda() for k in parts[0]};g=data.graph_for(nodes);pred=model(batch,g);loss=objective(pred,batch);loss.backward();opt=torch.optim.AdamW(model.parameters(),lr=.0003);opt.step();assert torch.isfinite(loss) and batch['features'].shape[-1]==372
  # 遮挡视图的额外监督必须来自已发布且合格的观测，监督替换不增加数量。
  extra=np.array(data.get(date,'extra_mask'));original=np.array(data.get(date,'mask'));assert not (extra&original).any();assert data.block(date,-23,nodes)[0]['time_valid'].sum()==1
  # 逐个原遮挡点注入极端值再遮回，186特征应逐位不变，证明无原端点残留。
  e=data.exp;obs=pd.read_parquet(e.masked_files[date]);ii=obs.station_id.map(e.si).to_numpy(dtype=int);ts=pd.to_datetime(obs.timestamp,utc=True);jj=(ts.dt.hour*12+ts.dt.minute//5).to_numpy(dtype=int);rows=extra.T[ii,jj] if False else extra[jj,ii];perturbed=obs.copy();perturbed[COLUMNS]=perturbed[COLUMNS].astype('float64');perturbed.loc[rows,COLUMNS]=1e6;perturbed.loc[rows,COLUMNS]=np.nan;x,b=data.features(date,perturbed);assert np.array_equal(x,np.asarray(data.get(date,'aug_x')),equal_nan=True) and np.array_equal(b,np.asarray(data.get(date,'aug_base')))
  records.append(dict(road=road,input_columns=372,parameters=sum(p.numel() for p in model.parameters()),loss=float(loss.detach()),boundary_padding_checked=True,augmentation_no_original_target_overlap=True,hidden_value_perturbation_invariance=True,ephemeral_optimizer_updates=1));del model,opt,data;gc.collect();torch.cuda.empty_cache()
 save(ROOT/'smoke.json',dict(passed=True,records=records,main_sha256=sha(Path(__file__)),formal_training_started=False));progress('smoke_complete')
if __name__=='__main__':
 torch.set_num_threads(4);torch.manual_seed(2026);random.seed(2026);np.random.seed(2026);sys.stdout.reconfigure(encoding='utf8');torch.backends.cuda.matmul.allow_tf32=False
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['smoke','run']);a=parser.parse_args()
 import msvcrt
 guard=(ROOT/'training.lock').open('a+b');guard.write(b'0');guard.flush();guard.seek(0)
 try:msvcrt.locking(guard.fileno(),msvcrt.LK_NBLCK,1)
 except OSError:print('已有本实验进程，拒绝重复启动。',flush=True);sys.exit(2)
 try:smoke() if a.mode=='smoke' else run()
 except Exception as e:save(ROOT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));progress('failed',error=repr(e));raise
 except KeyboardInterrupt:progress('interrupted_by_user');raise
