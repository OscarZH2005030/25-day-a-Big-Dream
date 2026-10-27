"""186特征训练入口：十道路全局训练 → 各道路从同一全局模型独立适配。

由用户手动启动：python train_global.py run（推荐使用同目录的“启动训练.cmd”）。
smoke 只做两日数据与一次前反向验收，不是正式训练，也不产生新成绩。
旧数据目录只读；特征源码放在同目录 feature-source；缓存写入本聊天 work/cache。
配置集中在 CONFIG 中；详细操作、结果位置和恢复限制见“训练说明.txt”。
"""
import os
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
import sys,json,time,datetime,hashlib,traceback,copy,gc,random,argparse,warnings
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';CACHE=ROOT/'work/cache';BUNDLE=OUT/'feature-source'
DATA=Path(r'C:\Users\18049\Documents\Codex\2026-10-22\trafficflowbench-kaggle-rtx-5080-16gb-1\work\competition-data\kaggle_public')
sys.path[:0]=[str(BUNDLE/'code/original'),str(BUNDLE/'code/features')]
from task1_baseline import Experiment,KEYS
from augment_features import append_features
from feature_bank import generate
from frozen_v2_features import transform
SCHEMA=json.loads((BUNDLE/'feature_schema.json').read_text());BASE=json.loads((BUNDLE/'base_feature_schema.json').read_text())
PANELS=json.loads((BUNDLE/'split.json').read_text())['panels']
DATES=pd.date_range('2030-06-01','2031-02-28').strftime('%Y-%m-%d').tolist();TRAIN=DATES[:204];VALID=DATES[204:]
# 配置说明：CPU限4线程；LightGBM按“道路×日期”抽样训练，以免大路主导。
# 下面的600/180是全局/适配树上限；18/12是Transformer全局/适配轮数上限。
# 早停只用69日选模验证集；最终报告对69日全部官方目标行重新计算。
# Transformer每轮每个训练日随机取一个24步窗口，批量4窗口；不是每天全288步都更新。
# 12步/小时，核心24步=2小时，两侧各6步=30分钟上下文。不同道路不混同图。
CONFIG = {'seed': 20261024,
 'train_days': 204,
 'validation_days': 69,
 'train_dates': ['2030-06-01', '2030-12-21'],
 'validation_dates': ['2030-12-22', '2031-02-28'],
 'validation_use': 'model selection/early stopping, not independent blind test',
 'history': 'masked released observations from 204 training dates only; no hidden targets; '
            'fixed for both splits',
 'lgb': {'global_rounds': 600,
         'adapt_rounds': 180,
         'patience': 45,
         'learning_rate': 0.04,
         'adapt_learning_rate': 0.015,
         'train_cap_per_road_day': 1500,
         'selection_cap_per_road_day': 1000,
         'num_threads': 4,
         'num_leaves': 63,
         'max_bin': 127},
 'transformer': {'width': 128,
                 'depth': 4,
                 'heads': 4,
                 'global_epochs': 18,
                 'adapt_epochs': 12,
                 'patience': 4,
                 'batch_windows': 4,
                 'core_slots': 24,
                 'halo_slots': 6,
                 'windows_per_training_day_per_epoch': 1,
                 'global_lr': 0.0003,
                 'adapt_lr': 5e-05,
                 'weight_decay': 0.01,
                 'checkpoint_every_epochs': 5},
 'fusion': 'none; each branch reported separately',
 'score': {'speed_weight': 0.54,
           'flow_weight': 0.46,
           'speed_scale': 25,
           'flow_per_lane_scale': 600,
           'aggregation': 'R1/R2/R3 macro then equal panels (five equal two-direction '
                          'families)'}}
def expand_fast(matrix):
 """Only the two selected families, vectorized six-point NaN quantiles."""
 assert SCHEMA['frozen_spec']=={'masked':[],'removed':[],'added':['gap_reliability','multiscale_robust'],'channel_masked':{'speed':[],'flow_per_lane':[]}}
 ix={n:i for i,n in enumerate(BASE['all_numeric_features'])};extra={}
 def get(n):return matrix[:,ix[n]]
 def put(n,v):extra['night_'+n]=np.where(np.isfinite(v),np.clip(v,-1e6,1e6),np.nan).astype('float32')
 def quantiles(a,qs):
  valid=np.isfinite(a);count=valid.sum(1);s=np.sort(np.where(valid,a,np.inf),axis=1);res=[]
  for q in qs:
   rank=np.maximum(count-1,0)*q;lo=np.floor(rank).astype(int);hi=np.ceil(rank).astype(int);frac=rank-lo
   left=s[np.arange(len(a)),lo];right=s[np.arange(len(a)),hi]
   v=left.astype('float64')+(right.astype('float64')-left)*frac;v[count==0]=np.nan;res.append(v.astype('float32'))
  return res
 with np.errstate(invalid='ignore',divide='ignore',over='ignore'):
  for ch,scale in [('speed_kmh',25.),('flow_vph',600.)]:
   initial=get(ch+'_initial');left=get(ch+'_prev');right=get(ch+'_next');before=get(ch+'_offset_-5min');after=get(ch+'_offset_5min');gl=np.maximum(get(ch+'_prev_gap_min'),0);gr=np.maximum(get(ch+'_next_gap_min'),0);span=gl+gr+1;p=ch+'_'
   put(p+'weighted_endpoint_delta',((left-initial)*(gr+1)+(right-initial)*(gl+1))/((span+1)*scale));put(p+'endpoint_slope',(right-left)/((span+1)*scale));put(p+'endpoint_disagreement',np.abs(right-left)/((np.sqrt(span)+1)*scale));put(p+'near_slope_confidence',(after-before)/(scale*(1+span/10)));put(p+'gap_balance_log',np.log1p(gl)-np.log1p(gr))
  for ch,scale in [('speed_kmh',25.),('flow_vph',600.)]:
   a=np.column_stack([get(ch+'_offset_'+str(o)+'min') for o in [-30,-15,-5,5,15,30]]);median,lower,upper=quantiles(a,[.5,.25,.75]);mad=quantiles(np.abs(a-median[:,None]),[.5])[0];center=get(ch+'_initial');p=ch+'_'
   for n,v in [('median_residual',(median-center)/scale),('iqr',(upper-lower)/scale),('mad',mad/scale),('robust_z',(center-median)/(mad+scale*.05)),('lower_tail',(lower-center)/scale),('upper_tail',(upper-center)/scale)]:put(p+n,v)
 numeric=np.column_stack(list(extra.values()));return np.concatenate([matrix,numeric,np.isfinite(numeric)],axis=1).astype('float32'),BASE['input_names']+list(extra)+[n+'__finite' for n in extra]
def save(path,v):
 # Windows监控软件可能短时占用JSON；临时文件+原子替换，并有限重试。
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf8')
 for i in range(40):
  try:os.replace(tmp,path);return
  except PermissionError:
   if i==39:raise
   time.sleep(.05)
def progress(phase,**kw):
 v=dict(phase=phase,pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw)
 try:save(OUT/'progress.json',v)
 except PermissionError:print('progress write skipped after bounded lock retries',flush=True)
 print(json.dumps(v),flush=True)
def save_npz(path,**arrays):
 # 缓存也先写临时文件；断电/中断不会留下“存在但不完整”的日期文件。
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
 for i in range(40):
  try:os.replace(tmp,path);return
  except PermissionError:
   if i==39:raise
   time.sleep(.05)
def exp(panel):return Experiment(SimpleNamespace(release_root=DATA,panel=panel,output=CACHE/panel/'metadata',seed=CONFIG['seed']))
def profile(e):
 # 重建历史均值：仅训练日的已发布掩码观测。不加载随包history.npz。
 # 不读训练隐藏目标标签；不汇总任何验证日期，避免验证标签进入特征。
 sums=np.zeros((2,e.n,7,288));counts=np.zeros_like(sums)
 for d in TRAIN:
  obs=pd.read_parquet(e.masked_files[d]);wd=pd.Timestamp(d).dayofweek
  for c,name in enumerate(['speed_kmh','flow_vph']):
   a=e.grid(obs,name)/(e.lanes[:,None] if c else 1);good=np.isfinite(a)&(e.grid(obs,'pct_observed')>=75)
   sums[c,:,wd]+=np.where(good,a,0);counts[c,:,wd]+=good
 fallback=sums.sum((2,3))/np.maximum(counts.sum((2,3)),1)
 history=np.divide(sums,counts,out=np.broadcast_to(fallback[:,:,None,None],sums.shape).copy(),where=counts>0)
 assert np.isfinite(history).all();e.history=history
 return history
def keys_all(e,panel):
 # 目标行以官方sample_submission_state.csv为准，不能用所有空值代替。
 p=CACHE/panel/'keys.parquet'
 if p.exists():k=pd.read_parquet(p)
 else:
  chunks=[]
  for chunk in pd.read_csv(DATA/'task1'/panel/'train/sample_submission_state.csv',dtype=str,usecols=KEYS,chunksize=250000):chunks.append(chunk)
  k=pd.concat(chunks,ignore_index=True);assert not k.duplicated(KEYS).any();k.to_parquet(p,index=False)
 groups={d:g for d,g in k.groupby(k.timestamp.str[:10])};assert sorted(groups)==DATES
 return groups
def build(e,d,keys):
 # 第一步只读掩码输入：日历、道路几何、观测端点、时空邻居及186项衍生量。
 # Task1是离线重建，允许同日已观测的未来端点；这不是因果预测模型。
 values,bases,_,obs=e.feature_grid(d)
 assert list(values)==BASE['original_features'];append_features(e,d,values,obs)
 raw=np.stack(list(values.values()),axis=-1).transpose(1,0,2)
 full,names,_=generate(raw,list(values));numeric=full[...,[names.index(n) for n in BASE['all_numeric_features']]].astype('float32')
 t,n,_=numeric.shape;mat=numeric.reshape(-1,164)
 X,names=expand_fast(np.concatenate([mat,np.isfinite(mat)],axis=1))
 assert names==SCHEMA['input_names'] and X.shape[1]==372
 # 严格保留包内顺序：164数值、164标记、22新增数值、22新增标记。
 ni=[i for i,nm in enumerate(names) if not nm.endswith('__finite')];fi=[names.index(names[i]+'__finite') for i in ni]
 assert len(ni)==186 and np.array_equal(X[:,fi],np.isfinite(X[:,ni]))
 ii=keys.station_id.map(e.si).to_numpy(dtype=int);ts=pd.to_datetime(keys.timestamp,utc=True);jj=(ts.dt.hour*12+ts.dt.minute//5).to_numpy(dtype=int)
 assert np.array_equal(keys.link_id.to_numpy(),np.asarray(e.links)[ii]);assert len(np.unique(jj*n+ii))==len(keys)
 for col in ['speed_kmh','flow_vph','occupancy']:
  assert not np.isfinite(e.grid(obs,col)[ii,jj]).any(),(d,col,'target visible')
 base=np.stack([bases[0]/25,bases[1]/600],axis=-1).transpose(1,0,2).astype('float32');assert np.isfinite(base).all()
 numeric186=X[:,ni].reshape(t,n,186)
 # Transformer缓存用固定signed-log压缩后float16保存，降低磁盘写入。
 # 有约半精度量化误差；LightGBM评估会重新生成float32，不用此压缩缓存判断树阈值。
 packed=(np.sign(numeric186)*np.log1p(np.abs(numeric186))/5).astype('float16')
 mask=np.zeros((t,n),bool);mask[jj,ii]=True
 # 输入全部完成后才读真值；仅官方目标行进入标签与损失，流量先除车道数。
 truth=pd.read_parquet(e.truth_files[d]);y=np.zeros((t,n,2),dtype='float32')
 for c,col in enumerate(['speed_kmh','flow_vph']):
  g=e.grid(truth,col)/(25 if c==0 else e.lanes[:,None]*600);y[jj,ii,c]=g[ii,jj]
 assert np.isfinite(y).all();assert (e.grid(truth,'pct_observed')[ii,jj]>=75).all()
 regime=keys.mask_regime.unique();assert len(regime)==1
 return X.reshape(t,n,372),dict(x=packed,base=base,y=y,mask=mask,regime=np.uint8(['R1','R2','R3'].index(regime[0]))),ii,jj
def topology(e):
 n=e.n;adj=np.eye(n,dtype=bool)
 for groups in e.neighbors.values():
  for i,ids in enumerate(groups):adj[i,ids]=True
 adj=adj|adj.T;adj=(adj.astype(int)@adj.astype(int))>0
 return adj
def prepare(smoke=False):
 # 正式训练前先准备十条道路；复用原始数据，只新建本次派生缓存。
 assert json.loads((OUT/'data-audit.json').read_text())['all_data_match']
 if not smoke:
  session=OUT/'run-config-lock.json'
  if session.exists():assert json.loads(session.read_text())==CONFIG,'配置与已启动批次不同，请使用独立输出目录，避免混用旧缓存/模型'
  else:save(session,CONFIG)
 save(OUT/'training-config.json',CONFIG);save(OUT/'split-manifest.json',dict(training_dates=TRAIN,validation_dates=VALID,panels=PANELS,shared_split_all_models=True))
 for ri,panel in enumerate(PANELS):
  folder=CACHE/panel;folder.mkdir(parents=True,exist_ok=True)
  if (folder/'ready.json').exists() and not smoke:continue
  progress('preparing',panel=panel,road_index=ri);e=exp(panel)
  assert sorted(e.truth_files)==DATES and sorted(e.masked_files)==DATES
  hp=folder/'history.npy'
  if hp.exists():e.history=np.load(hp)
  else:np.save(hp,profile(e))
  keys=keys_all(e,panel);adj=topology(e);np.save(folder/'adj.npy',adj)
  mile=e.grid(pd.read_parquet(e.masked_files[DATES[0]]),'milepost')[:,0].astype('float32');assert np.isfinite(mile).all();position=(mile-mile.min())/max(float(np.ptp(mile)),1)
  np.save(folder/'position.npy',position)
  save(folder/'metadata.json',dict(stations=e.stations,links=e.links,lanes=e.lanes.tolist(),nodes=e.n,road_index=ri))
  dates=[TRAIN[0],VALID[0]] if smoke else DATES
  rows={'train':[],'valid':[]};rng=np.random.default_rng(CONFIG['seed']+ri)
  for di,d in enumerate(dates):
   daily=folder/(d+'.npz');sample=folder/(d+'.sample.npz')
   if daily.exists() and sample.exists():
    with np.load(sample) as previous:current_sampling=('sampling_version' in previous and int(previous['sampling_version'])==2)
    if current_sampling:
     if not smoke:rows['train' if d in TRAIN else 'valid'].append(d)
     continue
   X,pack,ii,jj=build(e,d,keys[d]);save_npz(daily,**pack)
   kind='train' if d in TRAIN else 'valid';cap=CONFIG['lgb']['train_cap_per_road_day' if kind=='train' else 'selection_cap_per_road_day']
   day_rng=np.random.default_rng(CONFIG['seed']+ri*10000+DATES.index(d));ids=np.sort(day_rng.choice(len(ii),min(cap,len(ii)),replace=False));x=X[jj[ids],ii[ids]]
   extra=np.column_stack([np.tile(np.eye(10,dtype='float32')[ri],(len(ids),1)),position[ii[ids]]])
   save_npz(sample,X=np.concatenate([x,extra],axis=1),y=pack['y'][jj[ids],ii[ids]],base=pack['base'][jj[ids],ii[ids]],regime=pack['regime'],sampling_version=np.int32(2))
   rows[kind].append(d)
   if di%10==0:progress('preparing',panel=panel,day=d,days_done=di+1,days_total=len(dates))
  if smoke:
   save(OUT/'preparation-smoke.json',dict(passed=True,panel=panel,dates=dates,feature_columns=372,numerics=186,finite_flags=186,nodes=e.n,hidden_channels_checked=['speed','flow','occupancy'],official_template_identity=True,history_sources=TRAIN));return
  save(folder/'ready.json',dict(days=rows,official_target_counts={d:len(keys[d]) for d in DATES},history_source='204 training masked daily parquets only'))
  del e;gc.collect()
 save(OUT/'feature-leakage-audit.json',dict(passed=True,feature_code='audited bundle modules; unused standalone profile_pair/main paths never invoked',history='recomputed from released masked training observations; no truth files, no validation dates',labels='read only after all inputs complete; speed/flow/occupancy target invisibility asserted every day',context='offline Task1 same-day observed past/future endpoints and graph neighbors allowed; no midnight wrapping',statistics='Transformer signed-log transform has no fitted statistics',frozen_models='not used',raw_numeric_columns=186,finite_flags=186,road_identity_columns=10,relative_position_columns=1,official_weights=CONFIG['score'],split='204/69 identical across roads; validation selection only'))

def sample_matrix(panels,kind):
 xs=[];ys=[];bs=[];rs=[];weights=[]
 for panel in panels:
  for d in (TRAIN if kind=='train' else VALID):
   with np.load(CACHE/panel/(d+'.sample.npz')) as a:
    xs.append(a['X']);ys.append(a['y']);bs.append(a['base']);rs.append(np.full(len(a['y']),int(a['regime']),dtype='uint8'))
   weights.append(np.full(len(xs[-1]),1/len(xs[-1]),dtype='float32'))
 return np.concatenate(xs),np.concatenate(ys),np.concatenate(bs),np.concatenate(rs),np.concatenate(weights)
def score_acc(acc):
 # 官方评分：每个R1/R2/R3分别算RMSE和S_state，再对三种掩码宏平均。
 # 官方契约确认速度权重0.54、流量0.46，修正交接文本中写反的权重。
 rows=[]
 for r,(n,sv,sq) in enumerate(acc):
  assert n>0;v,q=np.sqrt(np.asarray([sv,sq])/n);rows.append(dict(regime=['R1','R2','R3'][r],rows=int(n),speed_rmse_kmh=float(v*25),flow_rmse_per_lane_vph=float(q*600),S_state=float(.54*max(0,1-v)+.46*max(0,1-q))))
 return dict(S_state=float(np.mean([r['S_state'] for r in rows])),rows=sum(r['rows'] for r in rows),regimes=rows,speed_rmse_kmh=float(np.sqrt(sum(a[1] for a in acc)/sum(a[0] for a in acc))*25),flow_rmse_per_lane_vph=float(np.sqrt(sum(a[2] for a in acc)/sum(a[0] for a in acc))*600))
def evaluate_lgb(models,panel,dest):
 # 使用所有验证日的所有官方目标行；速度与每车道流量分别报告。
 start=time.time();e=exp(panel);e.history=np.load(CACHE/panel/'history.npy');keys=keys_all(e,panel);pos=np.load(CACHE/panel/'position.npy');ri=PANELS.index(panel);acc=np.zeros((3,3));days=[]
 for di,d in enumerate(VALID):
  X,a,ii,jj=build(e,d,keys[d]);x=X[jj,ii];x=np.column_stack([x,np.tile(np.eye(10)[ri],(len(ii),1)),pos[ii]]).astype('float32');base=a['base'][jj,ii];y=a['y'][jj,ii]
  pred=np.maximum(base+np.column_stack([m.predict(x,num_threads=4) for m in models]),0);err=(pred-y).astype('float64');r=int(a['regime']);v=[len(y),float((err[:,0]**2).sum()),float((err[:,1]**2).sum())];acc[r]+=v;days.append(dict(date=d,regime=r,rows=v[0],speed_sse=v[1],flow_sse=v[2]));
  if di%15==0:progress('evaluating_lightgbm',panel=panel,day=d,model=str(dest))
 result=score_acc(acc);result.update(panel=panel,branch='LightGBM',scope='all official target rows on 69 selection-validation dates',seconds=time.time()-start,days=days);save(dest/'metrics.json',result);return result
def fit_lgb(panels,dest,init=None):
 # 全局从零训练两棵回归器；道路适配从同一全局模型继续加树。
 # 383列=372列包内特征+10列道路one-hot+1列道路相对位置。
 dest.mkdir(parents=True,exist_ok=True);names=SCHEMA['input_names']+['road_'+p for p in PANELS]+['relative_position'];start=time.time()
 if (dest/'fit.json').exists():return [lgb.Booster(model_file=str(dest/(c+'.txt'))) for c in ['speed','flow']]
 progress('fitting_lightgbm',panel=panels[0] if len(panels)==1 else 'ALL',scope=str(dest))
 X,y,b,r,w=sample_matrix(panels,'train');V,vy,vb,vr,vw=sample_matrix(panels,'valid');models=[];reports=[]
 params=dict(objective='regression',metric='None',learning_rate=.015 if init else .04,num_leaves=63,min_data_in_leaf=80,lambda_l2=2,feature_fraction=.9,max_bin=127,num_threads=4,verbosity=-1,seed=CONFIG['seed'],deterministic=True,force_col_wise=True,feature_pre_filter=False)
 # 每个道路日等权，避免目标多的道路压过其他道路。
 for c,name in enumerate(['speed','flow']):
  if (dest/(name+'.json')).exists():models.append(lgb.Booster(model_file=str(dest/(name+'.txt'))));reports.append(json.loads((dest/(name+'.json')).read_text()));continue
  ts=lgb.Dataset(X,label=y[:,c]-b[:,c],weight=w/w.mean(),feature_name=names,free_raw_data=False,params={'max_bin':127,'feature_pre_filter':False});vs=lgb.Dataset(V,label=vy[:,c]-vb[:,c],reference=ts,free_raw_data=False)
  def feval(pred,data):
   error=np.maximum(vb[:,c]+pred,0)-vy[:,c];val=np.mean([min(1,float(np.sqrt(np.mean(error[vr==i]**2)))) for i in range(3)]);return 'macro_nrmse',val,False
  trace={};rounds=CONFIG['lgb']['adapt_rounds' if init else 'global_rounds'];initial=init[c] if init else None
  def tick(env):
   if env.iteration%25==0:progress('fitting_lightgbm',panel=panels[0] if len(panels)==1 else 'ALL',channel=name,step=env.iteration+1,learning_rate=params['learning_rate'])
  m=lgb.train(params,ts,num_boost_round=rounds,valid_sets=[vs],feval=feval,init_model=initial,callbacks=[lgb.early_stopping(45,verbose=False),lgb.record_evaluation(trace),tick])
  # 保留候选适配树；若它在同一选模子集更差，选用零新增树的全局模型。
  candidate_metric=feval(m.predict(V,num_threads=4),vs)[1];initial_metric=None;selected_global=False;candidate_trees=m.best_iteration
  if initial:
   initial_metric=feval(initial.predict(V,num_threads=4),vs)[1];m.save_model(str(dest/(name+'.adapt-candidate.txt')))
   if initial_metric<=candidate_metric:m=lgb.Booster(model_str=initial.model_to_string());selected_global=True
  m.save_model(str(dest/(name+'.txt')));models.append(m)
  report=dict(channel=name,selected_total_trees=m.num_trees(),candidate_best_total_trees=candidate_trees,initial_trees=initial.num_trees() if initial else 0,actual_boosting_rounds=len(trace['valid_0']['macro_nrmse']),trace=trace,learning_rate=params['learning_rate'],candidate_selection_nrmse=candidate_metric,global_selection_nrmse=initial_metric,selected_unchanged_global=selected_global);reports.append(report);save(dest/(name+'.json'),report)
  del ts,vs;gc.collect()
 save(dest/'fit.json',dict(panels=panels,parameters=params,train_rows=len(y),selection_rows=len(vy),feature_columns=len(names),seconds=time.time()-start,channels=reports,initialization='same saved global booster' if init else 'new from scratch',optimizer_updates='not applicable to trees'))
 return models

class Block(nn.Module):
 # 先沿时间做注意力，再沿道路拓扑邻接做空间注意力；最后前馈变换。
 def __init__(self,w):
  super().__init__();self.norms=nn.ModuleList([nn.LayerNorm(w) for _ in range(3)]);self.time=nn.MultiheadAttention(w,4,dropout=.05,batch_first=True);self.space=nn.MultiheadAttention(w,4,dropout=.05,batch_first=True);self.ff=nn.Sequential(nn.Linear(w,4*w),nn.GELU(),nn.Dropout(.05),nn.Linear(4*w,w))
 def forward(self,x,blocked):
  b,t,n,w=x.shape;z=self.norms[0](x).permute(0,2,1,3).reshape(b*n,t,w);z=self.time(z,z,z,need_weights=False)[0].reshape(b,n,t,w).permute(0,2,1,3);x=x+z;z=self.norms[1](x).reshape(b*t,n,w);z=self.space(z,z,z,attn_mask=blocked,need_weights=False)[0].reshape(b,t,n,w);x=x+z;return x+self.ff(self.norms[2](x))
class GlobalTransformer(nn.Module):
 # 同一个网络覆盖十路：共享编码器/图时空层/回归头，另加道路身份嵌入。
 # 输出在合法插值基线上学习残差；约111万参数，不复用旧训练权重。
 def __init__(self):
  super().__init__();w=128;self.encoder=nn.Linear(374,w);self.roads=nn.Embedding(10,w);self.blocks=nn.ModuleList([Block(w) for _ in range(4)]);self.head=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,2));nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
 def forward(self,x,base,road,blocked):
  h=self.encoder(x)+self.roads.weight[road];
  for block in self.blocks:h=block(h,blocked)
  return base+self.head(h)
def tensor_batch(panel,items):
 # 一个批次来自同一路，items=(日期,核心起点)。仅核心窗口目标参与损失。
 # 上下文来自同日；午夜外补零，不读取次日或前日。
 folder=CACHE/panel;pos=np.load(folder/'position.npy');arrs=[];bases=[];ys=[];masks=[]
 for d,s in items:
  lo=max(0,s-6);hi=min(288,s+30);padleft=6-(s-lo);padright=36-padleft-(hi-lo)
  with np.load(folder/(d+'.npz')) as a:
   z=a['x'][lo:hi].astype('float32');finite=np.isfinite(z);z=np.nan_to_num(z,nan=0);t,n,_=z.shape
   # Preserve bundle column order; append graph position and absolute time.
   x=np.concatenate([z[...,:164],finite[...,:164],z[...,164:],finite[...,164:],np.broadcast_to(pos[None,:,None],(t,n,1)),np.broadcast_to((np.arange(lo,hi)/287)[:,None,None],(t,n,1))],axis=-1).astype('float32')
   base=a['base'][lo:hi];y=a['y'][lo:hi];mask=a['mask'][lo:hi].copy();mask[np.arange(lo,hi)<s]=False;mask[np.arange(lo,hi)>=s+24]=False
  padding=((padleft,padright),(0,0));arrs.append(np.pad(x,padding+((0,0),)));bases.append(np.pad(base,padding+((0,0),)));ys.append(np.pad(y,padding+((0,0),)));masks.append(np.pad(mask,padding))
 return [torch.from_numpy(np.stack(v)).cuda(non_blocking=True) for v in [arrs,bases,ys,masks]]
def checkpoint(path,model,opt=None,**kw):
 # best.pt只保存模型权重；last.pt每5轮/结束/早停保存可恢复优化器状态。
 tmp=path.with_suffix('.tmp');torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict() if opt else None,**kw),tmp)
 for i in range(40):
  try:os.replace(tmp,path);return
  except PermissionError:
   if i==39:raise
   time.sleep(.05)
@torch.no_grad()
def evaluate_transformer(model,panels,dest=None,selection=False):
 model.eval();results=[]
 for panel in panels:
  ri=PANELS.index(panel);blocked=torch.from_numpy(~np.load(CACHE/panel/'adj.npy')).cuda();acc=np.zeros((3,3));days=[]
  for di,d in enumerate(VALID):
   # Fixed sparse selection windows speed early stopping; final reports use all rows.
   starts=[(di*24)%288] if selection else list(range(0,288,24))
   with np.load(CACHE/panel/(d+'.npz')) as a:r=int(a['regime'])
   day=np.zeros(3)
   for k in range(0,len(starts),4):
    x,b,y,mask=tensor_batch(panel,[(d,s) for s in starts[k:k+4]])
    with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(x,b,ri,blocked).clamp_min(0)
    error=(pred.float()-y)[mask].double();v=[len(error),float(error[:,0].square().sum()),float(error[:,1].square().sum())];acc[r]+=v;day+=v
   days.append(dict(date=d,regime=r,rows=int(day[0]),speed_sse=day[1],flow_sse=day[2]))
  result=score_acc(acc);result.update(panel=panel,branch='Transformer',days=days,scope='fixed sparse selection windows' if selection else 'all official target rows on 69 selection-validation dates');results.append(result)
  if dest:save(dest/(panel+'.json'),result)
 return float(np.mean([v['S_state'] for v in results])),results
def fit_transformer(panels,dest,global_path=None):
 # 每条道路都重新创建模型并加载同一global/best.pt；绝不接着上一条路适配。
 dest.mkdir(parents=True,exist_ok=True);model=GlobalTransformer().cuda();cfg=CONFIG['transformer'];adapt=global_path is not None
 if (dest/'fit.json').exists():model.load_state_dict(torch.load(dest/'best.pt',map_location='cuda',weights_only=False)['model']);return model
 if adapt:model.load_state_dict(torch.load(global_path,map_location='cuda',weights_only=False)['model'])
 opt=torch.optim.AdamW(model.parameters(),lr=cfg['adapt_lr' if adapt else 'global_lr'],weight_decay=.01);epochs=cfg['adapt_epochs' if adapt else 'global_epochs'];start=time.time();steps=0;best=-1.;best_epoch=0;start_epoch=0;trace=[]
 # 恢复last.pt中的优化器、更新步数和随机状态；最多重做未保存的4轮。
 if (dest/'last.pt').exists():
  st=torch.load(dest/'last.pt',map_location='cpu',weights_only=False);model.load_state_dict(st['model']);opt.load_state_dict(st['optimizer']);start_epoch=st['epoch'];steps=st['steps'];best=st['best'];best_epoch=st['best_epoch'];trace=st['trace'];torch.set_rng_state(st['rng']);torch.cuda.set_rng_state_all(st['cuda_rng'])
 else:
  # 道路适配允许选择零步全局基线，避免微调后反而更差。
  if adapt:best,_=evaluate_transformer(model,panels,selection=True);checkpoint(dest/'best.pt',model,epoch=0,steps=0,score=best)
 for epoch in range(start_epoch,epochs):
  model.train();rng=np.random.default_rng(CONFIG['seed']+epoch);batches=[]
  for panel in panels:
   items=[(d,int(rng.integers(0,265))) for d in TRAIN];rng.shuffle(items)
   batches.extend((panel,items[i:i+4]) for i in range(0,len(items),4))
  rng.shuffle(batches);loss_total=0.;epoch_start=time.time()
  for bi,(panel,items) in enumerate(batches):
   frac=(epoch+bi/len(batches))/epochs;lr=cfg['adapt_lr' if adapt else 'global_lr']*(.1+.9*.5*(1+np.cos(np.pi*frac)))
   for g in opt.param_groups:g['lr']=float(lr)
   x,b,y,mask=tensor_batch(panel,items);blocked=torch.from_numpy(~np.load(CACHE/panel/'adj.npy')).cuda();opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(x,b,PANELS.index(panel),blocked);err=(pred-y)[mask];loss=(err.square()*err.new_tensor([.54,.46])).sum(-1).mean()
   assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();steps+=1;loss_total+=float(loss.detach())
   if bi%25==0:progress('training_transformer',panel=panel,scope=str(dest),epoch=epoch+1,epochs=epochs,step=steps,batch=bi+1,batches=len(batches),learning_rate=float(lr),loss=float(loss.detach()),gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/1048576)
  score,_=evaluate_transformer(model,panels,selection=True);record=dict(epoch=epoch+1,score=score,loss=loss_total/len(batches),updates=steps,learning_rate=float(lr),seconds=time.time()-epoch_start);trace.append(record);save(dest/'trace.json',trace)
  if score>best+1e-6:best=score;best_epoch=epoch+1;checkpoint(dest/'best.pt',model,epoch=epoch+1,steps=steps,score=score)
  progress('transformer_epoch_complete',scope=str(dest),**record,best_epoch=best_epoch,best_score=best)
  if (epoch+1)%5==0 or epoch+1==epochs or epoch+1-best_epoch>=cfg['patience']:checkpoint(dest/'last.pt',model,opt,epoch=epoch+1,steps=steps,best=best,best_epoch=best_epoch,trace=trace,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
  if epoch+1-best_epoch>=cfg['patience']:break
 model.load_state_dict(torch.load(dest/'best.pt',map_location='cuda',weights_only=False)['model']);save(dest/'fit.json',dict(panels=panels,best_epoch=best_epoch,best_selection_score=best,actual_optimizer_updates=steps,epochs_completed=len(trace),parameters=sum(p.numel() for p in model.parameters()),batch_windows=4,core_slots=24,halo_slots=6,seconds=time.time()-start,learning_rate_schedule='cosine over actual batch updates, floor 10%; reinitialized for each adaptation',initialization='same saved global best checkpoint' if adapt else 'new random weights',gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/1048576,trace=trace));return model
def smoke():
 # 小规模验收：优化特征与原实现对齐、两日数据合法、GPU前反向、继续加树和评分单元检查。
 rng=np.random.default_rng(123);mat=rng.normal(size=(500,328)).astype('float32');mat[rng.random(mat.shape)<.35]=np.nan
 for name in ['speed_kmh_prev_gap_min','speed_kmh_next_gap_min','flow_vph_prev_gap_min','flow_vph_next_gap_min']:mat[:,BASE['all_numeric_features'].index(name)]=rng.uniform(0,500,500)
 original,on=transform(mat,SCHEMA['frozen_spec']);fast,fn=expand_fast(mat);assert on==fn;assert np.allclose(original,fast,equal_nan=True,rtol=2e-6,atol=2e-6)
 save(OUT/'feature-optimization-parity.json',dict(passed=True,rows=500,relative_tolerance=2e-6,absolute_tolerance=2e-6,max_absolute_difference=float(np.nanmax(np.abs(original-fast))),implementation='vectorized selected-family quantiles; same 372 feature order'))
 prepare(True);panel=PANELS[0];x,b,y,m=tensor_batch(panel,[(TRAIN[0],0),(TRAIN[0],120),(TRAIN[0],240),(TRAIN[0],264)]);model=GlobalTransformer().cuda();blocked=torch.from_numpy(~np.load(CACHE/panel/'adj.npy')).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0003)
 real=np.load(CACHE/panel/(TRAIN[0]+'.sample.npz'))['X'][:,:328];original,on=transform(real,SCHEMA['frozen_spec']);fast,fn=expand_fast(real);assert on==fn and np.allclose(original,fast,equal_nan=True,rtol=2e-6,atol=2e-6)
 save(OUT/'feature-optimization-parity.json',dict(passed=True,synthetic_rows=500,real_target_rows=len(real),relative_tolerance=2e-6,absolute_tolerance=2e-6,max_real_absolute_difference=float(np.nanmax(np.abs(original-fast))),implementation='vectorized selected-family quantiles; same 372 feature order'))
 with torch.autocast('cuda',dtype=torch.bfloat16):p=model(x,b,0,blocked);loss=((p-y)[m]**2).mean()
 loss.backward();opt.step();assert torch.isfinite(loss)
 # Continue boosting must preserve global tree prefix and change only independent copies.
 X,ty,tb,_,_=sample_matrix_smoke(panel);dt=lgb.Dataset(X,label=ty[:,0]-tb[:,0],free_raw_data=False);params=dict(objective='regression',verbosity=-1,num_threads=4,min_data_in_leaf=5);g=lgb.train(params,dt,num_boost_round=3);before=g.model_to_string();a=lgb.train(params,dt,num_boost_round=2,init_model=g);assert g.model_to_string()==before and a.num_trees()==5
 acc=np.array([[10,0,0],[10,10,0],[10,0,10]],float);assert abs(score_acc(acc)['S_state']-2/3)<1e-8
 save(OUT/'correctness-smoke.json',dict(passed=True,transformer_forward_backward_optimizer=True,loss=float(loss.detach()),actual_optimizer_updates=1,lightgbm_init_model_prefix_preserved=True,source_global_booster_unchanged=True,score_synthetic_unit_check=True,gpu=torch.cuda.get_device_name(0),torch=torch.__version__,lightgbm=lgb.__version__,parameters=sum(p.numel() for p in model.parameters())))
def sample_matrix_smoke(panel):
 with np.load(CACHE/panel/(TRAIN[0]+'.sample.npz')) as a:return a['X'],a['y'],a['base'],None,None
def run():
 # 总顺序：准备十路 → 全局LGB → 全局Transformer → 逐路独立适配两分支 → 汇总。
 torch.set_num_threads(4);torch.manual_seed(CONFIG['seed']);np.random.seed(CONFIG['seed']);random.seed(CONFIG['seed']);torch.backends.cuda.matmul.allow_tf32=True
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['smoke','run']);args=ap.parse_args()
 # 文件锁阻止误双击造成两个正式GPU任务；进程退出时Windows自动释放锁。
 import msvcrt
 lock=(OUT/'training.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
 try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 except OSError:
  print('本目录已经有训练/验收进程，请勿重复启动；不会覆盖现有进度。',flush=True);sys.exit(2)
 if args.mode=='smoke':smoke();progress('smoke_complete');return
 assert json.loads((OUT/'correctness-smoke.json').read_text())['passed'];prepare()
 global_lgb=fit_lgb(PANELS,OUT/'global/lightgbm')
 for panel in PANELS:
  p=OUT/'global/lightgbm'/panel
  if not (p/'metrics.json').exists():evaluate_lgb(global_lgb,panel,p)
 global_tf=fit_transformer(PANELS,OUT/'global/transformer');evaluate_transformer(global_tf,PANELS,OUT/'global/transformer/full-validation');del global_tf;gc.collect();torch.cuda.empty_cache()
 for panel in PANELS:
  dest=OUT/'adapted'/panel
  if (dest/'complete.json').exists():continue
  adapted=fit_lgb([panel],dest/'lightgbm',global_lgb)
  if not (dest/'lightgbm/metrics.json').exists():evaluate_lgb(adapted,panel,dest/'lightgbm')
  model=fit_transformer([panel],dest/'transformer',OUT/'global/transformer/best.pt');score,rs=evaluate_transformer(model,[panel],dest/'transformer/full-validation');save(dest/'complete.json',dict(panel=panel,transformer=rs[0],lightgbm=json.loads((dest/'lightgbm/metrics.json').read_text()),initialization='independently from identical saved global models'))
  del model,adapted;gc.collect();torch.cuda.empty_cache();progress('road_complete',panel=panel)
 summary={}
 for scope in ['global','adapted']:
  summary[scope]={}
  for branch in ['lightgbm','transformer']:
   rows=[]
   for panel in PANELS:
    p=(OUT/'global/lightgbm'/panel/'metrics.json' if branch=='lightgbm' else OUT/'global/transformer/full-validation'/(panel+'.json')) if scope=='global' else (OUT/'adapted'/panel/'lightgbm/metrics.json' if branch=='lightgbm' else OUT/'adapted'/panel/'transformer/full-validation'/(panel+'.json'))
    rows.append(json.loads(p.read_text()))
   summary[scope][branch]=dict(S_state=float(np.mean([r['S_state'] for r in rows])),panels=rows)
 save(OUT/'results.json',summary);progress('complete')
if __name__=='__main__':
 try:run()
 except Exception as ex:
  save(OUT/'failure.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(ex),traceback=traceback.format_exc()));progress('failed',error=repr(ex));raise
 except KeyboardInterrupt:
  progress('interrupted_by_user',note='用户中断；下次运行从最近已保存断点恢复');raise
