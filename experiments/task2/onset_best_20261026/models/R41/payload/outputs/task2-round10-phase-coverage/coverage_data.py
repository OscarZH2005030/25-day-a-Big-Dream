"""保留旧窗口，补齐5/10分钟起点相位；完整公共fit缓存，标签侧均衡抽样。"""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;R6=ROOT.parent/'task2-round6-ramp-physics';R5=ROOT.parent/'task2-round5-distribution';R2=ROOT.parent/'task2-round2';sys.path[1:1]=[str(R6),str(R5),str(R2)]
from run_round2 import sample,arrays,PANELS,CACHE,OLD
from task2q.cache import read_arrays,atomic_json
from special_features import features,latest_state
from ramp_features import extras
SEED=20261027;WORK=ROOT.parents[1]/'work/round10-phase';WORK.mkdir(parents=True,exist_ok=True)
def matrix(s):
 r,p=extras(s);return np.concatenate([features(s),r,p],axis=1)
def expanded_index(progress):
 path=ROOT/'expanded-fit.json'
 if path.exists():return json.loads(path.read_text())
 old=json.loads((R2/'expanded-fit.json').read_text())['samples'];assert all(e['origin_slot']%3==0 for e in old);manifest=json.loads((OLD/'datasets/fit/index.json').read_text());nets={e['panel']:e['network_key'] for e in old};net_sources={p:manifest['assets']['networks'][k]['metadata']['sources'] for p,k in nets.items()};candidates={p:[] for p in PANELS};verified=0
 for key,record in manifest['assets']['days'].items():
  loaded=read_arrays(CACHE,key);assert loaded and loaded[1]['sha256']==record['sha256'];v=loaded[0];src=record['metadata']['sources'];panel=next(p for p,ns in net_sources.items() if all(src.get(a)==b for a,b in ns.items()));net=arrays(nets[panel]);speed=v['speed'];q=(speed<=net['threshold'])&np.isfinite(speed)&v['eligible'];eligible=v['eligible']&np.isfinite(speed)&np.isfinite(v['quality'])&(v['quality']>=75);date=pd.Timestamp(int(v['times_ns'][0]),tz='UTC').strftime('%Y-%m-%d')
  for slot in range(12,282):
   if slot%3==0:continue
   m=eligible[slot+1:slot+7]
   if not m.any():continue
   c=int(np.any(q[slot-12:slot].sum(0)>=2));positive=bool((q[slot+1:slot+7]&m).any());origin=pd.Timestamp(int(v['times_ns'][slot]),tz='UTC').isoformat();candidates[panel].append(dict(panel=panel,day_key=key,network_key=nets[panel],origin_slot=slot,condition=c,has_future_queue=positive,origin=origin))
  verified+=1
  if verified%200==0:progress('scanning_new_origin_phases',days=verified)
 selected=[];audit={};rng=np.random.default_rng(SEED+1000)
 for p in PANELS:
  original=[dict(e,source='original') for e in old if e['panel']==p];bydate={}
  for e in original:bydate.setdefault((e['origin'][:10],e['condition']),[]).append(e['origin_slot'])
  groups={}
  for e in candidates[p]:
   slots=bydate.get((e['origin'][:10],e['condition']),[]);distance=min((abs(e['origin_slot']-s) for s in slots),default=288);e['nearest_original_same_condition_minutes']=distance*5;groups.setdefault((e['condition'],e['has_future_queue'],e['origin'][:7]),[]).append(e)
  for g in groups.values():rng.shuffle(g);g.sort(key=lambda e:e['nearest_original_same_condition_minutes']>=45)
  added=[]
  while len(added)<1024:
   advanced=False
   for k in sorted(groups):
    if groups[k] and len(added)<1024:added.append(dict(groups[k].pop(),source='added_phase'));advanced=True
   if not advanced:break
  assert len(added)==1024;combined=original+sorted(added,key=lambda e:e['origin']);assert len({(e['origin'][:10],e['origin_slot']) for e in combined})==2048;selected.extend(combined);audit[p]=dict(original_windows=1024,added_windows=1024,phase_mod3_counts={str(i):sum(e['origin_slot']%3==i for e in combined) for i in [0,1,2]},added_at_least_45min_from_original_same_condition=sum(e['nearest_original_same_condition_minutes']>=45 for e in added),original_six_hour_blocks=len({(e['condition'],e['origin'][:10],e['origin_slot']//72) for e in original}),expanded_six_hour_blocks=len({(e['condition'],e['origin'][:10],e['origin_slot']//72) for e in combined}))
 result=dict(samples=selected,counts=audit,verified_days=verified,scope='June-November only; original8192 retained plus8192 true observations at origin phase +5/+10 minutes; correlated windows are not independent events');atomic_json(path,result);return result
def group_dataset(all_entries,p,c,progress):
 folder=WORK/f'{p}-{c}';folder.mkdir(exist_ok=True);done=folder/'complete.json'
 if done.exists():
  meta=json.loads(done.read_text());return {k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in ['X','y','w','early','control_early','window_positive']},meta
 old=[e for e in all_entries if e['panel']==p and e['condition']==c and e['source']=='original'];new=[e for e in all_entries if e['panel']==p and e['condition']==c and e['source']=='added_phase'];allgroup=old+new;counts={b:sum(e['has_future_queue']==b for e in allgroup) for b in [False,True]};oldcounts={b:sum(e['has_future_queue']==b for e in old) for b in [False,True]};base=ROOT.parents[1]/f'work/round5-distribution/{p}-{c}';ramp=ROOT.parents[1]/f'work/round6-ramp-matrix/{p}-{c}';ox=np.concatenate([np.load(base/'X.npy',mmap_mode='r'),np.load(ramp/'ramp.npy',mmap_mode='r'),np.load(ramp/'physics.npy',mmap_mode='r')],axis=1);oy=np.asarray(np.load(base/'y.npy',mmap_mode='r'));ow=np.asarray(np.load(base/'w.npy',mmap_mode='r')).copy();windows=[];cursor=0;early=[];positive=[]
 for e in old:
  s=sample(e);m=s['m'].ravel();y=s['y'].ravel();_,queued=latest_state(s);flip=y.astype(bool)!=np.tile(queued,6);mandatory=(m&(y>0))|(m&flip if c==1 else False);k=int(mandatory.sum()+min(192,(m&~mandatory).sum()));assert int(oy[cursor:cursor+k].sum())==int((y*m).sum());pos=e['has_future_queue'];ow[cursor:cursor+k]*=oldcounts[pos]/counts[pos];isearly=e['origin'][:10]<'2030-10-01';early.extend([isearly]*k);positive.extend([pos]*k);windows.append(dict(origin=e['origin'],positive=pos,source='original',start=cursor,end=cursor+k,early=isearly));cursor+=k
 assert cursor==len(oy);oldrows=cursor;nx=[];ny=[];nw=[];rng=np.random.default_rng(SEED+2000+PANELS.index(p)*2+c)
 for i,e in enumerate(new):
  s=sample(e);xx=matrix(s);m=s['m'].ravel();y=s['y'].ravel();last,queued=latest_state(s);flip=y.astype(bool)!=np.tile(queued,6);mandatory=(m&(y>0))|(m&flip if c==1 else False);keep=np.flatnonzero(mandatory);negative=np.flatnonzero(m&~mandatory);risk=np.tile((last<.9)|(s['x'][-2,:,1]>.7)|(s['x'][-2,:,2]>.20),6);a=negative[risk[negative]];b=negative[~risk[negative]];ka=min(96,len(a));kb=min(192-ka,len(b));ka=min(192-kb,len(a));aa=rng.choice(a,ka,replace=False) if ka else np.array([],int);bb=rng.choice(b,kb,replace=False) if kb else np.array([],int);ids=np.concatenate([keep,aa,bb]);inv=np.concatenate([np.ones(len(keep)),np.full(ka,len(a)/ka if ka else 0),np.full(kb,len(b)/kb if kb else 0)]);pos=e['has_future_queue'];weights=inv/m.sum()*(.8 if pos else .2)/counts[pos];nx.append(xx[ids]);ny.append(y[ids]);nw.append(weights.astype('float32'));k=len(ids);isearly=e['origin'][:10]<'2030-10-01';early.extend([isearly]*k);positive.extend([pos]*k);windows.append(dict(origin=e['origin'],positive=pos,source='added_phase',start=cursor,end=cursor+k,early=isearly));cursor+=k
  if (i+1)%128==0:progress('building_added_phase_rows',panel=p,condition=c,windows=i+1,total=len(new))
 X=np.concatenate([ox,*nx]);Y=np.concatenate([oy,*ny]);W=np.concatenate([ow,*nw]);E=np.asarray(early,dtype='uint8');P=np.asarray(positive,dtype='uint8');C=E.copy();C[oldrows:]=0;data=dict(X=X,y=Y,w=W,early=E,control_early=C,window_positive=P);meta=dict(panel=p,condition=c,rows=len(Y),original_rows=oldrows,features=555,original_windows=len(old),expanded_windows=len(allgroup),class_counts={str(k):v for k,v in counts.items()},windows=windows,sampling='original cells retained; extra windows use all positives/ongoing transitions and IPW negatives')
 for k,v in data.items():np.save(folder/(k+'.npy'),v)
 atomic_json(done,meta);return data,meta
