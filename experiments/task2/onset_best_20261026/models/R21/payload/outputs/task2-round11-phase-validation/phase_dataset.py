"""相位均衡代理人口：发布train观测；同方向/条件所有相位之间至少360分钟。"""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;R2=ROOT.parent/'task2-round2';sys.path.insert(1,str(R2))
from run_round2 import arrays,OLD,CACHE,PANELS
from task2q.cache import read_arrays,atomic_json
SEED=20261027;PER_PHASE=8
def build(role,progress):
 path=ROOT/f'{role}-phase-index.json'
 if path.exists():return json.loads(path.read_text())
 source=json.loads((OLD/'datasets'/role/'index.json').read_text());nets={e['panel']:e['network_key'] for e in source['samples']};net_sources={p:source['assets']['networks'][k]['metadata']['sources'] for p,k in nets.items()};candidates={(p,c,k):[] for p in PANELS for c in [0,1] for k in [0,1,2]};verified=0
 for key,record in source['assets']['days'].items():
  loaded=read_arrays(CACHE,key);assert loaded and loaded[1]['sha256']==record['sha256'];v=loaded[0];sources=record['metadata']['sources'];p=next(p for p,ns in net_sources.items() if all(sources.get(a)==b for a,b in ns.items()));net=arrays(nets[p]);speed=v['speed'];q=(speed<=net['threshold'])&np.isfinite(speed)&v['eligible'];m=v['eligible']&np.isfinite(speed)&np.isfinite(v['quality'])&(v['quality']>=75)
  for slot in range(12,282):
   if not (q[slot+1:slot+7]&m[slot+1:slot+7]).any():continue
   c=int(np.any(q[slot-12:slot].sum(0)>=2));origin=pd.Timestamp(int(v['times_ns'][slot]),tz='UTC').isoformat();candidates[(p,c,slot%3)].append(dict(panel=p,condition=c,origin_slot=slot,origin=origin,day_key=key,network_key=nets[p],has_future_queue=True,role=role,phase=slot%3,origin_minutes=int(v['times_ns'][slot]//60_000_000_000)))
  verified+=1
  if verified%100==0:progress('preparing_phase_balanced_proxy',role=role,days=verified)
 selected=[];counts={};warnings=[];rng=np.random.default_rng(SEED+{'tune':11,'calibrate':12,'audit':13}[role])
 for g in candidates.values():rng.shuffle(g)
 for p in PANELS:
  for c in [0,1]:
   group=[];byphase={k:0 for k in [0,1,2]}
   while any(byphase[k]<PER_PHASE for k in [0,1,2]):
    advanced=False
    for phase in [0,1,2]:
     if byphase[phase]>=PER_PHASE:continue
     pool=candidates[(p,c,phase)]
     while pool:
      e=pool.pop()
      if all(abs(e['origin_minutes']-a['origin_minutes'])>=360 for a in group):group.append(e);byphase[phase]+=1;advanced=True;break
    if not advanced:break
   selected.extend(group);counts[f'{p}/{c}']={str(k):n for k,n in byphase.items()}
   if any(n<PER_PHASE for n in byphase.values()):warnings.append(f'{p}/{c}: requested8 per phase, obtained{byphase}')
   times=sorted(e['origin_minutes'] for e in group);assert all(b-a>=360 for a,b in zip(times,times[1:]))
 doc=dict(samples=selected,role=role,requested_per_phase=PER_PHASE,counts=counts,warnings=warnings,verified_days=verified,scope='phase-balanced future-positive noisy public proxy; not hidden queue_true',minimum_spacing_minutes=360,spacing_applies_across_phases=True,same_day_history=True,source_month_previously_used=True)
 atomic_json(path,doc);return doc
