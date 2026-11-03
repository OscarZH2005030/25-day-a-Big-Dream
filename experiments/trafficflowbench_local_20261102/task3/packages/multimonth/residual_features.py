"""Masked-observation features, shared with the frozen January residual study."""
from pathlib import Path
import numpy as np
import pandas as pd
KEYS=['panel','timestamp','station_id','link_id','mask_regime']

def build_features(data_root,panel,date,p,split):
 assert not any(c.startswith('true') for c in p.columns),'Truth is not a feature'
 data_root=Path(data_root)
 path=next((data_root/'corridors'/panel/split/'mainline_states_masked').rglob('synthetic_mainline_'+date.replace('-','_')+'.parquet'))
 raw=pd.read_parquet(path,columns=['timestamp','station_id','link_id','speed_kmh','flow_vph']);raw.station_id=raw.station_id.astype(str);raw.link_id=raw.link_id.astype(str)
 fd=pd.read_csv(data_root/'corridors'/panel/'network/fd_parameters.csv',dtype={'link_id':str}).groupby('link_id').mean(numeric_only=True);raw['q_lane']=raw.flow_vph/raw.link_id.map(fd.lanes)
 ts=pd.to_datetime(raw.timestamp,utc=True);raw['tick']=ts.dt.hour*12+ts.dt.minute//5;stations=sorted(raw.station_id.unique());si={s:i for i,s in enumerate(stations)}
 def grid(col):return raw.pivot(index='tick',columns='station_id',values=col).reindex(index=range(288),columns=stations).to_numpy()
 vg=grid('speed_kmh');qg=grid('q_lane');t=pd.to_datetime(p.timestamp,utc=True);ti=(t.dt.hour*12+t.dt.minute//5).to_numpy();ii=p.station_id.map(si).to_numpy();idx=(ti,ii)
 assert not np.isfinite(vg[idx]).any() and not np.isfinite(qg[idx]).any()
 vv=p.speed_kmh.to_numpy();qq=p.flow_vph.to_numpy()/p.lanes.to_numpy();parts=[vv/25,qq/600,np.sin(ti*2*np.pi/288),np.cos(ti*2*np.pi/288),t.dt.dayofweek.to_numpy()/6]
 for g,base,scale in [(vg,vv,25),(qg,qq,600)]:
  f=pd.DataFrame(g);prev=f.ffill().to_numpy();nxt=f.bfill().to_numpy();ticks=np.broadcast_to(np.arange(288)[:,None],g.shape).astype(float);observed=np.where(np.isfinite(g),ticks,np.nan);before=pd.DataFrame(observed).ffill().to_numpy();after=pd.DataFrame(observed).bfill().to_numpy();gl=ticks-before;gr=after-ticks
  interp=(prev*gr+nxt*gl)/(gl+gr+1e-12);parts.extend([(prev[idx]-base)/scale,(nxt[idx]-base)/scale,(interp[idx]-base)/scale,np.log1p(gl[idx]),np.log1p(gr[idx])])
  for lag in [-3,-1,1,3]:
   shift=np.full_like(g,np.nan)
   if lag>0:shift[:-lag]=g[lag:]
   else:shift[-lag:]=g[:lag]
   parts.append((shift[idx]-base)/scale)
  count=np.isfinite(g).sum(1);mean=np.nansum(g,axis=1)/np.maximum(count,1);mean[count==0]=np.nan;parts.append((mean[ti]-base)/scale)
 pf=p.link_id.map(fd.free_speed_kmh).to_numpy();cap=p.link_id.map(fd.capacity_vph).to_numpy()/p.lanes.to_numpy();kc=cap/np.maximum(pf,1);kj=p.link_id.map(fd.k_jam).to_numpy()/p.lanes.to_numpy();density=qq/np.maximum(vv,1);qfd=np.where(density<=kc,pf*density,cap/np.maximum(kj-kc,1e-6)*np.maximum(kj-density,0));parts.extend([vv/np.maximum(pf,1),qq/np.maximum(cap,1),(qfd-qq)/600])
 X=np.column_stack(parts).astype('float32');X[~np.isfinite(X)]=np.nan
 return X
