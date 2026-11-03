"""Public-only spatial and ramp context for offline Task1 state correction."""
from pathlib import Path
import numpy as np,pandas as pd

def extra_features(data_root,panel,date,p,split):
 assert not any(c.startswith('true') for c in p.columns),'Truth cannot be a feature'
 root=Path(data_root)/'corridors'/panel;net=root/'network';path=next((root/split/'mainline_states_masked').rglob('synthetic_mainline_'+date.replace('-','_')+'.parquet'))
 raw=pd.read_parquet(path,columns=['timestamp','station_id','link_id','speed_kmh','flow_vph','pct_observed','is_score_eligible']);raw.station_id=raw.station_id.astype(str);raw.link_id=raw.link_id.astype(str);raw.timestamp=pd.to_datetime(raw.timestamp,utc=True);raw['tick']=raw.timestamp.dt.hour*12+raw.timestamp.dt.minute//5
 fd=pd.read_csv(net/'fd_parameters.csv',dtype={'link_id':str}).groupby('link_id').mean(numeric_only=True);topo=pd.read_csv(net/'lwr_mainline_topology.csv',dtype={'link_id':str});order=topo.drop_duplicates('link_id').set_index('link_id').order_index.to_dict();mapping=raw[['station_id','link_id']].drop_duplicates().set_index('station_id').link_id.to_dict();stations=sorted(mapping,key=lambda st:(order.get(mapping[st],1e9),st));si={st:i for i,st in enumerate(stations)}
 t=pd.to_datetime(p.timestamp,utc=True);ti=(t.dt.hour*12+t.dt.minute//5).to_numpy();ii=p.station_id.map(si).to_numpy();idx=(ti,ii)
 def grid(col):return raw.pivot(index='tick',columns='station_id',values=col).reindex(index=range(288),columns=stations).to_numpy()
 rawv=grid('speed_kmh');rawq=grid('flow_vph');assert not np.isfinite(rawv[idx]).any() and not np.isfinite(rawq[idx]).any()
 raw['q_lane']=raw.flow_vph/raw.link_id.map(fd.lanes);valid=raw.is_score_eligible.astype(bool)&(raw.pct_observed>=75);raw.loc[~valid,['speed_kmh','flow_vph','q_lane']]=np.nan
 vg=grid('speed_kmh');qg=grid('q_lane');fv=vg.copy();fq=qg.copy();fv[idx]=p.speed_kmh;fq[idx]=p.flow_vph/p.lanes;vv=p.speed_kmh.to_numpy();qq=p.flow_vph.to_numpy()/p.lanes.to_numpy();parts=[]
 def spatial(g,lag):
  result=np.full_like(g,np.nan)
  if lag>0:result[:,:-lag]=g[:,lag:]
  else:result[:,-lag:]=g[:,:lag]
  return result
 for g,base,scale in [(vg,vv,25),(qg,qq,600),(fv,vv,25),(fq,qq,600)]:
  for lag in [-2,-1,1,2]:parts.append((spatial(g,lag)[idx]-base)/scale)
 density=fq/np.maximum(fv,1);current=qq/np.maximum(vv,1)
 for lag in [-2,-1,1,2]:parts.extend([(spatial(density,lag)[idx]-current)/50,np.isfinite(spatial(vg,lag)[idx]).astype(float)])
 # UTC keys throughout, fixing the old diagnostic's string-format mismatch.
 amap=pd.read_csv(net/'ramp_attachment_map.csv',dtype=str);amap=amap.rename(columns={'nearest_mainline_link_id':'link_id'});rp=next((root/split/'ramp_states').rglob('synthetic_ramp_'+date.replace('-','_')+'.parquet'));ramp=pd.read_parquet(rp,columns=['timestamp','ramp_link_id','ramp_type','flow_vph','pct_observed','is_score_eligible']);ramp.ramp_link_id=ramp.ramp_link_id.astype(str);ramp.timestamp=pd.to_datetime(ramp.timestamp,utc=True);ramp=ramp[ramp.is_score_eligible.astype(bool)&(ramp.pct_observed>=75)&ramp.flow_vph.notna()].merge(amap[['ramp_link_id','link_id']],on='ramp_link_id',validate='many_to_one')
 ramp['on']=ramp.ramp_type.str.upper().isin(['OR','ON']);ramp['off']=ramp.ramp_type.str.upper().isin(['FR','OFF']);ramp['on_flow']=np.where(ramp.on,ramp.flow_vph,0);ramp['off_flow']=np.where(ramp.off,ramp.flow_vph,0);g=ramp.groupby(['timestamp','link_id'])[['on_flow','off_flow','on','off']].sum();index=pd.MultiIndex.from_arrays([t,p.link_id]);aligned=g.reindex(index)
 aon=amap[amap.ramp_type.str.upper().isin(['OR','ON'])].groupby('link_id').size();aoff=amap[amap.ramp_type.str.upper().isin(['FR','OFF'])].groupby('link_id').size();non=p.link_id.map(aon).fillna(0).to_numpy();noff=p.link_id.map(aoff).fillna(0).to_numpy();oncount=aligned.on.fillna(0).to_numpy();offcount=aligned.off.fillna(0).to_numpy();on=np.where(non==0,0,aligned.on_flow.to_numpy());off=np.where(noff==0,0,aligned.off_flow.to_numpy());on[(non>0)&(oncount==0)]=np.nan;off[(noff>0)&(offcount==0)]=np.nan;scale=p.lanes.to_numpy()*600
 parts.extend([on/scale,off/scale,(on-off)/scale,np.where(non==0,1,oncount/np.maximum(non,1)),np.where(noff==0,1,offcount/np.maximum(noff,1)),(non>0).astype(float),(noff>0).astype(float)])
 up=spatial(fq,-1)[idx];down=spatial(fq,1)[idx];parts.extend([(up-down)/600,(up-down+(on-off)/p.lanes.to_numpy())/600])
 X=np.column_stack(parts).astype('float32');X[~np.isfinite(X)]=np.nan
 return X
