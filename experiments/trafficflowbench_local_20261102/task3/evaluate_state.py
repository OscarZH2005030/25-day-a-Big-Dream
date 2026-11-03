"""Evaluate a portable residual package on released train labels only.

The result is a development diagnostic, not official Task3/LWR or leaderboard.
"""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
KEYS=['panel','timestamp','station_id','link_id','mask_regime']

def metric(items,values):
 rows=[]
 for f,p in zip(items,values):
  err=p-f[['true_v','true_q']].to_numpy();err[:,1]/=f.lanes.to_numpy()
  z=f[['station_id','link_id','timestamp','length_km','true_v','true_q']].copy()
  z['predN']=p[:,1]/np.maximum(p[:,0],1)*z.length_km
  z['trueN']=z.true_q/z.true_v.clip(lower=1)*z.length_km
  z=z.sort_values(['station_id','link_id','timestamp']);group=z.groupby(['station_id','link_id'])
  dt=(pd.to_datetime(group.timestamp.shift(-1),utc=True)-pd.to_datetime(z.timestamp,utc=True)).dt.total_seconds()
  dn=(group.predN.shift(-1)-z.predN)-(group.trueN.shift(-1)-z.trueN);good=(dt==300)&dn.notna()
  rows.append(dict(regime=f.mask_regime.iloc[0],n=len(f),v_sse=float((err[:,0]**2).sum()),q_sse=float((err[:,1]**2).sum()),dn_n=int(good.sum()),dn_sse=float((dn[good]**2).sum())))
 r=pd.DataFrame(rows).groupby('regime').sum(numeric_only=True)
 vr=np.sqrt(r.v_sse/r.n);qr=np.sqrt(r.q_sse/r.n);dn=np.sqrt(r.dn_sse/r.dn_n.replace(0,np.nan))
 return dict(S_state=float((.54*(1-vr/25).clip(lower=0)+.46*(1-qr/600).clip(lower=0)).mean()),speed_RMSE=float(vr.mean()),flow_per_lane_RMSE=float(qr.mean()),station_target_deltaN_RMSE=float(dn.mean()),target_rows=int(r.n.sum()),adjacent_target_pairs=int(r.dn_n.sum()))

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--release-root',type=Path,required=True)
 p.add_argument('--package',type=Path,required=True)
 p.add_argument('--baseline-state',type=Path,nargs='+',required=True)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--start',default='2031-02-01');p.add_argument('--end',default='2031-02-28')
 p.add_argument('--panels',nargs='+')
 args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 sys.path.insert(0,str(args.package.resolve()))
 spec=importlib.util.spec_from_file_location('portable_residual',args.package/'apply_residual.py')
 model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)
 frames=[pd.read_parquet(path) if path.suffix=='.parquet' else pd.read_csv(path,dtype={k:str for k in KEYS}) for path in args.baseline_state]
 baseline=pd.concat(frames,ignore_index=True)[KEYS+['speed_kmh','flow_vph']]
 baseline[KEYS]=baseline[KEYS].astype(str)
 if baseline.duplicated(KEYS).any():raise ValueError('Duplicate baseline state keys')
 baseline=baseline[baseline.timestamp.str[:10].between(args.start,args.end)]
 rows=[];daily=[]
 for panel,f in baseline.groupby('panel',sort=True):
  if args.panels and panel not in args.panels:continue
  pp,mm=model.load_panel(panel)
  fd=pd.read_csv(args.release_root/'corridors'/panel/'network/fd_parameters.csv',dtype={'link_id':str}).groupby('link_id').lanes.mean()
  lengths=pd.read_csv(args.release_root/'corridors'/panel/'network/links.csv',dtype={'link_id':str}).drop_duplicates('link_id').set_index('link_id').length_km
  items=[];values={'baseline':[],'candidate':[]}
  for date,part in f.groupby(f.timestamp.str[:10],sort=True):
   # No labels are supplied to or read by the portable correction.
   corrected=model.correct(args.release_root,panel,date,part,'train',pp,mm)
   corrected=part[KEYS].merge(corrected,on=KEYS,validate='one_to_one',sort=False)
   truthpath=next((args.release_root/'corridors'/panel/'train/mainline_states').rglob('synthetic_mainline_'+date.replace('-','_')+'.parquet'))
   truth=pd.read_parquet(truthpath,columns=['timestamp','station_id','link_id','speed_kmh','flow_vph'])
   truth[['timestamp','station_id','link_id']]=truth[['timestamp','station_id','link_id']].astype(str)
   frame=part.merge(truth.rename(columns={'speed_kmh':'true_v','flow_vph':'true_q'}),on=['timestamp','station_id','link_id'],validate='one_to_one',sort=False)
   frame['lanes']=frame.link_id.map(fd);frame['length_km']=frame.link_id.map(lengths)
   assert frame[['lanes','length_km','true_v','true_q']].notna().all().all()
   vv=dict(baseline=part[['speed_kmh','flow_vph']].to_numpy(),candidate=corrected[['speed_kmh','flow_vph']].to_numpy());items.append(frame)
   for name in values:values[name].append(vv[name]);daily.append(dict(panel=panel,date=date,variant=name,**metric([frame],[vv[name]])))
  rows.extend(dict(panel=panel,variant=name,**metric(items,v)) for name,v in values.items())
  pd.DataFrame(rows).to_csv(args.output/'by_panel.csv',index=False)
  pd.DataFrame(daily).to_csv(args.output/'by_date.csv',index=False)
  print(panel,rows[-2:],flush=True)
 if not rows:raise ValueError('No selected evaluation rows')
 columns=['S_state','speed_RMSE','flow_per_lane_RMSE','station_target_deltaN_RMSE']
 pd.DataFrame(rows).groupby('variant')[columns].mean().to_csv(args.output/'summary.csv')
 (args.output/'scope.json').write_text(json.dumps(dict(start=args.start,end=args.end,official_score=None,scope='Exposed public train development; accumulation-change diagnostic is not official LWR. Provide complete template target rows for a full comparison.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2),encoding='utf8')
if __name__=='__main__':main()
