# 中文说明：核对五分钟形状与站点元数据，不把速度推算成实测流量。
from pathlib import Path
import json,zipfile,hashlib
import pandas as pd,numpy as np
W=Path('work/fivemin_v4');O=Path('outputs/fivemin_v4');audit=[];daily=[]
for road in ['I-10','I-405']:
 j=json.loads((W/(road+'_link_performance.json')).read_text());info=pd.read_csv(W/(road+'_sensor_information.csv')).set_index('sensor_id')
 for sid,x in j.items():
  assert x['dt']=='5min' and len(x['f'])==x['n']==31*288
  t=pd.date_range(x['t0'],periods=x['n'],freq='5min');f=np.array(x['f'],float);s=np.array(x['s'],float);assert np.isfinite(f).all() and (f>=0).all()
  p=f.reshape(31,288)[:,15*12:19*12];pm=p.mean(axis=1)
  panel=f"D7_I{x['meta']['fwy']}_{x['meta']['dir']}"
  audit.append({'source':'cbi_plus','panel':panel,'station_id':sid,'rows':len(f),'cadence_minutes':5,'first':str(t[0]),'last':str(t[-1]),'flow_scope':'per_lane_vph','quality_observed_flag':False,'flow_finite':bool(np.isfinite(f).all()),'flow_zero_fraction':float((f==0).mean()),'speed_median_raw':float(np.median(s))})
  for k,v in enumerate(pm):daily.append({'panel':panel,'station_id':sid,'date':str(t[k*288].date()),'day_index':k,'PM_mean_flow_per_lane':v,'PM_std_flow_per_lane':float(p[k].std())})
pd.DataFrame(audit).to_csv(O/'flow_source_audit.csv',index=False);pd.DataFrame(daily).to_csv(O/'real_daily_PM_flows.csv',index=False)
info=pd.read_csv(W/'PeMSD7_L_Station_Info_1026.csv');info=info[info.Fwy.isin([5,10,210,405])]
with zipfile.ZipFile(W/'PeMSD7_Full.zip') as z:
 with z.open('PeMSD7_V_1026.csv') as stream:v=pd.read_csv(stream,header=None).to_numpy()
assert v.shape[1]==1026 and len(v)%288==0
rows=[]
for (r,d),sub in info.groupby(['Fwy','Dir']):
 vals=v[:,sub['Unnamed: 0'].to_numpy(int)]
 rows.append({'source':'STGCN_PeMSD7_L','district':7,'route':int(r),'direction':d,'stations':len(sub),'rows_per_station':len(v),'weekday_blocks':len(v)//288,'channel':'speed_only','cadence_minutes':5,'timestamps':'weekday blocks in May-Jun 2012; explicit per-row timestamps absent','zero_fraction':float((vals==0).mean()),'median_raw':float(np.median(vals))})
info.to_csv(O/'speed_source_station_metadata.csv',index=False);pd.DataFrame(rows).to_csv(O/'speed_source_coverage.csv',index=False)
print(pd.DataFrame(audit).groupby('panel').agg(stations=('station_id','size'),rows=('rows','sum')));print(pd.DataFrame(rows).to_string(index=False));print('speed matrix',v.shape)
