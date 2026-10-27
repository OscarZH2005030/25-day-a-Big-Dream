# 中文说明：只提取公开真实记录；缺测年份不补零，不把ADT转成PM绝对流量。
import pandas as pd,numpy as np,json
from pathlib import Path
W=Path('work/external_v3');O=Path('outputs/external_v3');parts=[]
for dist in [7,12]:
 p=W/f'd{dist}_ramps_2024.xlsx';d=pd.read_excel(p,sheet_name=1,dtype=str);d['route']=pd.to_numeric(d.RTE,errors='coerce');d=d[d.route.isin([5,10,210,405])].copy();d['district']=dist
 d['direction']=d.LOCATION_DESC.str.extract(r'\b([NSEW])B\b')[0];d['ramp_type']=d.LOCATION_DESC.str.extract(r'\b(ON|OFF)\b')[0]
 for y in range(2015,2025):d[f'YR_{y}']=pd.to_numeric(d[f'YR_{y}'],errors='coerce')
 d['latest_volume']=d[[f'YR_{y}' for y in range(2015,2025)]].ffill(axis=1).iloc[:,-1]
 d['latest_year']=np.nan
 for y in range(2015,2025):d.loc[d[f'YR_{y}'].notna(),'latest_year']=y
 parts.append(d)
r=pd.concat(parts,ignore_index=True);r.to_csv(O/'real_ramp_records.csv',index=False)
p=pd.read_excel(W/'peak_hours_2024.xlsx',sheet_name=1,dtype=str);p['district']=pd.to_numeric(p.DI,errors='coerce');p['route']=pd.to_numeric(p.RTE,errors='coerce');p=p[p.district.isin([7,12])&p.route.isin([5,10,210,405])];p.to_csv(O/'real_peak_records.csv',index=False)
j=json.loads((W/'aadt_selected_2024.json').read_text());a=pd.DataFrame([{**f['attributes'],'longitude':f['geometry']['x'],'latitude':f['geometry']['y']} for f in j['features']]);a.to_csv(O/'real_aadt_records.csv',index=False)
pars=[];pairs=[]
for (dist,route,direction),d in r.dropna(subset=['direction']).groupby(['district','route','direction']):
 if (dist,route) not in [(7,10),(7,210),(7,405),(12,5),(12,405)]:continue
 mus=[];deltas=[]
 for _,row in d.iterrows():
  vals=[(y,row[f'YR_{y}']) for y in range(2015,2025) if pd.notna(row[f'YR_{y}']) and row[f'YR_{y}']>0]
  for (y0,v0),(y1,v1) in zip(vals,vals[1:]):
   if y1-y0==1:
    mus.append((v0+v1)/2);deltas.append(abs(v1-v0));pairs.append({'district':dist,'route':route,'direction':direction,'year0':y0,'year1':y1,'v0':v0,'v1':v1})
 m=np.array(mus);e=np.array(deltas);valid=(e>0)&(m>0)
 # 稳健分箱拟合方差的流量指数，描述跨年绝对波动随流量的变化。
 bins=pd.qcut(m[valid],min(5,len(m[valid])//5),duplicates='drop') if valid.sum()>=15 else None
 gamma=1.
 if bins is not None:
  g=pd.DataFrame({'m':m[valid],'e':e[valid],'bin':bins}).groupby('bin',observed=True)[['m','e']].median();gamma=float(np.clip(2*np.polyfit(np.log(g.m),np.log(g.e),1)[0],0,2))
 off=d[(d.ramp_type=='OFF')&(d.latest_volume>0)].latest_volume.to_numpy(float)
 pars.append({'panel':f'D{dist}_I{route}_{direction}','n_ramp_records':len(d),'n_off_ramps':len(off),'n_year_pairs':len(m),'variance_exponent':gamma,'off_cv':float(np.std(off)/np.mean(off)) if len(off) else None,'observations_2024':int(d.YR_2024.notna().sum()),'min_latest_year':float(d.latest_year.min()),'max_latest_year':float(d.latest_year.max())})
pd.DataFrame(pairs).to_csv(O/'real_ramp_year_pairs.csv',index=False);pd.DataFrame(pars).to_csv(O/'external_parameters.csv',index=False)
print(pd.DataFrame(pars).to_string(index=False));print('rows ramps/peak/aadt',len(r),len(p),len(a))
