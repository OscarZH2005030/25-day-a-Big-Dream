"""Real regional OD and household trip distance aggregates; no fake zone matching."""
import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
from pathlib import Path
import pandas as pd,numpy as np,json,zipfile
O=Path(__file__).resolve().parent;W=O.parents[1]/'work/external_od_v17'
BINS=np.array([0,.5,1,2,3,5,8,12,20,30,50,80,120,200,400,1000,3000.])
def histrec(name,d,w):
 h,_=np.histogram(d,bins=BINS,weights=w);n,_=np.histogram(d,bins=BINS)
 return [dict(source=name,lo_km=float(BINS[i]),hi_km=float(BINS[i+1]),weight=float(v),records=int(n[i])) for i,v in enumerate(h)]
def main():
 stats={};rows=[];xw=pd.read_csv(W/'ca_xwalk.csv.gz',usecols=['tabblk2020','cty','blklatdd','blklondd']);xw=xw[xw.cty.isin([6037,6059])].set_index('tabblk2020');stats['real_LA_Orange_census_blocks']=len(xw);hists=np.zeros(len(BINS)-1);nh=np.zeros(len(BINS)-1,dtype=int);county={};scanned=kept=jobs=missing=0
 for x in pd.read_csv(W/'ca_od_main_JT01_2023.csv.gz',usecols=['w_geocode','h_geocode','S000'],chunksize=300000):
  scanned+=len(x);x=x[(x.w_geocode//10**10).isin([6037,6059])&(x.h_geocode//10**10).isin([6037,6059])].copy();kept+=len(x)
  if x.empty:continue
  la1=x.w_geocode.map(xw.blklatdd).to_numpy();lo1=x.w_geocode.map(xw.blklondd).to_numpy();la2=x.h_geocode.map(xw.blklatdd).to_numpy();lo2=x.h_geocode.map(xw.blklondd).to_numpy();good=np.isfinite(la1+lo1+la2+lo2);missing+=int((~good).sum());lat1=np.radians(la1[good]);lat2=np.radians(la2[good]);dl=np.radians(lo2[good]-lo1[good]);dist=6371*2*np.arcsin(np.sqrt(np.clip(np.sin((lat2-lat1)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dl/2)**2,0,1)));w=x.S000.to_numpy()[good];jobs+=int(w.sum());hists+=np.histogram(dist,bins=BINS,weights=w)[0];nh+=np.histogram(dist,bins=BINS)[0]
  xx=x[good].copy();xx['home_county']=xx.h_geocode//10**10;xx['work_county']=xx.w_geocode//10**10
  for key,v in xx.groupby(['home_county','work_county']).S000.sum().items():county[key]=county.get(key,0)+int(v)
 rows.extend([dict(source='LODES_LA_OC_primary_jobs_straight_line',lo_km=float(BINS[i]),hi_km=float(BINS[i+1]),weight=float(v),records=int(nh[i])) for i,v in enumerate(hists)])
 pd.DataFrame([dict(home_county=h,work_county=w,primary_jobs=v) for (h,w),v in county.items()]).to_csv(O/'real_county_home_work_od.csv',index=False);stats.update(LODES_CA_rows_scanned=scanned,LODES_LA_OC_OD_pairs=kept,LODES_LA_OC_primary_jobs=jobs,LODES_missing_coordinate_pairs=missing)
 z=zipfile.ZipFile(W/'nhts2017_Csv.zip');cols=['HOUSEID','HHSTATE','HH_CBSA','TRPMILES','TRPTRANS','DRVR_FLG','STRTTIME','TRAVDAY','WTTRDFIN'];parts=[];scanned=0
 for x in pd.read_csv(z.open('trippub.csv'),usecols=cols,dtype={'HHSTATE':str,'HH_CBSA':str},chunksize=200000):
  scanned+=len(x);x=x[(x.HHSTATE=='CA')&(x.HH_CBSA=='31080')&(x.TRPMILES>0)&(x.WTTRDFIN>0)&(x.DRVR_FLG==1)&x.TRPTRANS.isin([3,4,5,6,8,9,18])&x.TRAVDAY.between(2,6)].copy();parts.append(x)
 x=pd.concat(parts,ignore_index=True);stats['NHTS_national_trip_records_scanned']=scanned;stats['NHTS_LA_OC_weekday_vehicle_trips']=len(x);stats['NHTS_LA_OC_households']=x.HOUSEID.nunique()
 for name,s in [('NHTS_LA_OC_weekday_vehicle',x),('NHTS_LA_OC_PM_vehicle',x[x.STRTTIME.between(1500,1859)])]:
  stats[name+'_records']=len(s);stats[name+'_households']=s.HOUSEID.nunique();stats[name+'_weighted_mean_km']=float(np.average(s.TRPMILES*1.609344,weights=s.WTTRDFIN));rows.extend(histrec(name,s.TRPMILES.to_numpy()*1.609344,s.WTTRDFIN.to_numpy()))
 pd.DataFrame(rows).to_csv(O/'real_distance_histograms.csv',index=False)
 stats.update({'LODES_is':'annual employer/worker residence-workplace linkage; not observed vehicle trips, routes or PM departures','NHTS_is':'2017 sampled household driver trips; survey-weighted lengths, no target-zone geographic crosswalk','competition_geographic_zone_matches':0,'competition_counts_are':'synthetic released counts, not historical measured traffic rows','transfer_assumption':'trip lengths or commuter straight-line distances can only provide a weak regional length prior; corridor segment length is not full trip length'})
 (O/'real_data_summary.json').write_text(json.dumps(stats,indent=2),encoding='utf-8');print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
