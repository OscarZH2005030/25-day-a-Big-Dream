"""Real-data length priors on verified complete paths; same fixed 480 cases."""
import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import sys,json,time,hashlib
from pathlib import Path
O=Path(__file__).resolve().parent
sys.path[:0]=[str(O.parent/'joint_v13'),str(O.parent/'joint_v15')]
from run_invariant import REL,load,load_operator,reference,cases,metric,folds,METRICS,TARGETS,np,pd
from equality_projection import EqualProjector
from audit_geography import lengths
PRIOR_SOURCES={'nhts_pm_survival':('NHTS_LA_OC_PM_vehicle','survival',1.),'nhts_pm_density':('NHTS_LA_OC_PM_vehicle','density',1.),'nhts_weekday_survival':('NHTS_LA_OC_weekday_vehicle','survival',1.),'lodes_survival_circ13':('LODES_LA_OC_primary_jobs_straight_line','survival',1.3)}
SPECS={'v11':{'source':None,'weight':0.}}
for src in PRIOR_SOURCES:
 for w in [.05,.15,.3]:SPECS[f'{src}_mix{w:g}']={'source':src,'weight':w}

def shape(source,km):
 label,kind,circ=PRIOR_SOURCES[source];h=pd.read_csv(O/'real_distance_histograms.csv');h=h[h.source==label].sort_values('lo_km');edges=np.r_[h.lo_km.to_numpy(),h.hi_km.iloc[-1]]*circ;p=h.weight.to_numpy(float);p=p/p.sum()
 if kind=='survival':return np.maximum(np.interp(km,edges,np.r_[1.,1.-np.cumsum(p)]),1e-8)
 density=p/np.diff(edges);return np.maximum(np.interp(km,.5*(edges[:-1]+edges[1:]),density,left=density[0],right=density[-1]),1e-10)

def endpoints(A,c,b,paths,panel,E,sources=None):
 _,ref=reference(A,c,b,panel);known,km=lengths(panel,paths);out={};errors=[]
 if known.sum()<100:return ref,{src:ref for src in PRIOR_SOURCES},0.
 oi=pd.factorize(paths.origin_zone)[0]
 for src in (sources if sources is not None else PRIOR_SOURCES):
  q=ref.copy();s=shape(src,km[known]);idx=np.flatnonzero(known)
  for origin in np.unique(oi[idx]):
   use=oi[idx]==origin;ii=idx[use]
   if len(ii)<2:continue
   q[ii]=s[use]/s[use].sum()*ref[ii].sum()
  x,err=E.project(q,ref);out[src]=x;errors.append(err)
 return ref,out,max(errors,default=0.)

def mix(name,ref,ends):
 spec=SPECS[name]
 return ref if spec['source'] is None else (1-spec['weight'])*ref+spec['weight']*ends[spec['source']]

def main():
 start=time.time();old=pd.read_csv(O.parent/'joint_v11/comparison_trials.csv');old=old[old.method=='joint_dev_v11'];oldcv=pd.read_csv(O.parent/'joint_v12/real_count_cv.csv');oldcv=oldcv[oldcv.candidate=='v11'];panels=sorted(p.name for p in (REL/'task4').iterdir());calrows=[];counts=[];comparison=[];checks=[];policy={};cert=[]
 for pn,p in enumerate(panels):
  ids,ls,D,paths=load_operator(REL/'corridors'/p/'network');A,c,base,_,_=load(p,'train',ids,ls,D);dest=paths.destination_zone.to_numpy();E=EqualProjector(A,dest);data=cases(pn,base,A,c,paths)
  for phase,scenario,f,b,cc in data:
   if phase!='calibration':continue
   ref,ends,err=endpoints(A,cc,b,paths,p,E);cert.append(dict(panel=p,phase=phase,scenario=scenario,max_constraint_error=err))
   for name in SPECS:calrows.append(dict(panel=p,phase=phase,scenario=scenario,candidate=name,**metric(mix(name,ref,ends),f,b,A,cc,dest)))
  t=pd.DataFrame(calrows);t=t[t.panel==p];cal=t.groupby('candidate')[METRICS].mean();td=t[t.scenario.isin(TARGETS)].groupby(['scenario','candidate'])[METRICS].mean();eligible=['v11']
  for n in SPECS:
   d=cal.loc[n]-cal.loc['v11'];dd=td.xs(n,level='candidate')-td.xs('v11',level='candidate');ok=bool((d>=-1e-10).all() and (dd>=-1e-10).all().all());checks.append(dict(panel=p,candidate=n,synthetic_eligible=ok,target_worst_Sdev_delta=float(dd.S_dev.min()),**{'delta_'+k:float(d[k]) for k in METRICS}))
   if ok and n!='v11' and abs(d).max()>1e-10:eligible.append(n)
  for seed in [20261201,20261202,20261203,20261204]:
   fd=folds(A,seed)
   for k in range(3):
    if len(eligible)>1:
     M=A[fd!=k];cc=c[fd!=k];EE=EqualProjector(M,dest);ref,ends,err=endpoints(M,cc,base,paths,p,EE)
     for n in eligible:
      pred=mix(n,ref,ends);counts.append(dict(panel=p,seed=seed,fold=k,candidate=n,NMAE=float(abs(A[fd==k]@pred-c[fd==k]).sum()/c[fd==k].sum())))
    else:
     v=oldcv[(oldcv.panel==p)&(oldcv.seed==seed)&(oldcv.fold==k)].iloc[0];counts.append(dict(panel=p,seed=seed,fold=k,candidate='v11',NMAE=float(v.NMAE)))
  ct=pd.DataFrame(counts);ct=ct[(ct.panel==p)&(ct.seed<=20261202)].groupby(['seed','candidate']).NMAE.mean().unstack();passed=[n for n in eligible if ct[n].mean()<=ct.v11.mean()*1.000001 and (ct[n]/ct.v11).max()<=1.01];gain=cal-cal.loc['v11'];score=.5*(gain.S_od+gain.S_dev);chosen=str(score.loc[passed].idxmax());policy[p]=chosen
  for phase,scenario,f,b,cc in data:
   rr=old[(old.panel==p)&(old.phase==phase)&(old.scenario==scenario)].iloc[0];_,ref=reference(A,cc,b,p);baseline=metric(ref,f,b,A,cc,dest);assert max(abs(baseline[k]-rr[k]) for k in METRICS)<1e-10
   if chosen=='v11':m=baseline
   elif phase=='calibration':m=t[(t.scenario==scenario)&(t.candidate==chosen)].iloc[0][METRICS].to_dict()
   else:
    ref,ends,err=endpoints(A,cc,b,paths,p,E);cert.append(dict(panel=p,phase=phase,scenario=scenario,max_constraint_error=err));m=metric(mix(chosen,ref,ends),f,b,A,cc,dest)
   comparison.extend([dict(panel=p,phase=phase,scenario=scenario,method='v11',**baseline),dict(panel=p,phase=phase,scenario=scenario,method='external_od_v17',**m)])
  pd.DataFrame(calrows).to_csv(O/'calibration_candidates.csv',index=False);pd.DataFrame(counts).to_csv(O/'released_synthetic_count_cv.csv',index=False);pd.DataFrame(comparison).to_csv(O/'comparison_trials.csv',index=False);print('completed',p,chosen,flush=True)
 pd.DataFrame(checks).to_csv(O/'selection_checks.csv',index=False);pd.DataFrame(cert).to_csv(O/'invariance_certificates.csv',index=False);(O/'policy.json').write_text(json.dumps({'policy':policy,'specs':SPECS,'external_prior_sources':PRIOR_SOURCES,'minimum_complete_length_paths_per_panel':100,'selection':'maximize joint calibration Sod+Sdev with all-four/total guards overall and target families, released SYNTHETIC count CV mean no worse/worst <=1.01','selection_count_seeds':[20261201,20261202],'targets':TARGETS,'benchmark_previously_seen':True,'actual_geographic_OD_zone_matches':0},indent=2),encoding='utf-8')
 df=pd.DataFrame(comparison);df.groupby(['phase','method'])[METRICS].mean().to_csv(O/'four_components_overall.csv');df.groupby(['phase','scenario','method'])[METRICS].mean().to_csv(O/'four_components_by_scenario.csv');parts=[]
 for p,n in policy.items():
  for method,name in [('v11','v11'),('external_od_v17',n)]:
   q=pd.DataFrame(counts);q=q[(q.panel==p)&(q.candidate==name)].copy();q['method']=method;parts.append(q)
 pd.concat(parts).groupby(['seed','method']).NMAE.mean().to_csv(O/'released_synthetic_count_overall.csv');(O/'manifest.json').write_text(json.dumps({'seconds':time.time()-start,'same_fixed_cases':480,'candidate_count':len(SPECS),'real_external_histogram_sha256':hashlib.sha256((O/'real_distance_histograms.csv').read_bytes()).hexdigest(),'max_constraint_error':max(r['max_constraint_error'] for r in cert),'CPU_threads':4,'hidden_truth_available':False,'baseline_preserved':True},indent=2),encoding='utf-8');print('POLICY',policy,flush=True);print(df.groupby(['phase','method'])[METRICS].mean().to_string(),flush=True)
if __name__=='__main__':main()
