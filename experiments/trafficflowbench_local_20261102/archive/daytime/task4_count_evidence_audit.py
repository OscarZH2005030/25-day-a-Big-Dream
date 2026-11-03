"""Audit aggregation compatibility before importing public state/ramp constraints."""
import common as d
import numpy as np,pandas as pd,pyarrow.parquet as pq,json,msvcrt,traceback
from datetime import datetime,timezone
O=d.OUT/'task4_count_evidence_audit'

def main():
 d.save(O/'protocol.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),hypothesis='Public PM mainline/ramp averages may supply additional demand constraints only if compatible with released Task4 synthetic counts.',panels=list(d.night.predictor.load_routes()),splits=['train','validation','private'],hours_UTC=[15,19],aggregation='Official public fallback aggregates every finite flow_vph over15<=UTC hour<19 by link. Also report quality-eligible-only variant. Train uses released complete states; evaluation uses only masked observations.',selection='No model fitting or OD/path prediction changes; all panels and splits retained. Fitted global scales only diagnose compatibility, not evidence of hidden accuracy.',source_sha256=d.sha(__file__),scope='Source compatibility audit, not Task4 validation or hidden score. No Kaggle uploads or credential access; frozen Task4 baseline retained.'))
 rows=[];joined=[];hashes=[];daily=[]
 for panel in d.night.predictor.load_routes():
  for split in ['train','validation','private']:
   d.check_deadline();d.state(status='running',stage='task4_count_evidence_audit',panel=panel,split=split,next='Audit actual PM observation/ramp count compatibility before imposing any extra OD constraint; no model changes')
   cp=d.DATA/'task4'/panel/split/'synthetic_link_counts.csv';counts=pd.read_csv(cp,dtype={'link_id':str});hashes.append(dict(path=str(cp),sha256=d.sha(cp)));aggregates=[]
   for kind,dirname in [('mainline','mainline_states' if split=='train' else 'mainline_states_masked'),('ramp','ramp_states')]:
    files=sorted((d.DATA/'corridors'/panel/split/dirname).rglob('*.parquet'));assert files
    for path in files:
     d.check_deadline();columns=pq.read_schema(path).names;idcol='ramp_link_id' if kind=='ramp' and 'ramp_link_id' in columns else 'link_id';cols=[c for c in [idcol,'timestamp','flow_vph','is_score_eligible','pct_observed'] if c in columns];f=pd.read_parquet(path,columns=cols);ts=pd.to_datetime(f.timestamp,utc=True);f=f[(ts.dt.hour>=15)&(ts.dt.hour<19)].copy();f['link_id']=f[idcol].astype(str);f['all_flow']=pd.to_numeric(f.flow_vph,errors='coerce');eligible=(f.is_score_eligible.astype(bool) if 'is_score_eligible' in f else pd.Series(True,index=f.index))&(f.pct_observed>=75 if 'pct_observed' in f else True);f['eligible_flow']=f.all_flow.where(eligible);gg=f.groupby('link_id').agg(all_sum=('all_flow','sum'),all_n=('all_flow','count'),eligible_sum=('eligible_flow','sum'),eligible_n=('eligible_flow','count')).reset_index();gg['kind']=kind;aggregates.append(gg);date=path.stem[-10:].replace('_','-')
     for r in gg.to_dict('records'):daily.append(dict(panel=panel,split=split,date=date,**r))
     hashes.append(dict(path=str(path),sha256=d.sha(path)))
   agg=pd.concat(aggregates,ignore_index=True).groupby(['kind','link_id'],as_index=False)[['all_sum','all_n','eligible_sum','eligible_n']].sum();agg['all_mean']=agg.all_sum/agg.all_n.replace(0,np.nan);agg['eligible_mean']=agg.eligible_sum/agg.eligible_n.replace(0,np.nan);audit=counts.merge(agg,on='link_id',how='left',validate='one_to_one');audit['panel']=panel;audit['split']=split;joined.extend(audit.to_dict('records'))
   for kind in ['all_matched','mainline','ramp']:
    sel=audit if kind=='all_matched' else audit[audit.kind==kind]
    for variant in ['all_mean','eligible_mean']:
     good=sel[sel[variant].notna()&(sel[variant]>0)&sel['count'].notna()];c=good['count'].to_numpy();v=good[variant].to_numpy();scale=float(np.median(c/v)) if len(v) else None;err=float(np.abs(v-c).sum()/max(c.sum(),1e-9)) if len(v) else None;scaled=float(np.abs(v*scale-c).sum()/max(c.sum(),1e-9)) if len(v) else None;corr=float(np.corrcoef(c,v)[0,1]) if len(v)>2 and np.std(c)>0 and np.std(v)>0 else None;rows.append(dict(panel=panel,split=split,kind=kind,variant=variant,matched_links=len(v),published_links=len(counts),unmatched_published_links=int(audit.kind.isna().sum()),direct_count_NMAE=err,median_diagnostic_scale=scale,scaled_count_NMAE=scaled,Pearson=corr))
   pd.DataFrame(rows).to_csv(O/'compatibility.csv',index=False);pd.DataFrame(joined).to_csv(O/'per_link.csv',index=False);pd.DataFrame(daily).to_csv(O/'daily_PM_aggregates.csv',index=False);d.save(O/'input_sha256.json',hashes);print('COUNT EVIDENCE',panel,split,rows[-6:],flush=True)
 d.state(status='stage_complete',stage='task4_count_evidence_audit_complete',next='Assess all-link/mainline/ramp aggregation mismatch, require verified compatibility before additional demand constraints; keep Task4 baseline, no hidden-score claim.')

if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True);lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:main()
 except Exception as e:d.state(status='failed_or_cutoff',stage='task4_count_evidence_audit',error=repr(e),next='Repair public aggregation audit preserving UTC hours, all panels/splits and original data');traceback.print_exc();raise
 finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
