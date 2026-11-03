"""Add genuinely new five-minute training origins without changing the old feature library."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import json,time,hashlib,copy
from pathlib import Path
import numpy as np
import pandas as pd
import train_ongoing_speed as speed
import run_analog_forecast as analog
from ongoing_core import features,observed_state
from onset_context_core import PastContext
from speed_distribution_core import spatial_features,PANELS
B=speed.old.BASE;O=B/'outputs/dense_ongoing_final_data_v30_20261102';START=time.time()
def save(n,x):speed.old.write(O/n,x)
def log(**x):print(json.dumps(dict(seconds=round(time.time()-START,1),**x)),flush=True)

def main():
    O.mkdir(exist_ok=False);save('decision_before_preparation.json',dict(basis='October-selected dense100 primary and dense50 secondary both confirmed on expanded November; final-fit preparation only, no new model or alpha selection',fit='October-November new part; June-September frozen v28 reused via manifest',policy='Same actual origin shifts -10/-5/+5/+10 minutes; bounds12..281; causal ongoing condition; actual masked labels, no shifted targets; exact original X/y/m parity',prior='Load frozen September30 PastContext, only original roster targets update at day end; augmented targets never update priors',seed=20261204,correlation='Adjacent origins are correlated, not new independent dates',no_January_February=True,production_unchanged=True,Kaggle_upload=False));save('data_parts.json',dict(June_September='outputs/dense_ongoing_v28_20261102',October_November=str(O.relative_to(B)),primary='dense_100',secondary='dense_50',selection='Fixed October ranking; November confirmation only'));get,bases,rf,_=analog.setup();rf.WORK=O/'ramp_cache';rf.WORK.mkdir();entries=speed.old.prior.REF.entries('june-november-expanded-fit.json');bydate={(e['panel'],e['origin'][:10]):e for e in entries};dayroot=speed.old.prior.REF.RUN/'rebuild/days';from task2q.data import make_sample,day
    expected=speed.old.read(B/'outputs/ongoing_speed_distribution_20261030/feature_names.json');files=[Path(__file__),B/'outputs/ongoing_core.py',B/'outputs/onset_context_core.py',B/'outputs/speed_distribution_core.py',B/'outputs/dense_ongoing_v29_20261102/verification.json',B/'outputs/dense_ongoing_v29_20261102/November_confirmation.json',B/'outputs/dense_ongoing_v28_20261102/verification.json']
    for panel in PANELS:files.extend([B/'outputs/ongoing_20261030/development_reference'/f'{panel}.npz',speed.old.RUN/'fit'/panel/'complete.json',speed.old.RUN/'fit'/panel/'X.npy',speed.RUN/'fit'/panel/'extra.npy'])
    save('predeclared_hashes.json',{str(f.relative_to(B)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files});allstats={};checks=[]
    for panel in PANELS:
        source=speed.old.RUN/'fit'/panel;metadata=speed.old.read(source/'complete.json');oldbase=np.load(source/'X.npy',mmap_mode='r');oldextra=np.load(speed.RUN/'fit'/panel/'extra.npy',mmap_mode='r');oldy=np.load(source/'y.npy',mmap_mode='r');oldm=np.load(source/'m.npy',mmap_mode='r');oldw=np.load(source/'w.npy',mmap_mode='r');originals={};old_by_day={}
        for win in metadata['windows']:
            if '2030-10'<=win['origin'][:7]<='2030-11':old_by_day.setdefault(win['origin'][:10],[]).append(win)
        refentry=next(bydate[(panel,date)] for date in sorted(old_by_day));netfile=dayroot/f"{refentry['network_key']}.npz"
        with np.load(netfile) as z:net={k:z[k] for k in z.files}
        state=PastContext.load(B/'outputs/ongoing_20261030/development_reference'/f'{panel}.npz');assert state.last_date=='2030-09-30';rng=np.random.default_rng(20261204);parts={k:[] for k in ['X','y','m','w','month']};wins=[];cursor=0;matched=0;newcount=0;rejected=0;checked=False
        for date in pd.date_range('2030-10-01','2030-11-30').strftime('%Y-%m-%d'):
            entry=bydate.get((panel,date))
            if entry is not None:
                with np.load(dayroot/f"{entry['day_key']}.npz") as z:values={k:z[k] for k in z.files}
            else:
                values=day(speed.old.prior.REF.DATA,panel,date,cache_dir=O/'public_day_cache')
                log(stage='PUBLIC_DAY_FALLBACK',panel=panel,date=date)
            oldwins=old_by_day.get(date,[]);slotmap={pd.Timestamp(w['origin']).hour*12+pd.Timestamp(w['origin']).minute//5:w for w in oldwins};slots=sorted({slot+delta for slot in slotmap for delta in [-2,-1,0,1,2] if 12<=slot+delta<=281});targets=[]
            for slot in slots:
                s=make_sample(values,net,slot,panel)
                if s['condition']!=1:
                    assert slot not in slotmap;rejected+=1;continue
                ramp=rf.ramp_history(s);counts=rf.topology(panel)['counts'];base,n,_=features(s,ramp,counts,bases[panel]['position'],state);extra,en=spatial_features(s,bases[panel]['position']);assert n==expected['base'] and en==expected['extra'];x=np.column_stack([base,extra]).astype('float32');y=s['y'].T;m=s['m'].T;original=slot in slotmap
                if original:
                    win=slotmap[slot];ids=np.asarray(win['selected_links']);a,b=win['start'],win['end'];np.testing.assert_array_equal(x[ids],np.column_stack([oldbase[a:b],oldextra[a:b]]));np.testing.assert_array_equal(y[ids],oldy[a:b]);np.testing.assert_array_equal(m[ids],oldm[a:b]);weight=np.asarray(oldw[a:b]);targets.append((slot,y,m));matched+=1
                else:
                    known=m.any(1);_,_,_,_,q=observed_state(s);forced=known&((y.astype(bool)&m).any(1)|q[-6:].any(0));rest=np.flatnonzero(known&~forced);k=min(64,len(rest));pick=rng.choice(rest,k,replace=False) if k else rest;ids=np.r_[np.flatnonzero(forced),pick];pi=np.r_[np.ones(forced.sum()),np.full(k,k/max(len(rest),1))];weight=(1/(max(int(m.sum()),1)*pi)).astype('float32');newcount+=1
                if not checked:
                    other=copy.deepcopy(s);other['y']=1-s['y'];other['m']=~s['m'];other['speed_y']=np.full_like(s['speed_y'],999);other['has_future_queue']=not s['has_future_queue'];bb,nn,_=features(other,ramp,counts,bases[panel]['position'],state);ee,eenn=spatial_features(other,bases[panel]['position']);np.testing.assert_array_equal(x,np.column_stack([bb,ee]).astype('float32'));checked=True
                nrows=len(ids);density=float((y.astype(bool)&m).sum()/max(m.sum(),1));parts['X'].append(x[ids]);parts['y'].append(y[ids]);parts['m'].append(m[ids]);parts['w'].append(weight);parts['month'].append(np.full(nrows,int(date[:7].replace('-','')),dtype='int32'));wins.append(dict(panel=panel,origin=s['origin'],positive=bool(s['has_future_queue']),start=cursor,end=cursor+nrows,original_window=original,density=density));cursor+=nrows
            # Only old target windows update the historical priors, once all this day's inputs exist.
            assert len(targets)==len(oldwins);state.update_day(date,values,rf.day(panel,'train',date),net,rf.topology(panel)['counts'],targets)
            if date.endswith('-15') or date.endswith('-30'):log(stage='DENSE_INPUTS',panel=panel,date=date,original_windows=matched,new_windows=newcount,rows=cursor)
        assert matched==sum(len(v) for v in old_by_day.values()) and checked;folder=O/'fit'/panel;folder.mkdir(parents=True)
        for k,v in parts.items():np.save(folder/f'{k}.npy',np.concatenate(v))
        save(f'fit/{panel}/complete.json',dict(rows=cursor,windows=wins,original_windows_exactly_reproduced=matched,new_origins=newcount,rejected_nonongoing=rejected,input_columns=274));allstats[panel]=dict(original_windows=matched,new_origins=newcount,rows=cursor,rejected_nonongoing=rejected);checks.append(dict(panel=panel,all_original_X_y_m_exact=True,new_input_future_mutation_invariant=True,old_priors_not_updated_from_augmented_targets=True));log(stage='ROAD_COMPLETE',panel=panel,**allstats[panel]);del parts
    save('statistics.json',allstats);save('verification.json',dict(passed=len(checks)==8,checks=checks,October_November_only=True,June_September_readonly_reuse=True,new_labels_actual_observations=True,no_January_February=True,production_unchanged=True,Kaggle_upload=False));log(stage='COMPLETE',statistics=allstats)
if __name__=='__main__':main()
