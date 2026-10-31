"""Independent whole-corridor analog forecasting; no R127 predictions as inputs.

June-November positive-onset library; fixed16 distinct-day analogs. January
selects binary threshold only. Frozen February populations are evaluation only.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='4'
import argparse,json,hashlib,sys,time,importlib.util
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
import run_anchored_residual as ar
BASE=ar.BASE;REF=ar.ref;OUT=BASE/'outputs/analog_forecast';RUN=BASE/'work/analog_forecast'
K=16
TIME_INDEX=[0,3,6,9,11,12]
def read(p):return ar.read(p)
def write(p,v):return ar.write(p,v)
def log(**v):print(json.dumps(dict(seconds=round(time.time()-START,1),**v)),flush=True)

def project(values,valid,phi):
    valid=np.asarray(valid,bool)&np.isfinite(values)
    if not valid.any():return np.zeros(8)
    a=phi[valid]
    return np.linalg.solve(a.T@a+1e-3*np.eye(8),a.T@values[valid])

def neighbor_mean(values,valid,edges,reverse=False):
    """up->down edges: reverse=True pools downstream states at upstream nodes."""
    source,target=edges[::-1] if reverse else edges
    total=np.zeros(len(values));count=np.zeros(len(values))
    good=valid[source]&np.isfinite(values[source])
    np.add.at(total,target[good],values[source[good]]);np.add.at(count,target[good],1)
    return np.divide(total,count,out=np.zeros_like(total),where=count>0),count>0

def embedding(s,basis,ramps,ramp_net):
    x=s['x'];net=s['network'];phi=basis['position'];support=basis['supported'];parts={}
    for t in TIME_INDEX:
        for ch,missing,label in [(0,4,'speed'),(1,5,'flow'),(2,6,'occupancy')]:
            good=(x[t,:,missing]<.5)&(x[t,:,3]>=.75)&(x[t,:,7]>.5)&support
            coef=project(x[t,:,ch],good,phi)
            for k,c in enumerate(coef):parts[f'field_{label}_t{t-12}_k{k}']=c
            parts[f'coverage_{label}_t{t-12}']=good.mean()
    for t in [9,12]:
        ownv=x[t,:,0];ownq=x[t,:,1]
        vg=(x[t,:,4]<.5)&(x[t,:,3]>=.75)&(x[t,:,7]>.5)
        qg=(x[t,:,5]<.5)&(x[t,:,3]>=.75)&(x[t,:,7]>.5)
        down,down_good=neighbor_mean(ownv,vg,net['edges'],True)
        up,up_good=neighbor_mean(ownq,qg,net['edges'],False)
        for label,values,valid in [('downstream_speed_gradient',down-ownv,down_good&vg),('upstream_demand_gradient',up-ownq,up_good&qg)]:
            coef=project(values,valid&support,phi)
            for k,c in enumerate(coef):parts[f'graph_{label}_t{t-12}_k{k}']=c
            parts[f'graph_{label}_coverage_t{t-12}']=valid.mean()
    # All ramp observations stop at T-5. Nodes with no attached ramp have known0.
    histories=[];validities=[]
    for side in [0,1]:
        counts=ramp_net['counts'][:,side]
        coverage=np.divide(ramps[:,:,side+2],counts[None],out=np.ones(ramps.shape[:2]),where=counts[None]>0)
        valid=(coverage>=.75)|(counts[None]==0)
        ratio=ramps[:,:,side]/net['capacity'][None]
        histories.append(ratio);validities.append(valid)
        for t in [8,11]:
            coef=project(ratio[t],valid[t]&support,phi)
            for k,c in enumerate(coef):parts[f'ramp_{side}_t{t-12}_k{k}']=c
            parts[f'ramp_{side}_coverage_t{t-12}']=valid[t].mean()
    netr=histories[0]-histories[1];valid=validities[0]&validities[1]
    coef=project(netr[11]-netr[8],valid[11]&valid[8]&support,phi)
    for k,c in enumerate(coef):parts[f'ramp_net_change15_k{k}']=c
    parts['clock_sin']=float(x[-1,0,11]);parts['clock_cos']=float(x[-1,0,12])
    return np.array(list(parts.values()),dtype='float32'),list(parts)

def setup():
    # Load only public-data utilities. No parent predictor or weights initialized.
    sys.path.insert(0,str(BASE/'work/repo/experiments/task2_queue_20261024'))
    from task2q.data import make_sample,PANELS
    source=REF.REPRO/'training/original_sources/task2-round10-phase-coverage/ramp_features.py'
    spec=importlib.util.spec_from_file_location('analog_ramp_features',source)
    rf=importlib.util.module_from_spec(spec);spec.loader.exec_module(rf)
    rf.DATA=REF.DATA;rf.WORK=REF.RUN/'rebuild/ramps';rf.WORK.mkdir(parents=True,exist_ok=True)
    bases={}
    for panel in PANELS:
        with np.load(REF.REPRO/'training/basis'/f'{panel}-basis.npz') as f:bases[panel]={k:f[k] for k in ['position','supported']}
    root=REF.RUN/'rebuild/days'
    @lru_cache(maxsize=8)
    def arrays(key):
        with np.load(root/(key+'.npz'),allow_pickle=False) as f:return {k:f[k] for k in f.files}
    def get(e):
        values=arrays(e['day_key']);net=arrays(e['network_key'])
        s=make_sample(values,net,e['origin_slot'],e['panel'])
        assert s['condition']==e['condition'] and s['has_future_queue']==e['has_future_queue']
        return s
    return get,bases,rf,PANELS

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
    protocol=dict(method='Independent whole-corridor analog forecast; no parent prediction/gate/residual',
        training='Frozen June-November fit roster, onset and known proxy future-positive windows only, matching evaluation positive-onset population',
        k=K,neighbors='Nearest16 different calendar days; at most1 analog per day',
        embeddings='Causal speed/flow/occupancy spatial fields, directional gradients on actual released internal edges, T-5 ramp flows/changes, observation coverage, clock',
        control='Same library, labels, distinct-day16 and aggregation; clock-only distance',
        distance='Fit-only standardized features, equally normalized field/graph/ramp/coverage groups; clock group weight.25',
        weights='exp(-squared_distance/median selected squared_distance); neighbor future-label masks used only as training-library supervision weights',
        smoothing='.05 fit-library per-link/horizon prior; no use of query future mask or coverage',
        tuning='FixedK16 and distance design before evaluation. January threshold grid .05.. .95 by.05; no December/February model search',
        February='Same183/200 onset positive windows; reused development, not blind',
        sources=['https://www.sciencedirect.com/science/article/pii/S0968090X15003812','https://www.sciencedirect.com/science/article/pii/S0968090X15003186'],official_S_queue=None)
    write(OUT/'protocol.json',protocol)
    get,bases,rf,panels=setup();roster=REF.entries('june-november-expanded-fit.json')
    roster=[e for e in roster if e['condition']==0 and e['has_future_queue']]
    names=None;checks=[];stats={}
    for panel in panels:
        path=RUN/(panel+'.npz')
        if path.exists():continue
        entries=sorted([e for e in roster if e['panel']==panel],key=lambda e:e['origin']);emb=[];yy=[];mm=[];dates=[];origins=[]
        for j,e in enumerate(entries):
            s=get(e);ramps=rf.ramp_history(s);rnet=rf.topology(panel)
            z,nn=embedding(s,bases[panel],ramps,rnet)
            if names is None:names=nn
            assert nn==names and np.isfinite(z).all()
            emb.append(z);yy.append(s['y']);mm.append(s['m']);dates.append(s['origin'][:10]);origins.append(s['origin'])
            if j==0:
                other=dict(s,y=1-s['y'],m=~s['m'],has_future_queue=not s['has_future_queue'])
                zz,_=embedding(other,bases[panel],ramps,rnet);np.testing.assert_array_equal(z,zz)
                # Ramp perturbation at T/future never changes the T-5 history.
                original=rf.day;date=s['origin'][:10];slot=e['origin_slot']
                def perturb(p,split,dt):
                    values=original(p,split,dt).copy()
                    if dt>=date:values[slot if dt==date else 0:]=999999
                    return values
                try:
                    rf.day=perturb;perturbed=rf.ramp_history(s)
                finally:rf.day=original
                np.testing.assert_array_equal(ramps,perturbed)
                checks.append(dict(panel=panel,labels_ignored=True,ramp_T_and_future_ignored=True))
            if (j+1)%100==0:log(phase='library',panel=panel,windows=j+1,total=len(entries))
        np.savez_compressed(path,X=np.stack(emb),y=np.stack(yy),m=np.stack(mm),dates=np.array(dates),origins=np.array(origins))
        stats[panel]=dict(windows=len(entries),distinct_days=len(set(dates)),nodes=yy[0].shape[1])
        write(OUT/'feature_names.json',names);write(OUT/'causality.json',checks)
    # Same cached, verified arrays as previous experiments. Only <=T fields
    # enter embeddings; query masks/labels are written for scoring separately.
    for pop,file in [('december','december-development.json'),('january','january-global-calibration.json'),('new_phase_population','february-new_phase_population.json'),('old410','february-old410.json')]:
        folder=RUN/pop;folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists():continue
        embeddings=[];ys=[];ms=[];meta=[];cursor=0
        for e in REF.entries(file):
            if e['condition']!=0:continue
            s=get(e);z,nn=embedding(s,bases[s['panel']],rf.ramp_history(s),rf.topology(s['panel']))
            embeddings.append(z);n=s['x'].shape[1];ys.append(s['y'].T);ms.append(s['m'].T)
            meta.append(dict(panel=s['panel'],origin=s['origin'],start=cursor,end=cursor+n,positive=bool(s['has_future_queue'])));cursor+=n
        np.save(folder/'X.npy',np.stack(embeddings));np.save(folder/'y.npy',np.concatenate(ys));np.save(folder/'m.npy',np.concatenate(ms));write(folder/'complete.json',dict(windows=meta))
        log(phase='prepared',population=pop,windows=len(meta))
    if not stats:
        for panel in panels:
            with np.load(RUN/(panel+'.npz')) as f:stats[panel]=dict(windows=len(f['X']),distinct_days=len(set(f['dates'])),nodes=f['y'].shape[2])
    write(OUT/'library.json',stats);log(phase='PREPARE_COMPLETE',library_windows=sum(v['windows'] for v in stats.values()))

def distance_transform(lib,query,names,clock_only):
    mean=lib.mean(0);scale=np.maximum(lib.std(0),.05)
    a=(lib-mean)/scale;b=(query-mean)/scale
    groups={}
    for j,n in enumerate(names):
        group='clock' if n.startswith('clock') else 'coverage' if 'coverage' in n else n.split('_')[0]
        groups.setdefault(group,[]).append(j)
    selected=groups['clock'] if clock_only else list(range(len(names)))
    for group,ids in groups.items():
        weight=(1. if clock_only else .25) if group=='clock' else 1.
        factor=np.sqrt(weight/len(ids));a[:,ids]*=factor;b[:,ids]*=factor
    return a[:,selected],b[:,selected]

def forecast(lib,query,names,clock_only=False):
    a,b=distance_transform(lib['X'],query,names,clock_only)
    distances=cdist(b,a,metric='sqeuclidean');out=[];neighbors=[]
    mask=lib['m'].astype(float);labels=np.where(mask>0,lib['y'],0)
    prior=np.divide((labels*mask).sum(0),mask.sum(0),out=np.zeros(mask.shape[1:]),where=mask.sum(0)>0)
    for dist in distances:
        seen=set();chosen=[]
        for j in np.argsort(dist,kind='stable'):
            date=str(lib['dates'][j])
            if date not in seen:chosen.append(j);seen.add(date)
            if len(chosen)==K:break
        chosen=np.array(chosen,int);den=max(float(np.median(dist[chosen])),1e-8)
        weight=np.exp(-dist[chosen]/den);weight/=weight.sum()
        known=(weight[:,None,None]*mask[chosen]).sum(0)
        positive=(weight[:,None,None]*mask[chosen]*labels[chosen]).sum(0)
        p=(positive+.05*prior)/(known+.05)
        out.append(p.T.astype('float32'));neighbors.append([str(lib['origins'][j]) for j in chosen])
    return out,neighbors

def evaluate():
    names=read(OUT/'feature_names.json');reports={};raw={};metas={};neighbors={}
    for pop in ['december','january','new_phase_population','old410']:
        folder=RUN/pop;X=np.load(folder/'X.npy');meta=read(folder/'complete.json');y=np.load(folder/'y.npy');m=np.load(folder/'m.npy');metas[pop]=(meta,y,m)
        for variant in ['state_analog','clock_analog']:
            predictions=[None]*len(meta['windows']);nbs=[None]*len(predictions)
            for panel in sorted({e['panel'] for e in meta['windows']}):
                idx=[i for i,e in enumerate(meta['windows']) if e['panel']==panel]
                with np.load(RUN/(panel+'.npz')) as f:lib={k:f[k] for k in f.files}
                pred,nn=forecast(lib,X[idx],names,variant=='clock_analog')
                for i,p,nb in zip(idx,pred,nn):predictions[i]=p;nbs[i]=nb
            p=np.concatenate(predictions);raw[(pop,variant)]=p;np.save(OUT/f'{pop}-{variant}-probability.npy',p);neighbors[pop+'/'+variant]=nbs
    selections={}
    jm,jy,jmask=metas['january']
    for variant in ['state_analog','clock_analog']:
        grid=np.round(np.arange(.05,.951,.05),2).tolist()
        metrics=[REF.score_population(raw[('january',variant)],dict(y=jy,m=jmask),jm,t)['onset_IoU'] for t in grid]
        selections[variant]=dict(threshold=grid[int(np.argmax(metrics))],grid=grid,IoU=metrics)
    write(OUT/'January_thresholds.json',selections)
    for pop,(meta,y,m) in metas.items():
        parent=np.load(REF.RUN/pop/'parent.npy');baseline=ar.score(parent,y,m,meta)
        report={'R127':baseline}
        for variant in ['state_analog','clock_analog']:
            p=raw[(pop,variant)];t=selections[variant]['threshold']
            # ar.score has fixed .5; binary predictions preserve selected decision.
            result=ar.score((p>=t).astype('float32'),y,m,meta)
            result['threshold']=t;result['IoU_vs_R127']=REF.bootstrap(result,baseline);result['FP_vs_R127']=ar.fp_bootstrap(result,baseline)
            result['fixed05']=ar.score(p,y,m,meta)
            report[variant]=result
        report['state_vs_clock_IoU']=REF.bootstrap(report['state_analog'],report['clock_analog'])
        reports[pop]=report
        log(phase='evaluated',population=pop,R127=baseline['onset_IoU'],state=report['state_analog']['onset_IoU'],clock=report['clock_analog']['onset_IoU'],state_FP=report['state_analog']['cell_counts']['FP_cells'],state_FN=report['state_analog']['cell_counts']['FN_cells'])
    write(OUT/'neighbors.json',neighbors);write(OUT/'results.json',dict(populations=reports,official_S_queue=None,February_reused_development=True))
    log(phase='EVALUATION_COMPLETE')

if __name__=='__main__':
    START=time.time();ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','evaluate']);args=ap.parse_args()
    {'prepare':prepare,'evaluate':evaluate}[args.action]()
