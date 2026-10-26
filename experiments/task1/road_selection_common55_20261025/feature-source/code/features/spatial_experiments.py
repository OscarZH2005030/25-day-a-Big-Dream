"""Fit residual feature groups on January only; evaluate February once.

Only released masked mainline and quality-filtered ramp observations enter X.
Unmasked observations enter y exclusively. Temporal baseline fits Jun-Dec for
January residual training, Jun-Jan for February evaluation.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.linalg import solve
from temporal_features import replace_temporal,KEYS

sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor/src'))
from task1 import baseline_task1_historical_mean as b

GROUPS={'bias_only':[], 'spatial':['spatial'], 'topology':['topology'],
        'ramps':['ramps'], 'spatial_topology':['spatial','topology'],
        'all_features':['spatial','topology','ramps']}
ALPHA=0.001


def profile_pair(folder):
    paths=b.files(folder,'train')
    link_ids=sorted(pd.read_parquet(paths[0],columns=['link_id']).link_id.astype(str).unique())
    mapping={x:i for i,x in enumerate(link_ids)}
    shape=(len(link_ids),7,288,2)
    sums=[np.zeros(shape),np.zeros(shape)];nums=[np.zeros(shape),np.zeros(shape)]
    for path in paths:
        if 'year_month=2031-02' in str(path):continue
        january='year_month=2031-01' in str(path)
        d=pd.read_parquet(path,columns=['timestamp','link_id','speed_kmh','flow_vph','pct_observed'])
        x=d.link_id.astype(str).map(mapping).to_numpy(dtype=int)
        weekday,tod=b.slot_values(d)
        values=d[['speed_kmh','flow_vph']].to_numpy(dtype=float)
        good=(d.pct_observed.to_numpy()>=75)&np.isfinite(values).all(axis=1)
        for ch in range(2):
            np.add.at(sums[int(january)],(x[good],weekday[good],tod[good],ch),values[good,ch])
            np.add.at(nums[int(january)],(x[good],weekday[good],tod[good],ch),1)
    result=[]
    for ss,nn in [(sums[0],nums[0]),(sums[0]+sums[1],nums[0]+nums[1])]:
        fallback=ss.sum(axis=(1,2),keepdims=True)/np.maximum(nn.sum(axis=(1,2),keepdims=True),1)
        result.append(np.where(nn>0,ss/np.maximum(nn,1),fallback))
    return mapping,result


def graph_neighbors(topology,links):
    index={x:i for i,x in enumerate(links)}
    result=[]
    for column in ['incoming_link_ids','outgoing_link_ids']:
        graph={str(r.mainline_link_id):str(getattr(r,column)).split(';') for r in topology.itertuples()}
        all_candidates=[]
        for link in links:
            visited={link};queue=deque((x,1) for x in graph.get(link,[]));found=[]
            while queue and len(found)<3:
                node,depth=queue.popleft()
                if node in visited or node=='nan':continue
                visited.add(node)
                if node in index:found.append(index[node])
                if depth<12:
                    queue.extend((x,depth+1) for x in graph.get(node,[]) if x not in visited)
            all_candidates.append(found)
        result.append(all_candidates)
    return result


class Features:
    def __init__(self,folder,first_obs,include_ramps=True):
        self.folder=folder
        self.include_ramps=include_ramps
        self.links=sorted(first_obs.link_id.astype(str).unique())
        self.index={x:i for i,x in enumerate(self.links)}
        net=pd.read_csv(folder/'network/links.csv',dtype={'link_id':str}).set_index('link_id').reindex(self.links)
        self.lanes=net.lanes.to_numpy(dtype=float)
        if not np.isfinite(self.lanes).all() or (self.lanes<=0).any():raise ValueError('Invalid lane metadata')
        self.static=np.column_stack([net.length_km.to_numpy()/5., self.lanes/6.,
                                     net.capacity_vph.to_numpy()/self.lanes/2500.,
                                     net.free_speed_kmh.to_numpy()/120.])
        positions=first_obs.groupby('link_id').milepost.median().reindex(self.links).to_numpy()
        if not np.isfinite(positions).all():raise ValueError('Missing mileposts')
        self.order=np.argsort(positions,kind='stable');self.inverse=np.argsort(self.order)
        self.neighbors=graph_neighbors(pd.read_csv(folder/'network/lwr_mainline_topology.csv'),self.links)
        if include_ramps:
            attachments=pd.read_csv(folder/'network/ramp_attachment_map.csv',dtype=str)
            self.attach=attachments[attachments.nearest_mainline_link_id.isin(self.links)].copy()
            self.ramp_files={p.name:p for p in (folder/'train/ramp_states').rglob('*.parquet')}

    def get(self,obs,targets,pred):
        obs=obs.copy()
        obs['link_id']=obs.link_id.astype(str);obs['timestamp']=obs.timestamp.astype(str)
        grid=sorted(obs.timestamp.unique())
        ti=pd.Index(grid).get_indexer(targets.timestamp.astype(str))
        ci=pd.Index(self.links).get_indexer(targets.link_id.astype(str))
        if (ti<0).any() or (ci<0).any():raise ValueError('Unmapped target')
        norm=np.column_stack([np.full(len(targets),25.),600.*self.lanes[ci]])
        normalized=pred/norm
        mats=[]
        for ch,column in enumerate(['speed_kmh','flow_vph']):
            d=obs.assign(value=obs[column].where(obs.pct_observed>=75))
            m=d.pivot(index='timestamp',columns='link_id',values='value').reindex(index=grid,columns=self.links).to_numpy()
            m=m/(25. if ch==0 else 600.*self.lanes[None,:])
            mats.append(m)
        spatial=[];spatial_names=[];topo=[];topo_names=[]
        for ch,m in enumerate(mats):
            name=['speed','flow_per_lane'][ch]
            ordered=pd.DataFrame(m[:,self.order])
            marker=pd.DataFrame(np.where(ordered.notna(),np.arange(len(self.links))[None,:],np.nan))
            for side in ['left','right']:
                if side=='left':
                    neighbor=ordered.ffill(axis=1).shift(1,axis=1)
                    gap=np.arange(len(self.links))[None,:]-marker.ffill(axis=1).shift(1,axis=1).to_numpy()
                else:
                    neighbor=ordered.bfill(axis=1).shift(-1,axis=1)
                    gap=marker.bfill(axis=1).shift(-1,axis=1).to_numpy()-np.arange(len(self.links))[None,:]
                neighbor=neighbor.where(gap<=3).to_numpy()[:,self.inverse]
                values=neighbor[ti,ci];valid=np.isfinite(values)
                spatial.extend([np.where(valid,values-normalized[:,ch],0),valid.astype(float)])
                spatial_names.extend([f'{side}_{name}_residual',f'{side}_{name}_available'])
            for direction,candidates in zip(['upstream','downstream'],self.neighbors):
                neighbor=np.full_like(m,np.nan)
                for j,choices in enumerate(candidates):
                    for k in choices:
                        missing=~np.isfinite(neighbor[:,j])
                        neighbor[missing,j]=m[missing,k]
                values=neighbor[ti,ci];valid=np.isfinite(values)
                topo.extend([np.where(valid,values-normalized[:,ch],0),valid.astype(float)])
                topo_names.extend([f'{direction}_{name}_residual',f'{direction}_{name}_available'])
        for i,name in enumerate(['length','lanes','capacity_per_lane','free_speed']):
            topo.append(self.static[ci,i]);topo_names.append(name)
        if not self.include_ramps:
            return {'spatial':np.column_stack(spatial),'topology':np.column_stack(topo)}, {'spatial':spatial_names,'topology':topo_names},norm
        # The ramp timestamp matches the Task 1 target, so this is offline use.
        date=str(targets.timestamp.iloc[0])[:10]
        ramp_name='synthetic_ramp_'+date.replace('-','_')+'.parquet'
        path=self.ramp_files.get(ramp_name)
        if path is None:raise FileNotFoundError(ramp_name)
        ramp=pd.read_parquet(path)
        ramp['ramp_link_id']=ramp.ramp_link_id.astype(str)
        ramp['timestamp']=ramp.timestamp.astype(str)
        ramp=ramp.merge(self.attach[['ramp_link_id','nearest_mainline_link_id']],on='ramp_link_id',validate='many_to_one')
        ramp['valid']=((ramp.pct_observed>=75)&np.isfinite(ramp.flow_vph)).astype(float)
        ramp['available_flow']=ramp.flow_vph.where(ramp.valid>0)
        ramps=[];ramp_names=[]
        for kind in ['OR','FR']:
            d=ramp[ramp.ramp_type==kind]
            total=d.groupby(['timestamp','nearest_mainline_link_id']).available_flow.sum(min_count=1).unstack()
            coverage=d.groupby(['timestamp','nearest_mainline_link_id']).valid.mean().unstack()
            total=total.reindex(index=grid,columns=self.links)
            coverage=coverage.reindex(index=grid,columns=self.links).fillna(0)
            now=total.to_numpy()[ti,ci]/norm[:,1]
            lag=total.shift(1).to_numpy()[ti,ci]/norm[:,1]
            avail=coverage.to_numpy()[ti,ci]
            ramps.extend([np.nan_to_num(now),avail,np.nan_to_num(now-lag)])
            ramp_names.extend([f'{kind}_flow_per_lane',f'{kind}_coverage',f'{kind}_change_5m'])
        return {'spatial':np.column_stack(spatial),'topology':np.column_stack(topo),
                'ramps':np.column_stack(ramps)}, {'spatial':spatial_names,'topology':topo_names,'ramps':ramp_names},norm


def design(groups,chosen):
    n=len(next(iter(groups.values())))
    return np.column_stack([np.ones(n),*[groups[x] for x in chosen]])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--release-root',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    (out/'models').mkdir(exist_ok=True)
    manifest=json.loads((args.release_root/'config/corridors.json').read_text())
    results=[];feature_names=None
    for info in manifest['panels']:
        panel=info['corridor_id'];folder=args.release_root/'corridors'/panel
        print(f'[profiles] {panel}',flush=True)
        mapping,profiles=profile_pair(folder)
        target_parts=[]
        for part in pd.read_csv(args.release_root/'task1'/panel/'train/sample_submission_state.csv',dtype=str,chunksize=200000):
            target_parts.append(part[part.timestamp.str.startswith(('2031-01','2031-02'))])
        target=pd.concat(target_parts,ignore_index=True)
        targets_by_day={day:d for day,d in target.groupby(target.timestamp.str[:10])}
        truth_paths={p.name:p for p in b.files(folder,'train')}
        masked_paths={str(pd.to_datetime(p.stem.removeprefix('synthetic_mainline_').replace('_','-')).date()):p
                      for p in b.masked_files(folder,'train') if 'synthetic_mainline_2031_01_' in p.name or 'synthetic_mainline_2031_02_' in p.name}
        first=pd.read_parquet(next(iter(masked_paths.values())))
        features=Features(folder,first)
        coefficients={};stats={};train_n=0
        for month,profile in [('2031-01',profiles[0]),('2031-02',profiles[1])]:
            if month=='2031-02':
                for variant,(xx,xy,n) in stats.items():
                    regularizer=np.eye(len(xx))*ALPHA*n;regularizer[0,0]=1e-8
                    coefficients[variant]=solve(xx+regularizer,xy,assume_a='pos')
                np.savez(out/'models'/f'{panel}.npz',**coefficients)
            errors={v:{r:[0,0.,0.] for r in ['R1','R2','R3']} for v in ['temporal_baseline',*GROUPS]}
            for date,targets in sorted(targets_by_day.items()):
                if not date.startswith(month):continue
                path=masked_paths[date]
                obs=pd.read_parquet(path)
                truth=pd.read_parquet(truth_paths[path.name])
                for key in KEYS:truth[key]=truth[key].astype(str)
                truth=truth.set_index(KEYS).reindex(pd.MultiIndex.from_frame(targets[KEYS]))
                y=truth[['speed_kmh','flow_vph']].to_numpy(dtype=float)
                weekday,tod=b.slot_values(targets);ix=targets.link_id.map(mapping).to_numpy(dtype=int)
                base=profile[ix,weekday,tod,:]
                base,_=replace_temporal(obs,targets,base)
                groups,names,norm=features.get(obs,targets,base)
                if feature_names is None:feature_names=names
                if not np.isfinite(y).all():raise ValueError('Missing target truth')
                if month=='2031-01':
                    residual=(y-base)/norm
                    for variant,chosen in GROUPS.items():
                        x=design(groups,chosen)
                        if variant not in stats:stats[variant]=[np.zeros((x.shape[1],x.shape[1])),np.zeros((x.shape[1],2)),0]
                        stats[variant][0]+=x.T@x;stats[variant][1]+=x.T@residual;stats[variant][2]+=len(x)
                    train_n+=len(y)
                    continue
                predictions={'temporal_baseline':base}
                for variant,chosen in GROUPS.items():
                    correction=design(groups,chosen)@coefficients[variant]
                    predictions[variant]=np.maximum(base+correction*norm,0)
                regimes=targets.mask_regime.to_numpy()
                for variant,pred in predictions.items():
                    delta=pred-y;delta[:,1]/=features.lanes[targets.link_id.map(features.index).to_numpy(dtype=int)]
                    if not np.isfinite(delta).all():raise ValueError('Nonfinite model output')
                    for regime,acc in errors[variant].items():
                        sel=regimes==regime;acc[0]+=int(sel.sum());acc[1]+=float(np.sum(delta[sel,0]**2));acc[2]+=float(np.sum(delta[sel,1]**2))
            if month=='2031-02':
                for variant,regimes in errors.items():
                    scores=[]
                    for n,sv,sq in regimes.values():
                        if not n:raise ValueError('No regime targets')
                        scores.append(.54*max(0,1-np.sqrt(sv/n)/25)+.46*max(0,1-np.sqrt(sq/n)/600))
                    results.append({'panel':panel,'family':info['family_id'],'variant':variant,
                                    'S_state':float(np.mean(scores)),'fit_residual_rows':train_n,
                                    'holdout_targets':sum(x[0] for x in regimes.values())})
        (out/'progress.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        print(f'[done] {panel}',flush=True)
    frame=pd.DataFrame(results)
    overall=frame.groupby(['variant','family']).S_state.mean().groupby('variant').mean().to_dict()
    reference=overall['temporal_baseline']
    baseline_by_panel=frame[frame.variant=='temporal_baseline'].set_index('panel').S_state
    summary=[]
    for variant,score in overall.items():
        values=frame[frame.variant==variant].set_index('panel').S_state-baseline_by_panel
        summary.append({'variant':variant,'S_state':score,'delta_vs_temporal':score-reference,
                        'weighted_state_delta':.35*(score-reference),
                        'improved_panels':int((values>1e-10).sum()),'worse_panels':int((values< -1e-10).sum())})
    result={'calendar_profile_fit':'June-December for January; June-January for February',
            'residual_model_fit':'January 2031 released Task 1 targets only','holdout':'February 2031',
            'ridge_alpha':ALPHA,'features':feature_names,'models':GROUPS,
            'note':'Fixed residual ridge models, not the maximum possible value of each data source. Task 1 only; no hidden total/physics claims.',
            'summary':summary,'details':results}
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
