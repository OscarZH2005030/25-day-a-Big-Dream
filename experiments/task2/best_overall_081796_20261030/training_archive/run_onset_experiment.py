"""Reproducible R127 / two-stage / occupancy-profile ablation experiment.

Run using the recovered portable Python. All output paths are explicit.
Actions: prepare, train, evaluate. See experiment_protocol.json for fixed choices.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='4'
import argparse, copy, hashlib, importlib, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

BASE=Path(__file__).resolve().parents[1]
REPRO=BASE/'work/repo/experiments/task2/r127_reproduction_20261027'
PACKAGE=BASE/'work/r127/ConditionalPosition-R127'
DATA=BASE/'work/local_transfer/data/kaggle_public'
RUN=BASE/'work/onset_experiment'
OUT=BASE/'outputs/onset_experiment'
PATTERNS=((np.arange(64)[:,None] >> np.arange(6)) & 1).astype('float32')
SEEDS=[20261029,20261030,20261031]
EPOCHS=30

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2),encoding='utf8')
def log(**v):print(json.dumps(dict(seconds=round(time.time()-START,1),**v)),flush=True)

def backend():
    sys.path.insert(0,str(REPRO))
    from repro_common import setup
    merged={}
    for name in ['training-array-hashes.json','evaluation-array-hashes.json']:
        merged.update(read(REPRO/'indices'/name))
    hp=RUN/'merged-hashes.json';write(hp,merged)
    backend_root=RUN/('intervention_rebuild' if len(sys.argv)>1 and sys.argv[1]=='intervention' else 'rebuild')
    model,sample,r2,audit=setup(PACKAGE,DATA,backend_root,True,hp,[OUT,RUN,Path(__file__).parent])
    # Historical chains share anchor predictor instances. Cache their repeated
    # calls within one public sample; clear between samples/interventions.
    # This changes no arithmetic inside a prediction and is checked bitwise.
    memo={};visited=set()
    def wrap_children(obj):
        if id(obj) in visited:return
        visited.add(id(obj))
        for child in list(vars(obj).values()):
            if hasattr(child,'probability') and callable(child.probability):
                wrap_children(child)
        if obj is model:return
        bound=obj.probability;identity=id(obj)
        def cached(s):
            key=(identity,id(s))
            if key not in memo:memo[key]=bound(s).copy()
            return memo[key].copy()
        obj.probability=cached
    original_top=model.probability
    # Save unwrapped methods so first use in every road can verify equivalence.
    originals={}
    def collect(obj):
        if id(obj) in originals:return
        originals[id(obj)]=(obj,obj.probability)
        for child in list(vars(obj).values()):
            if hasattr(child,'probability') and callable(child.probability):collect(child)
    collect(model);wrap_children(model)
    cached_methods={i:obj.probability for i,(obj,_) in originals.items()}
    checked=set()
    def cached_top(s):
        memo.clear()
        if s['panel'] not in checked:
            for obj,bound in originals.values():obj.probability=bound
            reference=original_top(s)
            for i,(obj,_) in originals.items():obj.probability=cached_methods[i]
            model.probability=cached_top
            memo.clear();fast=original_top(s)
            np.testing.assert_array_equal(reference,fast);checked.add(s['panel'])
        else:fast=original_top(s)
        memo.clear();return fast
    model.probability=cached_top
    from timing_core import features
    from position_mode_core import extras
    from two_stage_onset import causal_precursors
    bases={}
    for p in r2.PANELS:
        with np.load(REPRO/'training/basis'/f'{p}-basis.npz') as f:
            bases[p]={k:f[k] for k in ['position','random','supported']}
    def featurize(s):
        prec,names=causal_precursors(s)
        z=np.column_stack([features(s),extras(s,bases[s['panel']],'position'),prec,np.isfinite(prec)])
        assert z.shape==(s['x'].shape[1],940)
        return z.astype('float32')
    original_sample=sample
    derived_keys=set()
    def sample(e):
        if e['day_key'] in merged:return original_sample(e)
        # R127's published training-array roster covers the rows used by its
        # selected heads, not every negative day in the larger frozen fit roster.
        # These extra days come from the separately SHA256-verified raw dataset.
        from task2q.data import network,day,make_sample
        nk,dk=e['network_key'],e['day_key']
        if not (r2.CACHE/(nk+'.npz')).exists():
            net=network(DATA,e['panel'])
            for k,rec in merged[nk].items():
                value=net[k]
                assert list(value.shape)==rec['shape'] and str(value.dtype)==rec['dtype']
                assert hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()==rec['sha256']
            np.savez_compressed(r2.CACHE/(nk+'.npz'),**net)
        if dk not in derived_keys:
            value=day(DATA,e['panel'],e['origin'][:10],cache_dir=backend_root/'public-day-cache')
            np.savez_compressed(r2.CACHE/(dk+'.npz'),**{k:v for k,v in value.items() if isinstance(v,np.ndarray)})
            derived_keys.add(dk)
        s=make_sample(r2.arrays(dk),r2.arrays(nk),e['origin_slot'],e['panel'])
        assert s['condition']==e['condition'] and s['has_future_queue']==e['has_future_queue']
        return s
    return model,sample,r2,featurize,audit

def entries(name):return read(REPRO/'indices'/name)['samples']

def prepare():
    RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(seeds=SEEDS,epochs=EPOCHS,batch_size=4096,hidden=[256,128],dropout=.1,
        learning_rate=.001,weight_decay=.0001,
        fit='All frozen June-November onset windows; all known-positive links and <=64 random remaining supervised links per window',
        loss='Marginal likelihood over all compatible 6-bit patterns; two-stage event x 63 conditional nonempty patterns',
        weighting='Within positive window .7 positive-link mass/.3 remaining-link mass; positive/negative window mass .8/.2 within road; IPW for sampled remaining links',
        ablation='Zero the 104 occupancy position-profile columns only; same input width, architecture, initialization, batches, epochs and weights',
        dimensions=940,baseline='Frozen R127; verify both February populations exactly',
        threshold='Primary .5 for every candidate; secondary choose one threshold per seed/variant on January from fixed grid .05.. .90 step .05; no per-road selection',
        december='diagnostic only; fixed 30 epochs, no checkpoint selection',
        February='Previously used development evaluation; both frozen populations reported separately; not blind',
        official_S_queue=None)
    write(OUT/'experiment_protocol.json',protocol)
    model,sample,r2,features,audit=backend()
    all_entries=entries('june-november-expanded-fit.json')
    fit=sorted([e for e in all_entries if e['condition']==0],key=lambda e:(e['panel'],e['origin']))
    assert all('2030-06'<=e['origin'][:7]<='2030-11' for e in fit)
    roster=RUN/'fit/complete.json'
    if not roster.exists():
        folder=roster.parent;folder.mkdir(parents=True,exist_ok=True)
        # Exact capacity from public network node counts; disk-backed, bounded RAM.
        from task2q.data import network
        ns={p:len(network(DATA,p)['free_speed']) for p in r2.PANELS}
        cap=sum(ns[e['panel']] for e in fit)
        resume_windows=0
        if (folder/'partial.json').exists():resume_windows=read(folder/'partial.json')['processed_windows']
        elif (folder/'X.npy').exists() and (BASE/'work/prepare.log').exists():
            for line in (BASE/'work/prepare.log').read_text(encoding='utf-8-sig').splitlines():
                if line.startswith('{'):
                    row=json.loads(line)
                    if row.get('phase')=='fit_features':resume_windows=row['windows']
        mode='r+' if resume_windows else 'w+'
        X=np.lib.format.open_memmap(folder/'X.npy',mode=mode,dtype='float32',shape=(cap,940))
        y=np.lib.format.open_memmap(folder/'y.npy',mode=mode,dtype='uint8',shape=(cap,6))
        m=np.lib.format.open_memmap(folder/'m.npy',mode=mode,dtype='bool',shape=(cap,6))
        w=np.lib.format.open_memmap(folder/'w.npy',mode=mode,dtype='float32',shape=(cap,))
        counts={}
        for e in fit:counts[(e['panel'],e['has_future_queue'])]=counts.get((e['panel'],e['has_future_queue']),0)+1
        rng=np.random.default_rng(SEEDS[0]);cursor=0;windows=[];checks=[]
        for j,e in enumerate(fit):
            s=sample(e);yy=s['y'].T;mm=s['m'].T
            z=features(s) if j>=resume_windows else None
            known=mm.any(1);pos=((yy>0)&mm).any(1);neg=known&~pos
            pi=np.flatnonzero(pos);ni=np.flatnonzero(neg);k=min(64,len(ni))
            chosen=rng.choice(ni,k,replace=False) if k else ni
            ids=np.r_[pi,chosen]
            if len(ids)==0:continue
            mass=(.8 if e['has_future_queue'] else .2)/counts[(e['panel'],e['has_future_queue'])]
            weights=np.zeros(len(ids))
            if len(pi):
                weights[:len(pi)]=(.7 if k else 1.)/len(pi)
                if k:weights[len(pi):]=.3/k
            else:weights[:]=1./len(ids)
            # Equal probability sampling of remaining links: normalized IPW gives
            # equal mass per retained negative/unknown event-identity row.
            weights*=mass
            end=cursor+len(ids)
            if j>=resume_windows:X[cursor:end]=z[ids];y[cursor:end]=yy[ids];m[cursor:end]=mm[ids];w[cursor:end]=weights
            else:
                np.testing.assert_array_equal(y[cursor:end],yy[ids]);np.testing.assert_array_equal(m[cursor:end],mm[ids])
            windows.append(dict(panel=e['panel'],origin=e['origin'],start=cursor,end=end,positive=bool(e['has_future_queue']),selected_links=ids.tolist()))
            cursor=end
            if not any(c['panel']==e['panel'] for c in checks):
                if z is None:z=features(s)
                altered=copy.deepcopy(s);altered['y']=1-s['y'];altered['m']=~s['m'];altered['has_future_queue']=not s['has_future_queue']
                np.testing.assert_allclose(z,features(altered),equal_nan=True)
                v=copy.deepcopy(r2.arrays(e['day_key']));slot=e['origin_slot']
                for key in ['speed','flow','occupancy','quality','origin_speed','origin_flow','origin_occupancy','origin_quality']:
                    v[key][slot+1:]=999
                v['eligible'][slot+1:]=False
                from task2q.data import make_sample
                other=make_sample(v,s['network'],slot,s['panel'])
                np.testing.assert_allclose(z,features(other),equal_nan=True)
                checks.append(dict(panel=e['panel'],label_and_future_mainline_perturbation=True))
            if (j+1)%100==0:
                for arr in [X,y,m,w]:arr.flush()
                write(folder/'partial.json',dict(processed_windows=j+1,rows=cursor))
                log(phase='fit_features',windows=j+1,total=len(fit),rows=cursor)
        for arr in [X,y,m,w]:arr.flush()
        write(roster,dict(rows=cursor,windows=windows,causality=checks,fit_windows=len(fit),capacity=cap))
        del X,y,m,w
    # Evaluation windows are not subsampled; baseline and candidate see the same masks.
    reports={}
    for name,file in [('december','december-development.json'),('january','january-global-calibration.json'),('new_phase_population','february-new_phase_population.json'),('old410','february-old410.json')]:
        folder=RUN/name;folder.mkdir(exist_ok=True)
        path=folder/'complete.json'
        population=entries(file)
        if not path.exists():
            xx=[];yy=[];mm=[];pp=[];metadata=[];baseline_samples=[];baseline_probs=[];cursor=0
            for j,e in enumerate(population):
                s=sample(e);p=model.probability(s)
                baseline_samples.append(s);baseline_probs.append(p)
                if s['condition']==0:
                    z=features(s);n=len(z)
                    xx.append(z);yy.append(s['y'].T);mm.append(s['m'].T);pp.append(p.T)
                    metadata.append(dict(panel=e['panel'],origin=e['origin'],start=cursor,end=cursor+n,positive=bool(s['has_future_queue'])))
                    cursor+=n
                if (j+1)%50==0:log(phase=name,windows=j+1,total=len(population))
            for key,value in [('X',np.concatenate(xx)),('y',np.concatenate(yy)),('m',np.concatenate(mm)),('parent',np.concatenate(pp))]:np.save(folder/(key+'.npy'),value)
            score=r2.score(baseline_samples,baseline_probs,[.5,.25])
            if name in ['new_phase_population','old410']:
                expected=read(REPRO/'expected-metrics.json')['reports'][name+'_selected']
                assert abs(score['proxy_IoU']-expected['proxy_IoU'])<1e-12,(name,score,expected)
                for k,v in score['by_panel_condition'].items():assert abs(v-expected['by_panel_condition'][k])<1e-12
            write(path,dict(windows=metadata,rows=cursor,baseline_all_conditions=score,raw_array_hash_verified=True))
            del baseline_samples,baseline_probs
        reports[name]=read(path)['baseline_all_conditions']
        log(phase='baseline_verified',population=name,score=reports[name]['proxy_IoU'])
    write(OUT/'r127_reproduction.json',dict(reports=reports,February_populations_exact=True,official_S_queue=None))
    log(phase='PREPARE_COMPLETE')

def prepare_evaluation(development_only=False):
    """Independent calendar partitions; may run alongside fit preprocessing."""
    model,sample,r2,features,audit=backend()
    reports={}
    for name,file in [('december','december-development.json'),('january','january-global-calibration.json'),('new_phase_population','february-new_phase_population.json'),('old410','february-old410.json')]:
        if development_only and name!='december':continue
        folder=RUN/name;folder.mkdir(exist_ok=True);path=folder/'complete.json'
        if path.exists():continue
        xx=[];yy=[];mm=[];pp=[];metadata=[];ss=[];ps=[];cursor=0;population=entries(file)
        for j,e in enumerate(population):
            s=sample(e);p=model.probability(s);ss.append(s);ps.append(p)
            if s['condition']==0:
                z=features(s);n=len(z);xx.append(z);yy.append(s['y'].T);mm.append(s['m'].T);pp.append(p.T)
                metadata.append(dict(panel=e['panel'],origin=e['origin'],start=cursor,end=cursor+n,positive=bool(s['has_future_queue'])));cursor+=n
            if (j+1)%50==0:log(phase=name,windows=j+1,total=len(population))
        for key,value in [('X',np.concatenate(xx)),('y',np.concatenate(yy)),('m',np.concatenate(mm)),('parent',np.concatenate(pp))]:np.save(folder/(key+'.npy'),value)
        score=r2.score(ss,ps,[.5,.25])
        if name in ['new_phase_population','old410']:
            golden=read(REPRO/'expected-metrics.json')['reports'][name+'_selected']
            assert abs(score['proxy_IoU']-golden['proxy_IoU'])<1e-12
            for key,value in score['by_panel_condition'].items():assert abs(value-golden['by_panel_condition'][key])<1e-12
        write(path,dict(windows=metadata,rows=cursor,baseline_all_conditions=score,raw_array_hash_verified=True))
        log(phase='baseline_verified',population=name,score=score['proxy_IoU'])
    log(phase='EVALUATION_PREPARED')
    if not development_only:
        reports={name:read(RUN/name/'complete.json')['baseline_all_conditions'] for name in ['december','january','new_phase_population','old410']}
        write(OUT/'r127_reproduction.json',dict(reports=reports,February_populations_exact=True,official_S_queue=None))

class TwoStage(nn.Module):
    def __init__(self,d=940):
        super().__init__()
        self.shared=nn.Sequential(nn.Linear(d,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,128),nn.GELU(),nn.Dropout(.1))
        self.event=nn.Linear(128,1);self.pattern=nn.Linear(128,63)
    def forward(self,x):
        z=self.shared(x);return self.event(z).squeeze(-1),self.pattern(z)

def partial_loss(a,b,y,m):
    patterns=torch.as_tensor(PATTERNS,device=y.device)
    valid=((patterns[None]==y[:,None])|~m[:,None]).all(-1)
    lp=torch.cat([F.logsigmoid(-a)[:,None],F.logsigmoid(a)[:,None]+F.log_softmax(b,dim=-1)],dim=-1)
    return -torch.logsumexp(lp.masked_fill(~valid,-torch.inf),dim=-1)

def load(name):
    folder=RUN/name;meta=read(folder/'complete.json')
    return {k:np.load(folder/(k+'.npy'),mmap_mode='r')[:meta['rows']] for k in ['X','y','m']},meta

def normalization(X):
    path=RUN/'normalization.npz'
    if path.exists():
        with np.load(path) as f:return f['mean'],f['scale']
    total=np.zeros(X.shape[1]);squares=total.copy();count=total.copy()
    for j in range(0,len(X),8192):
        a=np.asarray(X[j:j+8192],float);valid=np.isfinite(a);total+=np.where(valid,a,0).sum(0);squares+=np.where(valid,a*a,0).sum(0);count+=valid.sum(0)
    mean=total/np.maximum(count,1);scale=np.sqrt(np.maximum(squares/np.maximum(count,1)-mean*mean,1e-4))
    np.savez(path,mean=mean.astype('float32'),scale=scale.astype('float32'))
    return mean.astype('float32'),scale.astype('float32')

def occupancy_cols():return np.array([557+t*24+16+k for t in range(13) for k in range(8)])

def transform(x,mean,scale,variant):
    z=np.nan_to_num((np.asarray(x)-mean)/scale,nan=0,posinf=0,neginf=0).clip(-8,8)
    if variant=='no_occupancy_profile':z[:,occupancy_cols()]=0
    return z.astype('float32',copy=False)

def predict(model,X,mean,scale,variant,device):
    model.eval();out=[]
    with torch.inference_mode():
        pats=torch.as_tensor(PATTERNS[1:],device=device)
        for j in range(0,len(X),4096):
            z=torch.from_numpy(transform(X[j:j+4096],mean,scale,variant)).to(device)
            a,b=model(z);out.append((a.sigmoid()[:,None]*(b.softmax(-1)@pats)).cpu().numpy())
    return np.concatenate(out)

def train():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    data,meta=load('fit');mean,scale=normalization(data['X'])
    w=np.load(RUN/'fit/w.npy',mmap_mode='r')[:meta['rows']].copy();w=w/w.mean()
    dec,dm=load('december')
    gpu_data=None
    if device=='cuda':
        gx=torch.empty((meta['rows'],940),device=device,dtype=torch.float32)
        for j in range(0,meta['rows'],8192):
            gx[j:j+8192]=torch.from_numpy(transform(data['X'][j:j+8192],mean,scale,'full')).to(device)
        gpu_data=dict(x=gx,y=torch.tensor(np.array(data['y']),device=device,dtype=torch.float32),
                      m=torch.tensor(np.array(data['m']),device=device,dtype=torch.bool),w=torch.from_numpy(w).to(device))
        log(phase='GPU_DATA_READY',rows=meta['rows'],allocated_GB=torch.cuda.memory_allocated()/1e9)
    env=dict(torch=torch.__version__,numpy=np.__version__,device=device,gpu=torch.cuda.get_device_name() if device=='cuda' else None,threads=4)
    write(OUT/'environment.json',env)
    for seed in SEEDS:
        for variant in ['full','no_occupancy_profile']:
            folder=OUT/f'{variant}_{seed}';folder.mkdir(exist_ok=True)
            if (folder/'complete.json').exists():continue
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            if device=='cuda':torch.cuda.manual_seed_all(seed)
            model=TwoStage().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            first=0;trace=[];rng=np.random.default_rng(seed)
            checkpoint=folder/'checkpoint.pt'
            if checkpoint.exists():
                ck=torch.load(checkpoint,map_location=device,weights_only=False)
                model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);first=ck['epoch'];trace=ck['trace'];rng.bit_generator.state=ck['numpy_rng'];torch.set_rng_state(ck['torch_rng'].cpu())
                if device=='cuda':torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
            for epoch in range(first,EPOCHS):
                model.train();permutation=rng.permutation(meta['rows']);total=0.;num=0
                for j in range(0,len(permutation),4096):
                    ids=permutation[j:j+4096]
                    if gpu_data is not None:
                        gi=torch.as_tensor(ids,device=device)
                        x=gpu_data['x'][gi]
                        if variant=='no_occupancy_profile':x[:,occupancy_cols()]=0
                        y=gpu_data['y'][gi];m=gpu_data['m'][gi];weight=gpu_data['w'][gi]
                    else:
                        x=torch.from_numpy(transform(data['X'][ids],mean,scale,variant)).to(device)
                        y=torch.from_numpy(np.asarray(data['y'][ids],dtype='float32')).to(device)
                        m=torch.from_numpy(np.asarray(data['m'][ids],dtype='bool')).to(device)
                        weight=torch.from_numpy(w[ids]).to(device)
                    opt.zero_grad(set_to_none=True);a,b=model(x)
                    loss=(partial_loss(a,b,y,m)*weight).mean()
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                    total+=float(loss.detach())*len(ids);num+=len(ids)
                row=dict(epoch=epoch+1,loss=total/num);trace.append(row)
                ck=dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,trace=trace,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state() if device=='cuda' else None)
                torch.save(ck,checkpoint)
                if (epoch+1)%5==0:
                    pred=predict(model,dec['X'],mean,scale,variant,device)
                    row['December_onset_at_05']=score_population(pred,dec,dm,.5)['onset_IoU']
                    log(phase='training',variant=variant,seed=seed,**row)
                write(folder/'trace.json',trace)
            torch.save(dict(model=model.state_dict(),mean=mean,scale=scale,variant=variant,seed=seed,epochs=EPOCHS),folder/'model.pt')
            write(folder/'complete.json',dict(trained=True,epochs=EPOCHS,rows=meta['rows'],fit_windows=meta['fit_windows'],parameters=sum(p.numel() for p in model.parameters())))
    log(phase='TRAIN_COMPLETE')

FAMILY={'D7_I10':'I10','D7_I210':'I210','D7_I405':'I405','D12_I5':'I5'}
def score_population(p,data,meta,threshold=.5,positive_only=True):
    per={};rows=[]
    errors={k:0 for k in ['TP_cells','FP_cells','FN_cells','missed_support_FN_cells','timing_FN_cells','positive_link_timing_FP_cells','complete_negative_link_FP_cells','unknown_event_link_FP_cells','known_positive_links','missed_positive_links']}
    for e in meta['windows']:
        if positive_only and not e['positive']:continue
        a,b=e['start'],e['end'];m=np.asarray(data['m'][a:b]);y=np.asarray(data['y'][a:b])>0;q=p[a:b]>=threshold
        union=int(((y|q)&m).sum());inter=int(((y&q)&m).sum());v=inter/union if union else (1. if m.any() else 0.)
        per.setdefault(e['panel'],[]).append(v);rows.append(dict(panel=e['panel'],origin=e['origin'],iou=v))
        positive=(y&m).any(1);negative=m.all(1)&~positive;support=q.any(1)
        fn=y&~q&m;fp=~y&q&m
        errors['TP_cells']+=inter;errors['FP_cells']+=int(fp.sum());errors['FN_cells']+=int(fn.sum())
        errors['missed_support_FN_cells']+=int(fn[~support].sum());errors['timing_FN_cells']+=int(fn[support].sum())
        errors['positive_link_timing_FP_cells']+=int(fp[positive].sum());errors['complete_negative_link_FP_cells']+=int(fp[negative].sum());errors['unknown_event_link_FP_cells']+=int(fp[~(positive|negative)].sum())
        errors['known_positive_links']+=int(positive.sum());errors['missed_positive_links']+=int((positive&~support).sum())
    road={k:float(np.mean(v)) for k,v in per.items()};families={}
    for roadname,v in road.items():families.setdefault(FAMILY[roadname.rsplit('_',1)[0]],[]).append(v)
    return dict(onset_IoU=float(np.mean([np.mean(v) for v in families.values()])),per_road=road,windows=len(rows),window_rows=rows,cell_counts=errors)

def bootstrap(a,b,seed=42,reps=2000):
    # Cluster by calendar day across roads, preserving correlated views within day.
    ra=a['window_rows'];rb=b['window_rows'];assert [(r['panel'],r['origin']) for r in ra]==[(r['panel'],r['origin']) for r in rb]
    days=sorted({r['origin'][:10] for r in ra});rng=np.random.default_rng(seed);deltas=[]
    for _ in range(reps):
        picked=rng.choice(days,len(days));mult={d:int(sum(picked==d)) for d in days};roads={}
        for x,y in zip(ra,rb):
            count=mult[x['origin'][:10]]
            if count:roads.setdefault(x['panel'],[]).extend([x['iou']-y['iou']]*count)
        fam={}
        for p,v in roads.items():fam.setdefault(FAMILY[p.rsplit('_',1)[0]],[]).append(np.mean(v))
        if len(roads)==8:deltas.append(np.mean([np.mean(v) for v in fam.values()]))
    return dict(delta=a['onset_IoU']-b['onset_IoU'],day_cluster_95CI=np.quantile(deltas,[.025,.975]).tolist(),valid_resamples=len(deltas),days=len(days))

def average_seed_score(scores):
    result=copy.deepcopy(scores[0])
    for j,row in enumerate(result['window_rows']):
        row['iou']=float(np.mean([s['window_rows'][j]['iou'] for s in scores]))
    result['onset_IoU']=float(np.mean([s['onset_IoU'] for s in scores]))
    result['per_road']={p:float(np.mean([s['per_road'][p] for s in scores])) for p in result['per_road']}
    return result

def evaluate():
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    mean,scale=normalization(load('fit')[0]['X']);reports={};calibration={};stored={}
    while not (RUN/'january/complete.json').exists():time.sleep(1)
    jan,jm=load('january')
    for seed in SEEDS:
        for variant in ['full','no_occupancy_profile']:
            key=f'{variant}_{seed}';folder=OUT/key
            while not (folder/'complete.json').exists():time.sleep(1)
            ck=torch.load(folder/'model.pt',map_location=device,weights_only=False);model=TwoStage().to(device);model.load_state_dict(ck['model'])
            p=predict(model,jan['X'],mean,scale,variant,device)
            grid=[float(round(t,2)) for t in np.arange(.05,.901,.05)]
            scores=[score_population(p,jan,jm,t)['onset_IoU'] for t in grid]
            threshold=grid[int(np.argmax(scores))]
            extended_grid=grid+[.925,.95,.975,.99,.995,.999]
            extended_scores=scores+[score_population(p,jan,jm,t)['onset_IoU'] for t in extended_grid[len(grid):]]
            extended_threshold=extended_grid[int(np.argmax(extended_scores))]
            calibration[key]=dict(threshold=threshold,grid=grid,scores=scores,extended_threshold=extended_threshold,extended_grid=extended_grid,extended_scores=extended_scores)
            for pop in ['december','new_phase_population','old410']:
                while not (RUN/pop/'complete.json').exists():time.sleep(1)
                data,meta=load(pop);pred=predict(model,data['X'],mean,scale,variant,device)
                np.save(folder/(pop+'-probability.npy'),pred)
                result={name:score_population(pred,data,meta,t) for name,t in [('fixed05',.5),('January_calibrated',threshold),('January_extended',extended_threshold)]}
                result['all_onset_windows_fixed05']=score_population(pred,data,meta,.5,False)
                reports.setdefault(pop,{})[key]=result;stored[(pop,key)]=pred
            log(phase='evaluation',variant=variant,seed=seed,threshold=threshold,extended_threshold=extended_threshold)
    write(OUT/'calibration.json',calibration)
    comparisons={}
    for pop in reports:
        data,meta=load(pop);parent=np.load(RUN/pop/'parent.npy')
        reports[pop]['R127']=dict(fixed05=score_population(parent,data,meta,.5),all_onset_windows_fixed05=score_population(parent,data,meta,.5,False))
        summary={}
        for variant in ['full','no_occupancy_profile']:
            summary[variant]={mode:dict(mean=float(np.mean([reports[pop][f'{variant}_{s}'][mode]['onset_IoU'] for s in SEEDS])),std=float(np.std([reports[pop][f'{variant}_{s}'][mode]['onset_IoU'] for s in SEEDS],ddof=1))) for mode in ['fixed05','January_calibrated','January_extended']}
        summary['R127_onset_IoU']=reports[pop]['R127']['fixed05']['onset_IoU']
        for mode in ['fixed05','January_calibrated','January_extended']:
            full=average_seed_score([reports[pop][f'full_{s}'][mode] for s in SEEDS])
            removed=average_seed_score([reports[pop][f'no_occupancy_profile_{s}'][mode] for s in SEEDS])
            summary[f'{mode}_mean_model_full_vs_R127']=bootstrap(full,reports[pop]['R127']['fixed05'])
            summary[f'{mode}_mean_model_full_vs_no_occupancy_profile']=bootstrap(full,removed)
        # Ensemble uses the same three seeds in both branches. Threshold .5 is fixed.
        ensemble={}
        for variant in ['full','no_occupancy_profile']:
            pred=np.mean([stored[(pop,f'{variant}_{s}')] for s in SEEDS],axis=0)
            ensemble[variant]=score_population(pred,data,meta,.5)
            np.save(OUT/f'{pop}-{variant}-ensemble.npy',pred)
        summary['ensemble_full_vs_R127']=bootstrap(ensemble['full'],reports[pop]['R127']['fixed05'])
        summary['ensemble_full_vs_no_occupancy_profile']=bootstrap(ensemble['full'],ensemble['no_occupancy_profile'])
        comparisons[pop]=summary
    write(OUT/'results.json',dict(summary=comparisons,reports=reports,official_S_queue=None,February_repeated_development_not_blind=True))
    log(phase='EVALUATION_COMPLETE',summary=comparisons)

def intervention():
    """Supplementary frozen-R127 reliance test; not a retrained ablation."""
    model,sample,r2,features,audit=backend()
    import position_mode_core
    original=position_mode_core.extras
    columns=np.array([t*24+16+k for t in range(13) for k in range(8)])
    def removed(*args,**kwargs):
        z=original(*args,**kwargs).copy();assert z.shape[1]==359
        z[:,columns]=0
        return z
    bindings=[]
    for module in list(sys.modules.values()):
        if module is None:continue
        for key,value in list(vars(module).items()):
            if value is original:bindings.append((module,key))
    reports={}
    for pop,file in [('december','december-development.json'),('new_phase_population','february-new_phase_population.json'),('old410','february-old410.json')]:
        while not (RUN/pop/'complete.json').exists():time.sleep(1)
        data,meta=load(pop);counter=[];cursor=0;checked_roads=set();parent=np.load(RUN/pop/'parent.npy',mmap_mode='r')
        for e in entries(file):
            if e['condition']!=0:continue
            s=sample(e);n=s['x'].shape[1]
            if e['panel'] not in checked_roads:
                np.testing.assert_array_equal(model.probability(s).T,parent[cursor:cursor+n]);checked_roads.add(e['panel'])
            try:
                for module,key in bindings:setattr(module,key,removed)
                altered=model.probability(s).T
            finally:
                for module,key in bindings:setattr(module,key,original)
            counter.append(altered);cursor+=n
        p=np.concatenate(counter);np.save(OUT/f'{pop}-R127-removed-occupancy-profile.npy',p)
        base=score_population(parent,data,meta,.5)
        changed=score_population(p,data,meta,.5)
        reports[pop]=dict(R127=base,removed=changed,full_vs_removed=bootstrap(base,changed))
        write(OUT/'R127_occupancy_intervention.json',dict(reports=reports,complete=False,interpretation='Frozen-weight input reliance; may be out of distribution; complement with retrained paired ablation'))
        log(phase='R127_occupancy_intervention',population=pop,full=base['onset_IoU'],removed=changed['onset_IoU'])
    write(OUT/'R127_occupancy_intervention.json',dict(reports=reports,interpretation='Frozen-weight input reliance; may be out of distribution; complement with retrained paired ablation',replaced_bindings=[m.__name__+'.'+k for m,k in bindings]))

if __name__=='__main__':
    START=time.time();ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','prepare-evaluation','prepare-december','train','evaluate','intervention']);args=ap.parse_args()
    {'prepare':prepare,'prepare-evaluation':prepare_evaluation,'prepare-december':lambda:prepare_evaluation(True),'train':train,'evaluate':evaluate,'intervention':intervention}[args.action]()
