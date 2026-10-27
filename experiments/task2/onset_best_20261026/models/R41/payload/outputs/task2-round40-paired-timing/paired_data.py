"""Original conditional rows plus balanced, matched forecast-origin trios."""
import json,hashlib
from pathlib import Path
import numpy as np
from run_round2 import sample
from timing_core import features,unseen_queue
from task2q.cache import atomic_json
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1];AUDIT=ROOT.parent/'task2-round39-paired-phase-calendar';WORK=PROJECT/'work/round40-paired-timing';WORK.mkdir(exist_ok=True,parents=True)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def pair_data(period,progress):
    folder=WORK/('paired-'+period);folder.mkdir(exist_ok=True);done=folder/'complete.json'
    if done.exists():return {k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in ['X','y','w','panel']},read(done)
    rows=[r for r in read(AUDIT/'paired-targets.json') if r['period']==period];groups={}
    for r in rows:groups.setdefault((r['panel'],r['origin']),[]).append(r)
    chunks=[];ys=[];ws=[];ps=[]
    for i,rs in enumerate(groups.values()):
        s=sample(rs[0]);idx=np.asarray([r['target_link_index'] for r in rs]);yy=np.asarray([r['first_step'] for r in rs],dtype='int32');assert unseen_queue(s)[idx].all() and s['m'][:,idx].all() and s['y'][:,idx].any(0).all();assert np.array_equal(np.argmax(s['y'][:,idx],axis=0),yy)
        chunks.append(features(s)[idx]);ys.append(yy);ws.append(np.asarray([r['raw_weight'] for r in rs],dtype='float32'));ps.extend([s['panel']]*len(rs))
        if (i+1)%256==0:progress('building_paired_features',period=period,origins=i+1,total=len(groups))
    data=dict(X=np.concatenate(chunks),y=np.concatenate(ys),w=np.concatenate(ws),panel=np.asarray(ps));ng=len({r['label_side_group'] for r in rows});assert np.isclose(data['w'].sum(),ng,rtol=1e-6)
    meta=dict(period=period,rows=len(data['y']),features=557,groups=ng,forecast_origins=len(groups),overlapping_original_views=int(sum(r['already_in_original'] for r in rows)),raw_group_mass1=True,phase_counts={str(k):int(sum(r['phase']==k for r in rows)) for k in range(3)},future_metadata_not_in_features=True)
    assert len(set(meta['phase_counts'].values()))==1
    for k,v in data.items():np.save(folder/(k+'.npy'),v)
    atomic_json(done,meta);return data,meta
def build(mode,progress):
    folder=WORK/mode;folder.mkdir(exist_ok=True);done=folder/'complete.json'
    if done.exists():return {k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in ['X','y','w','panel']},read(done)
    early=mode=='early';oldfolder=PROJECT/('work/round33-conditional-start' if early else 'work/round34-conditional-start-full');oldroot=ROOT.parent/('task2-round33-conditional-start' if early else 'task2-round34-conditional-start-full');old={k:np.load(oldfolder/(k+'.npy'),mmap_mode='r') for k in ['X','y','w']};orig=[r for r in read(AUDIT/'original-conditional-targets.json') if not early or r['period']=='early'];assert np.array_equal(old['y'],np.asarray([r['first_step'] for r in orig]));oldwindows=len(read(oldroot/'conditional-target-index.json'));oldraw=np.asarray(old['w'])*oldwindows
    parts=[pair_data('early',progress)];
    if not early:parts.append(pair_data('held',progress))
    raw=np.concatenate([oldraw,*[d['w'] for d,m in parts]]);mass=float(raw.sum());data=dict(X=np.concatenate([old['X'],*[d['X'] for d,m in parts]]),y=np.concatenate([old['y'],*[d['y'] for d,m in parts]]),w=raw/mass,panel=np.concatenate([np.asarray([r['panel'] for r in orig]),*[d['panel'] for d,m in parts]]))
    n=len(old['y']);assert np.array_equal(data['X'][:n],old['X']) and np.array_equal(data['y'][:n],old['y']);assert (data['w']>0).all() and np.isclose(data['w'].sum(),1,atol=1e-6);assert np.allclose(data['w'][:n]/old['w'],oldwindows/mass,rtol=1e-6)
    meta=dict(mode=mode,rows=len(data['y']),original_rows=n,added_paired_views=sum(m['rows'] for d,m in parts),added_paired_groups=sum(m['groups'] for d,m in parts),original_windows=oldwindows,features=557,original_X_y_exact=True,original_relative_weights_preserved=True,original_mass=float(data['w'][:n].sum()),paired_mass=float(data['w'][n:].sum()),raw_mass=mass,paired_overlap_views=sum(m['overlapping_original_views'] for d,m in parts),labels_and_sampling_only_future_metadata=True,source_sha256={k:hashlib.sha256((oldfolder/(k+'.npy')).read_bytes()).hexdigest() for k in ['X','y','w']},loss_units='original conditional window raw mass1 plus each paired physical-observation group mass1; normalized; not unbiased official macroIoU')
    for k,v in data.items():np.save(folder/(k+'.npy'),v)
    atomic_json(done,meta);return data,meta
