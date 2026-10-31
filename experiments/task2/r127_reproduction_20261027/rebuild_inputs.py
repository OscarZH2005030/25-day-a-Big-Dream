"""Rebuild all 15 exact native-training arrays from public observations and frozen sampling/weight protocol."""
import argparse,time,gc
from pathlib import Path
import numpy as np
from repro_common import HERE,read,write,sha,setup
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,required=True);ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--output',type=Path,default=Path.home()/'tfb-r127-work/rebuilt-inputs');ap.add_argument('--work',type=Path,default=Path.home()/'tfb-r127-work/rebuild-cache');args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False);start=time.time();protocol=read(HERE/'training/raw-rebuild-recipe.json');manifest=read(HERE/'training/input-manifest.json');model,sample,r2,audit=setup(args.package,args.data_root,args.work,True,HERE/'indices/training-array-hashes.json',[args.output])
    from coverage_data import matrix
    from timing_core import features,unseen_queue
    from position_mode_core import extras
    bases={}
    for p,rec in protocol['basis'].items():
        path=HERE/rec['path'];assert sha(path)==rec['sha256']
        with np.load(path) as f:bases[p]={k:f[k] for k in ['position','random','supported']}
    files={n:rec for n,rec in manifest['files'].items() if n.endswith('.npy')}
    def create(name):
        rec=files[name];path=args.output/name;path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();return np.lib.format.open_memmap(path,mode='w+',dtype=rec['dtype'],shape=tuple(rec['shape']))
    for p,path in protocol['position_recipes'].items():
        recipe=read(HERE/path);prefix='position/'+p+'/';X=create(prefix+'X555.npy');pos=create(prefix+'position.npy');random=create(prefix+'random.npy');y=create(prefix+'y.npy');w=create(prefix+'w.npy');w[:]=np.asarray(recipe['weights'],dtype=w.dtype)
        for i,row in enumerate(recipe['windows']):
            s=sample(row['entry']);ids=np.asarray(row['selected_cells'],dtype=int);a,b=row['start'],row['end'];assert len(ids)==b-a;X[a:b]=matrix(s)[ids];n=s['x'].shape[1];pos[a:b]=extras(s,bases[p],'position')[ids%n];random[a:b]=extras(s,bases[p],'random')[ids%n];y[a:b]=s['y'].ravel()[ids]
            if (i+1)%100==0:print(p,i+1,'/',len(recipe['windows']),flush=True)
        for v in [X,pos,random,y,w]:v.flush()
        del X,pos,random,y,w;gc.collect()
        for name in files:
            if name.startswith(prefix):assert sha(args.output/name)==files[name]['sha256'],'Rebuilt array differs: '+name
    rows=read(HERE/protocol['conditional_rows']);groups={}
    for i,row in enumerate(rows):groups.setdefault((row['panel'],row['origin']),[]).append((i,row))
    pos=create('conditional/position.npy');random=create('conditional/random.npy');y=create('conditional/y.npy');w=create('conditional/w.npy');panel=create('conditional/panel.npy');w[:]=np.asarray(protocol['conditional_weights'],dtype=w.dtype)
    for i,rr in enumerate(groups.values()):
        s=sample(rr[0][1]);ids=np.array([r['target_link_index'] for j,r in rr]);ix=np.array([j for j,r in rr]);assert unseen_queue(s)[ids].all() and s['m'][:,ids].all() and s['y'][:,ids].any(0).all();base=features(s)
        for mode,output in [('position',pos),('random',random)]:output[ix]=np.column_stack([base,extras(s,bases[s['panel']],mode)]).astype('float32')[ids]
        first=s['y'][:,ids].argmax(0);assert np.array_equal(first,[r['first_step'] for j,r in rr]);y[ix]=first;panel[ix]=s['panel']
        if (i+1)%500==0:print('conditional',i+1,'/',len(groups),flush=True)
    for v in [pos,random,y,w,panel]:v.flush()
    del pos,random,y,w,panel;gc.collect();hashes={}
    for name,rec in files.items():assert sha(args.output/name)==rec['sha256'],'Rebuilt array differs: '+name;hashes[name]=rec['sha256']
    # Reference model text is optional evidence in the convenience ZIP, not a training input.
    write(args.output/'manifest.json',dict(manifest,files=files));result=dict(passed=True,arrays=len(hashes),byte_exact_array_hashes=hashes,seconds=time.time()-start,original_model_cache_IO_blocked=True,raw_day_arrays_reconstructed_and_verified=len(audit['generated_day_keys']),frozen_sampling_and_weights_provided_as_protocol=True,features_and_proxy_targets_rebuilt_from_raw_observations=True,official_S_queue=None);write(args.output/'rebuild-verification.json',result);print('PASS',len(hashes),'byte-identical training arrays')
if __name__=='__main__':main()
