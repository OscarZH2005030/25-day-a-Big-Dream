"""Retrain all three added native heads and their matched controls, frozen parents unchanged."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='4'
import argparse,json,time,gc
from pathlib import Path
import numpy as np
import lightgbm as lgb
from repro_common import HERE,read,write,sha
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--output',type=Path,default=Path.home()/'tfb-r127-work/retrained');ap.add_argument('--models',choices=['selected','controls','all'],default='all');args=ap.parse_args();root=args.inputs.resolve();manifest=read(root/'manifest.json');args.output.mkdir(parents=True,exist_ok=True)
    for name,rec in manifest['files'].items():assert sha(root/name)==rec['sha256'],'Training input changed: '+name
    records=[];start=time.time()
    for spec in manifest['models']:
        if args.models!='all' and spec['group']!=args.models:continue
        target=args.output/spec['file'];assert not target.exists(),'Refuse to overwrite model: '+str(target)
        parts=[np.load(root/name,mmap_mode='r') for name in spec['X_parts']];X=np.empty((len(parts[0]),sum(v.shape[1] for v in parts)),dtype='float32');cursor=0
        for v in parts:X[:,cursor:cursor+v.shape[1]]=v;cursor+=v.shape[1]
        y=np.load(root/spec['y']);w=np.array(np.load(root/spec['w']),copy=True);w/=w.mean();ds=lgb.Dataset(X,label=y,weight=w)
        def checkpoint(env):
            if (env.iteration+1)%100==0:env.model.save_model(str(target.with_suffix('.checkpoint.txt')));print(spec['file'],env.iteration+1,flush=True)
        model=lgb.train(spec['params'],ds,num_boost_round=spec['rounds'],callbacks=[checkpoint]);model.save_model(str(target));actual=sha(target);assert actual==spec['model_sha256'],'Retrained model differs: '+spec['file']
        records.append(dict(file=spec['file'],sha256=actual,trees=model.num_trees(),byte_exact=True));write(args.output/'verification.json',dict(passed=True,completed=records,seconds=time.time()-start,frozen_parent_chain_not_retrained=True,official_S_queue=None));del ds,X,y,w,parts,model;gc.collect()
    assert records;print(json.dumps(dict(passed=True,models=len(records),byte_exact=True)))
if __name__=='__main__':main()
