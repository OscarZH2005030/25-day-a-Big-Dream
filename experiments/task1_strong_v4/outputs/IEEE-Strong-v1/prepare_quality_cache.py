"""Append observed occupancy/quality, preserving all previous 48 channels."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from quality_features import QUALITY_FEATURES,quality_features

def main(args):
    metadata=json.loads((args.base/'metadata.json').read_text())
    metadata=copy.deepcopy(metadata);metadata['features']+=QUALITY_FEATURES
    metadata['feature_revision']='48 existing channels plus 16 occupancy and sensor-quality channels'
    args.out.mkdir(parents=True,exist_ok=True)
    for pm in metadata['panels']:
        panel=pm['panel'];dest=args.out/panel;dest.mkdir(exist_ok=True)
        pm['features']=len(metadata['features'])
        if (dest/'COMPLETE.json').exists():continue
        source=args.base/panel
        base=np.load(source/'features.npy',mmap_mode='r')
        target=np.load(Path(pm['old_cache'])/'target.npy',mmap_mode='r')
        graph=dict(np.load(source/'graph.npz'))
        files={f.stem[-10:].replace('_','-'):f for f in
            (args.release/'corridors'/panel/'train/mainline_states_masked').rglob('*.parquet')}
        result=np.lib.format.open_memmap(dest/'features.npy',mode='w+',dtype=np.float16,
            shape=(273,len(metadata['features']),288,pm['nodes']))
        for d,date in enumerate(pm['dates']):
            frame=pd.read_parquet(files[date],columns=['timestamp','link_id','speed_kmh','flow_vph',
                'pct_observed','occupancy','is_imputed','is_observed'])
            extra=quality_features(frame,pm['links'],graph)
            assert not extra[1][target[d]].any(),'Hidden target occupancy was exposed'
            result[d,:48]=base[d];result[d,48:]=extra
            if d%100==0:print(json.dumps(dict(panel=panel,day=d)),flush=True)
        result.flush()
        for name in ['graph.npz','historical-profile.npz']:shutil.copyfile(source/name,dest/name)
        (dest/'metadata.json').write_text(json.dumps(pm,indent=2))
        (dest/'COMPLETE.json').write_text(json.dumps(dict(status='complete',features=64)))
        print(json.dumps(dict(panel=panel,status='complete')),flush=True)
    (args.out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print('All 64-feature panels complete.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--release',type=Path,required=True)
    main(p.parse_args())
