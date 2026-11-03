"""Portable masked-only inference for the fixed eight-road control checkpoints."""
import argparse,hashlib,json,sys,importlib.util
from pathlib import Path
import torch
import pandas as pd
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True,type=Path);ap.add_argument('--work',required=True,type=Path);ap.add_argument('--base-bundle',type=Path,default=HERE.parent/'task1/best_road_combination_20261026');ap.add_argument('--road',required=True);ap.add_argument('--date',required=True);ap.add_argument('--device',default='cpu');ap.add_argument('--reference',type=Path);a=ap.parse_args();a.work.mkdir(parents=True,exist_ok=True)
    bundle=a.base_bundle.resolve();sys.path.insert(0,str(bundle));spec=importlib.util.spec_from_file_location('published_selected',bundle/'predict_selected.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);routes=module.load_routes();manifest=json.loads((HERE/'model-manifest.json').read_text())
    if a.road in manifest:
        info=manifest[a.road];source=HERE/info['path'];assert sha(source)==info['sha256'];converted=a.work/(a.road+'-wrapped.pt');torch.save(dict(model=torch.load(source,map_location='cpu',weights_only=True)),converted);routes[a.road]['paths']['transformer']=str(converted.resolve());routes[a.road]['sha256']['transformer']=sha(converted)
    original=pd.read_parquet
    def guarded(path,*args,**kwargs):
        if '/mainline_states/' in str(path).replace('\\','/'):raise RuntimeError('Unmasked mainline truth read forbidden')
        return original(path,*args,**kwargs)
    pd.read_parquet=guarded;torch.set_num_threads(4)
    try:pred=module.predict(a.road,a.date,routes[a.road],a.data_root.resolve(),a.work,a.device)
    finally:pd.read_parquet=original
    dest=a.work/(a.road+'_'+a.date+'.csv');pred.to_csv(dest,index=False)
    if a.reference:
        ref=pd.read_parquet(a.reference) if a.reference.suffix=='.parquet' else pd.read_csv(a.reference);ref=ref[ref.timestamp.astype(str).str[:10]==a.date].reset_index(drop=True);keys=['panel','timestamp','station_id','link_id','mask_regime'];assert pred[keys].astype(str).reset_index(drop=True).equals(ref[keys].astype(str).reset_index(drop=True))
        import numpy as np
        np.testing.assert_allclose(pred[['speed_kmh','flow_vph']],ref[['speed_kmh','flow_vph']],rtol=1e-6,atol=1e-4)
    print(json.dumps(dict(output=str(dest),rows=len(pred),sha256=sha(dest),reference_checked=bool(a.reference),no_truth_reads=True,official_score=None)))
if __name__=='__main__':main()
