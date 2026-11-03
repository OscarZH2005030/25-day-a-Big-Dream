"""Replay one frozen-weight day and compare to the original submitted cache."""
import argparse,importlib.util,json,os,sys
from pathlib import Path
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='4'
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');args=parser.parse_args()
device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
torch.set_num_threads(4)
bundle=ROOT/'project/experiments/task1/best_road_combination_20261026'
sys.path.insert(0,str(bundle))
spec=importlib.util.spec_from_file_location('portable',ROOT/'work/tfb_submit/complete/code/portable_entry.py');portable=importlib.util.module_from_spec(spec);spec.loader.exec_module(portable);portable.install_unicode_loader()
import predict_selected as pm
panel,date='D12_I405_N','2031-03-01'
expected=pd.read_parquet(ROOT/'work/tfb_submit/task1/submissions/validation'/f'{panel}_{date}.parquet')
keys=expected[pm.base.KEYS].copy();pm.keys_for=lambda *unused:keys.copy()
reader=pd.read_parquet
def guarded(path,*a,**kw):
    if '/mainline_states/' in str(path).replace('\\','/'):raise RuntimeError('Inference attempted truth access')
    return reader(path,*a,**kw)
pd.read_parquet=guarded
try:actual=pm.predict(panel,date,pm.load_routes()[panel],ROOT/'data/kaggle_public',ROOT/'local_reports/smoke_predictions',device)
finally:pd.read_parquet=reader
assert actual[pm.base.KEYS].equals(keys)
a=actual[['speed_kmh','flow_vph']].to_numpy();b=expected[['speed_kmh','flow_vph']].to_numpy()
np.testing.assert_allclose(a,b,rtol=1e-5,atol=5e-4)
report={'passed':True,'device':device,'panel':panel,'date':date,'rows':len(actual),'max_abs_difference_by_channel':abs(a-b).max(axis=0).tolist(),'comparison':'rtol=1e-5, atol=5e-4; no training or submission'}
(ROOT/'local_reports/smoke_inference.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))
