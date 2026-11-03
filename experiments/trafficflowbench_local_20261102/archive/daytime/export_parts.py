"""Read-only baseline inputs, complete candidate state parts, independent readback."""
import common as d
import pandas as pd,numpy as np,importlib.util

def run(package,dest):
 spec=importlib.util.spec_from_file_location('portable_state_export',package/'apply_residual.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);policies=__import__('json').loads((package/'policy.json').read_text())['policy'];acceptance=[];reader=pd.read_parquet
 def masked_only(path,*args,**kwargs):
  if '/mainline_states/' in str(path).replace('\\','/'):raise RuntimeError('Truth read during deployment inference')
  return reader(path,*args,**kwargs)
 pd.read_parquet=masked_only
 try:
  for split in ['validation','private']:
   for panel in policies:
    d.check_deadline();d.state(status='running',stage='portable_full_state_export',package=package.name,panel=panel,split=split,next='Complete state parts and independent readback; no complete Kaggle submission, no upload')
    baseline=d.ROOT/'outputs/overnight/scored_context_parts/baseline'/split/(panel+'.csv');f=pd.read_csv(baseline,dtype={k:str for k in d.KEYS});pp,mm=module.load_panel(panel);result=pd.concat([module.correct(d.DATA,panel,date,p,split,pp,mm) for date,p in f.groupby(f.timestamp.str[:10],sort=False)],ignore_index=True);assert len(result)==len(f) and not result.duplicated(d.KEYS).any();result=f[d.KEYS].merge(result,on=d.KEYS,how='left',validate='one_to_one',sort=False);assert np.isfinite(result[['speed_kmh','flow_vph']].to_numpy()).all() and (result[['speed_kmh','flow_vph']].to_numpy()>=0).all();target=pd.read_csv(d.DATA/'task1'/panel/split/'sample_submission_state.csv',usecols=d.KEYS,dtype=str);assert len(target)==len(result) and len(target.merge(result[d.KEYS],on=d.KEYS,validate='one_to_one'))==len(target);path=dest/split/(panel+'.csv');path.parent.mkdir(parents=True,exist_ok=True);result.to_csv(path,index=False);back=pd.read_csv(path,dtype={k:str for k in d.KEYS});assert back[d.KEYS].equals(result[d.KEYS]) and np.allclose(back[['speed_kmh','flow_vph']],result[['speed_kmh','flow_vph']],atol=1e-10,rtol=1e-10);acceptance.append(dict(panel=panel,split=split,rows=len(result),candidate_sha256=d.sha(path),baseline_path=str(baseline),baseline_sha256=d.sha(baseline),finite_nonnegative=True,template_keys_match=True,independent_readback=True));d.save(dest/'acceptance.json',dict(files=acceptance,total_rows=sum(r['rows'] for r in acceptance),package_manifest_sha256=d.sha(package/'manifest.json'),official_score=None,no_kaggle_upload=True));print('STATE EXPORT',package.name,panel,split,len(result),flush=True)
 finally:pd.read_parquet=reader
 assert len(acceptance)==20 and sum(r['rows'] for r in acceptance)==6735795
