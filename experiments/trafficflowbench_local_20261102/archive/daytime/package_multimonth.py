"""Package frozen multi-month weights and generate local state parts only."""
import common as d
import augmentation_fit as a
import json,shutil,importlib.util,msvcrt,traceback
import numpy as np,pandas as pd,lightgbm as lgb
O=d.OUT/'multimonth_candidate';S=d.OUT/'multimonth_fit';PARTS=d.OUT/'scored_multimonth_parts'

def main():
 source=d.ROOT/'outputs/overnight/context_candidate';policy=json.loads((S/'frozen_policy.json').read_text())['policy'];portable={};provenance={}
 for name in ['apply_residual.py','residual_features.py','context_features.py']:
  shutil.copy2(source/name,O/name);assert d.sha(source/name)==d.sha(O/name)
 for panel,p in policy.items():
  if p['recipe']=='context':root=source/'models'/panel;features=json.loads((source/'policy.json').read_text())['policy'][panel]['features']
  else:root=S/'models'/panel if p['recipe']=='december_january' else a.O/'models'/panel/'balanced_original';features=61
  dest=O/'models'/panel;dest.mkdir(parents=True,exist_ok=True);portable[panel]=dict(recipe=p['recipe'],alpha=p['alpha'],features=features)
  for name in ['speed.txt','flow_per_lane.txt']:
   shutil.copy2(root/name,dest/name);assert d.sha(root/name)==d.sha(dest/name);provenance[str((dest/name).relative_to(O))]=dict(source=str(root/name),sha256=d.sha(dest/name))
 d.save(O/'policy.json',dict(policy=portable,official_score=None,development_scope='Exposed February evaluation; not official or whole-system blind',training_dates=json.loads((S/'protocol.json').read_text())['train_dates']))
 d.save(O/'provenance.json',dict(files=provenance,frozen_selection_sha256=d.sha(S/'frozen_policy.json'),packager_sha256=d.sha(__file__)))
 (O/'README.txt').write_text('Local multi-month residual candidate. No Kaggle upload.\nJanuary-only selection; expanded routes train Dec14-31 and Jan05-20. Other routes retain frozen context.\nIndependent state parts are Task1/Task3 inputs only; Task2 and Task4 unchanged, no complete submission created.\nFebruary is exposed development evaluation, not whole-system blind or official improvement.\nOriginal February state improves, but auxiliary point-missing accumulation changes can worsen. Keep the official 0.81135 baseline and all earlier candidates.\nRun apply_residual.py --help. Masked offline Task1 features may include future observations; do not apply to Task2 forecasting.\n',encoding='utf8')
 spec=importlib.util.spec_from_file_location('day_multimonth_apply',O/'apply_residual.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);checks=[]
 for panel,p in portable.items():
  d.check_deadline();d.state(status='running',stage='multimonth_package_verification',panel=panel,next='Verify portable inference then generate complete local state parts; no complete submission or upload')
  it,X=a.original(panel,'2031-01-26');f=it['frame'];pp,mm=module.load_panel(panel);expected=a.correct(f,X,mm,pp['alpha']);actual=module.correct(d.DATA,panel,'2031-01-26',f,'train',pp,mm);delta=float(np.max(np.abs(actual[['speed_kmh','flow_vph']].to_numpy()-expected)));assert delta<1e-9;checks.append(dict(panel=panel,max_absolute_delta=delta,rows=len(actual)))
 d.save(O/'inference_equivalence.json',checks);d.save(O/'manifest.json',dict(files={str(p.relative_to(O)):d.sha(p) for p in O.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json'}));shutil.make_archive(str(d.OUT/'multimonth_candidate'),'zip',O);d.save(d.OUT/'multimonth_candidate_archive.json',dict(sha256=d.sha(d.OUT/'multimonth_candidate.zip'),bytes=(d.OUT/'multimonth_candidate.zip').stat().st_size))
 acceptance=[];reader=pd.read_parquet
 def masked_only(path,*args,**kwargs):
  if '/mainline_states/' in str(path).replace('\\','/'):raise RuntimeError('Truth read during deployment inference')
  return reader(path,*args,**kwargs)
 pd.read_parquet=masked_only
 try:
  for split in ['validation','private']:
   for panel in portable:
    d.check_deadline();d.state(status='running',stage='multimonth_full_state_parts',panel=panel,split=split,next='Generate and verify local keyed state files; never upload or alter baseline')
    baseline=d.ROOT/'outputs/overnight/scored_context_parts/baseline'/split/(panel+'.csv');f=pd.read_csv(baseline,dtype={k:str for k in d.KEYS});pp,mm=module.load_panel(panel);result=pd.concat([module.correct(d.DATA,panel,date,p,split,pp,mm) for date,p in f.groupby(f.timestamp.str[:10],sort=False)],ignore_index=True);assert len(result)==len(f) and not result.duplicated(d.KEYS).any();result=f[d.KEYS].merge(result,on=d.KEYS,validate='one_to_one',how='left',sort=False);assert np.isfinite(result[['speed_kmh','flow_vph']].to_numpy()).all() and (result[['speed_kmh','flow_vph']].to_numpy()>=0).all();target=pd.read_csv(d.DATA/'task1'/panel/split/'sample_submission_state.csv',usecols=d.KEYS,dtype=str);assert len(target)==len(result) and len(target.merge(result[d.KEYS],on=d.KEYS,validate='one_to_one'))==len(target);path=PARTS/split/(panel+'.csv');path.parent.mkdir(parents=True,exist_ok=True);result.to_csv(path,index=False);readback=pd.read_csv(path,dtype={k:str for k in d.KEYS});assert readback[d.KEYS].equals(result[d.KEYS]) and np.allclose(readback[['speed_kmh','flow_vph']],result[['speed_kmh','flow_vph']],atol=1e-10,rtol=1e-10);acceptance.append(dict(panel=panel,split=split,rows=len(result),candidate_sha256=d.sha(path),baseline_path=str(baseline),baseline_sha256=d.sha(baseline),finite_nonnegative=True,template_keys_match=True,independent_readback=True));d.save(PARTS/'acceptance.json',dict(files=acceptance,total_rows=sum(r['rows'] for r in acceptance),package_sha256=d.sha(d.OUT/'multimonth_candidate.zip'),official_score=None,no_kaggle_upload=True));print('MULTIMONTH PART',panel,split,len(result),flush=True)
 finally:pd.read_parquet=reader
 assert sum(r['rows'] for r in acceptance)==6735795
 d.state(status='stage_complete',stage='multimonth_package_complete',next='Portable package and all20 state parts checked; retain sparse-mask accumulation caveat; investigate new hypothesis without overwriting any candidate.')

if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True);lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:main()
 except Exception as e:d.state(status='failed_or_cutoff',stage='package_multimonth',error=repr(e),next='Repair export/integrity issue without changing frozen models or input baselines');traceback.print_exc();raise
 finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
