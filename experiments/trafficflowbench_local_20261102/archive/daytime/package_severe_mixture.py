"""Independent mixture package and complete local state artifact export."""
import common as d
import augmentation_fit as a
import severe_augmentation_fit as fit
import export_parts
import json,shutil,zipfile,importlib.util,msvcrt,traceback
import numpy as np,pandas as pd,lightgbm as lgb
O=d.OUT/'severe_mixture_candidate';PARENT=d.OUT/'multimonth_candidate'

def main():
 selected=json.loads((fit.O/'frozen_policy.json').read_text())['policy'];parentpolicy=json.loads((PARENT/'policy.json').read_text())['policy'];portable={};sources={}
 shutil.copy2(d.WORK/'mixture_apply.py',O/'apply_residual.py')
 for n in ['residual_features.py','context_features.py']:shutil.copy2(PARENT/n,O/n);assert d.sha(PARENT/n)==d.sha(O/n)
 for panel,p in selected.items():
  parent=parentpolicy[panel];portable[panel]=dict(gamma=p['gamma'],parent_alpha=p['parent_alpha'],parent_features=parent['features'])
  folders=[('parent',PARENT/'models'/panel)]
  if p['gamma']>0:folders.append(('augmented',fit.O/'models'/panel))
  for kind,root in folders:
   dest=O/'models'/panel/kind;dest.mkdir(parents=True,exist_ok=True)
   for n in ['speed.txt','flow_per_lane.txt']:
    shutil.copy2(root/n,dest/n);assert d.sha(root/n)==d.sha(dest/n);sources[str((dest/n).relative_to(O))]=dict(source=str(root/n),sha256=d.sha(dest/n))
 d.save(O/'policy.json',dict(policy=portable,scope='January-selected severe augmentation mixture; February exposed development, not official or whole-system blind',official_score=None))
 d.save(O/'provenance.json',dict(files=sources,selected_policy_sha256=d.sha(fit.O/'frozen_policy.json'),training_protocol_sha256=d.sha(fit.O/'protocol.json'),prepared_cases_sha256=d.sha(fit.O/'prepared_cases.csv'),source_sha256=d.sha(__file__)))
 (O/'README.txt').write_text('Local severe-mask mixture candidate. No Kaggle upload.\nParent is the frozen December/January residual candidate. New models use original50%/station15-augmentation25%/block30-augmentation25% public training weights.\nJanuary alone selects per-road gamma0/.25/.5/1, with original-state and per-condition accumulation-error guards.\nInference clips parent and augmented predictions independently, then mixes; this exact order is required. All features use public masked offline Task1 observations. Do not use future-observation features for Task2.\nFebruary development state and accumulation means improved, as did two new-seed severe artificial conditions. Individual roads can worsen. These are adaptively reused exposed public data, not official Task3 or leaderboard improvement.\nStation15 hides eligible observable cells; noneligible signals may remain, so it is an outage sensitivity scenario rather than complete signal removal.\nPreserve original official0.81135 baseline, all earlier candidates, and two remaining Kaggle opportunities. State files are not complete submissions; no Task2/Task4 merge or upload.\n',encoding='utf8')
 spec=importlib.util.spec_from_file_location('verified_mixture_apply',O/'apply_residual.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);checks=[]
 for panel,p in selected.items():
  d.check_deadline();d.state(status='running',stage='severe_mixture_package_verification',panel=panel,next='Verify independent mixture application then export complete state parts locally')
  it,X=a.original(panel,'2031-01-26');f=it['frame'];parent=[lgb.Booster(model_file=str(PARENT/'models'/panel/n)) for n in ['speed.txt','flow_per_lane.txt']];new=[lgb.Booster(model_file=str(fit.O/'models'/panel/n)) for n in ['speed.txt','flow_per_lane.txt']];expected=fit.mixture(f,X,parent,new,p['parent_alpha'],p['gamma']);policy,models=module.load_panel(panel);actual=module.correct(d.DATA,panel,'2031-01-26',f,'train',policy,models);delta=float(np.max(np.abs(actual[['speed_kmh','flow_vph']].to_numpy()-expected)));assert delta<1e-9
  polluted=f.assign(true_speed_kmh=999999.,future_label=-999999.);other=module.correct(d.DATA,panel,'2031-01-26',polluted,'train',policy,models);assert np.array_equal(actual[['speed_kmh','flow_vph']],other[['speed_kmh','flow_vph']])
  try:module.correct(d.DATA,panel,'2031-01-26',pd.concat([f,f.iloc[:1]],ignore_index=True),'train',policy,models);raise AssertionError('Duplicate keys accepted')
  except ValueError:pass
  shuffled=f.sample(frac=1,random_state=2026110284);reordered=module.correct(d.DATA,panel,'2031-01-26',shuffled,'train',policy,models);aligned=actual[d.KEYS].merge(reordered,on=d.KEYS,validate='one_to_one',sort=False);assert np.allclose(aligned[['speed_kmh','flow_vph']],actual[['speed_kmh','flow_vph']],atol=1e-9,rtol=0)
  checks.append(dict(panel=panel,max_absolute_delta=delta,rows=len(actual),extra_truth_columns_ignored=True,duplicate_keys_rejected=True,row_order_invariant=True))
 d.save(O/'inference_equivalence.json',checks);files={str(p.relative_to(O)):d.sha(p) for p in O.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json'};d.save(O/'manifest.json',dict(files=files))
 with zipfile.ZipFile(d.OUT/'severe_mixture_candidate.zip','w',zipfile.ZIP_DEFLATED) as archive:
  for name in list(files)+['manifest.json']:archive.write(O/name,name)
 d.save(d.OUT/'severe_mixture_archive.json',dict(sha256=d.sha(d.OUT/'severe_mixture_candidate.zip'),bytes=(d.OUT/'severe_mixture_candidate.zip').stat().st_size))
 export_parts.run(O,d.OUT/'scored_severe_mixture_parts');d.state(status='stage_complete',stage='severe_mixture_package_complete',next='Independent mixture and all20 local state files verified. Preserve all previous candidates; next assess milder masks without retuning and retain negative roads; no uploads.')

if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True);lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:main()
 except Exception as e:d.state(status='failed_or_cutoff',stage='package_severe_mixture',error=repr(e),next='Repair independent inference/export discrepancy without changing January choices or original inputs');traceback.print_exc();raise
 finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
