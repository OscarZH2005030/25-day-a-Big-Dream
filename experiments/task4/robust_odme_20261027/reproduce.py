"""中文说明：可迁移Task4复现入口；不提交Kaggle，不修改输入数据。"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def verify(release:Path):
 import numpy as np,pandas as pd
 manifest=json.loads((ROOT/'competition_input_manifest.json').read_text())
 for item in manifest:
  p=release/item['path']
  if not p.exists():raise FileNotFoundError(p)
  if hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Input hash mismatch: '+str(p))
 templates=pd.concat([pd.read_csv(p/s/'sample_submission_path_flow.csv',dtype=str) for s in ['validation','private'] for p in sorted((release/'task4').iterdir())],ignore_index=True)
 keys=['panel','departure_time','path_id','origin_zone','destination_zone'];target=templates[keys].sort_values(keys).reset_index(drop=True);checked=[]
 for p in (ROOT/'outputs').rglob('task4*.csv'):
  x=pd.read_csv(p,dtype=str)
  if not set(keys+['path_flow']).issubset(x.columns):continue
  if not x[keys].sort_values(keys).reset_index(drop=True).equals(target):raise ValueError('Template mismatch: '+str(p))
  v=x.path_flow.to_numpy(float)
  if not np.isfinite(v).all() or (v<0).any() or x.duplicated(keys[:3]).any():raise ValueError('Invalid flows: '+str(p))
  checked.append(p.relative_to(ROOT).as_posix())
 print(json.dumps({'input_files_hashed':len(manifest),'prediction_files_validated':len(checked),'rows_per_file':len(target),'path_truth_available':False},indent=2))
 return checked

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--release-root',type=Path,required=True,help='Official kaggle_public directory; read-only')
 parser.add_argument('--stage',choices=['verify','latest','baseline','external','scale','bounds','robust','joint','joint-infer','joint-dev','joint-dev-infer','all'],default='latest')
 parser.add_argument('--data-python',default=sys.executable,help='Optional Python with requirements-data.txt for XLSX extraction')
 args=parser.parse_args();release=args.release_root.expanduser().resolve()
 env={**os.environ,'TFB_RELEASE_ROOT':str(release),'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONUTF8':'1'}
 def run(path,data=False):
  print('RUN '+path,flush=True);subprocess.run([args.data_python if data else sys.executable,str(ROOT/path)],cwd=ROOT,env=env,check=True)
 if args.stage=='verify':verify(release);return
 if args.stage in ['baseline','all']:
  run('outputs/run_task4.py');run('outputs/validate_task4.py')
 if args.stage in ['external','all']:
  run('outputs/external_v3/acquire_real.py',True);run('outputs/external_v3/prepare_real.py',True)
  run('outputs/fivemin_v4/acquire_5min.py');run('outputs/fivemin_v4/audit_5min.py')
 if args.stage=='all':
  for p in ['improvement_v2/run_v2.py','external_v3/run_external.py','external_v3/run_hybrid.py','external_v3/validate_external.py','fivemin_v4/run_path_experiments.py','dev_v5/run_dev.py','dev_v5/run_stress.py','adaptive_v6/run_adaptive.py','adaptive_v6/run_fine_amplitude.py','magnitude_v7/run_magnitude.py','scale_shape_v8/run_scale_shape.py','scale_shape_v8/select_balanced.py']:
   run('outputs/'+p)
 if args.stage=='scale':run('outputs/scale_shape_v8/infer_balanced.py')
 if args.stage in ['bounds','robust','all']:run('outputs/robust_v9/run_feasible_audit.py')
 if args.stage in ['robust','all']:
  for p in ['run_multi_error.py','select_conservative.py','audit_conservative.py']:run('outputs/robust_v9/'+p)
 if args.stage in ['latest','robust','all']:run('outputs/robust_v9/infer_conservative.py')
 if args.stage in ['joint','all']:run('outputs/joint_v10/run_joint.py')
 if args.stage=='joint-infer':run('outputs/joint_v10/infer_joint.py')
 if args.stage in ['joint-dev','all']:
  run('outputs/joint_v11/run_joint_dev.py');run('outputs/joint_v11/audit_fresh_counts.py');run('outputs/joint_v11/infer_joint_dev.py')
 if args.stage=='joint-dev-infer':run('outputs/joint_v11/infer_joint_dev.py')
 verify(release)
if __name__=='__main__':main()
