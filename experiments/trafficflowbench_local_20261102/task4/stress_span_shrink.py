"""Reproduce the frozen artificial span-smoothing stress, without old workspace."""
import argparse,hashlib,importlib.util,json,os,shutil,sys,types
from pathlib import Path
from datetime import datetime,timezone
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()

def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf8')

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--release-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();out=args.output.resolve();work=out/'work';work.mkdir(parents=True,exist_ok=True);here=Path(__file__).resolve().parents[1];archive=here/'archive/daytime'
 common=types.ModuleType('common');common.DATA=args.release_root.resolve();common.WORK=work;common.ROOT=work;common.OUT=out;common.sha=sha;common.save=save;common.check_deadline=lambda:None;common.state=lambda **kw:save(work/'STATE.json',dict(utc=datetime.now(timezone.utc).isoformat(),**kw));common.night=types.SimpleNamespace(predictor=types.SimpleNamespace(load_routes=lambda:{p.name:None for p in sorted((common.DATA/'task4').iterdir()) if p.is_dir()}));sys.modules['common']=common
 shutil.copy2(archive/'task4_prior_structure.py',work/'task4_prior_structure.py');load('task4_prior_structure',archive/'task4_prior_structure.py')
 covariance=out/'task4_covariance_audit/full_train_frozen_models.json';covariance.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(here/'results/task4_covariance_audit/full_train_frozen_models.json',covariance)
 baseline=work/'work/review-repo/experiments/task4/robust_odme_20261027/outputs/task4_path_flow.csv';baseline.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(here/'task4/frozen_reference_path_flow.csv',baseline)
 module=load('registered_span_shrink_stress',archive/'task4_span_shrink_stress.py');module.O=out/'stress';module.O.mkdir(parents=True,exist_ok=True);module.main();save(out/'portable_entrypoint.json',dict(entrypoint_sha256=sha(__file__),source_sha256=sha(archive/'task4_span_shrink_stress.py'),scope='Artificial sensitivity only, not hidden OD accuracy. Baseline unchanged.',official_score=None))
if __name__=='__main__':main()
