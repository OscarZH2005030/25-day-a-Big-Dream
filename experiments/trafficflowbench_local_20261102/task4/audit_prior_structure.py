"""Portable reproduction of weak-prior structure holdouts and recipe transfer."""
import argparse,hashlib,importlib.util,json,os,sys,types,shutil
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
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--release-root',type=Path,required=True)
 parser.add_argument('--output',type=Path,required=True)
 parser.add_argument('--train-only',action='store_true')
 args=parser.parse_args();out=args.output.resolve();work=out/'work';work.mkdir(parents=True,exist_ok=True)
 common=types.ModuleType('common');common.DATA=args.release_root.resolve();common.WORK=work;common.ROOT=work;common.OUT=out
 common.sha=sha;common.save=save;common.check_deadline=lambda:None
 common.state=lambda **kw:save(work/'STATE.json',dict(utc=datetime.now(timezone.utc).isoformat(),**kw))
 common.night=types.SimpleNamespace(predictor=types.SimpleNamespace(load_routes=lambda:{p.name:None for p in sorted((common.DATA/'task4').iterdir()) if p.is_dir()}));sys.modules['common']=common
 archive=Path(__file__).resolve().parents[1]/'archive/daytime'
 # The transfer protocol fingerprints this exact feature source in WORK.
 shutil.copy2(archive/'task4_prior_structure.py',work/'task4_prior_structure.py')
 # The archive is the original registered scientific code. This facade supplies
 # portable filesystem helpers, without importing the original model workspace.
 train=load('task4_prior_structure',archive/'task4_prior_structure.py');train.O=out/'train';train.O.mkdir(parents=True,exist_ok=True);train.main()
 if not args.train_only:
  transfer=load('registered_prior_structure_transfer',archive/'task4_prior_structure_transfer.py');transfer.O=out/'transfer';transfer.O.mkdir(parents=True,exist_ok=True);transfer.main()
 save(out/'portable_entrypoint.json',dict(entrypoint_sha256=sha(__file__),target='Published weak-prior entries, NOT hidden path truth',official_score=None))
if __name__=='__main__':main()
