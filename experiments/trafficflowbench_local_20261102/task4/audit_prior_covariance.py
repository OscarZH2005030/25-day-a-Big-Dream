"""Portable entrypoint for the archived, fixed-protocol prior covariance audit."""
import argparse,hashlib,importlib.util,json,sys,types
from pathlib import Path
from datetime import datetime,timezone

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()

def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf8')

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--release-root',type=Path,required=True)
 parser.add_argument('--output',type=Path,required=True)
 parser.add_argument('--work',type=Path)
 args=parser.parse_args();work=args.work or args.output/'work';work.mkdir(parents=True,exist_ok=True)
 # Supply only filesystem/state helpers and panel inventory. No original model
 # loader, local machine paths, authentication or research workspace is needed.
 common=types.ModuleType('common');common.DATA=args.release_root;common.WORK=work;common.OUT=args.output.parent
 common.sha=sha;common.save=save;common.check_deadline=lambda:None
 common.state=lambda **kw:save(work/'STATE.json',dict(utc=datetime.now(timezone.utc).isoformat(),**kw))
 common.night=types.SimpleNamespace(predictor=types.SimpleNamespace(load_routes=lambda:{p.name:None for p in sorted((args.release_root/'task4').iterdir()) if p.is_dir()}))
 sys.modules['common']=common
 source=Path(__file__).resolve().parents[1]/'archive/daytime/task4_covariance_audit.py'
 spec=importlib.util.spec_from_file_location('registered_covariance_audit',source)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 module.O=args.output;args.output.mkdir(parents=True,exist_ok=True);module.main()
 save(args.output/'portable_entrypoint.json',dict(source_sha256=sha(source),entrypoint_sha256=sha(__file__),official_score=None,scope='Public count/prior residual likelihood only; not hidden path accuracy.'))
if __name__=='__main__':main()
