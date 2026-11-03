"""Reproduce cross-month weak-prior pair prediction and marginal controls."""
import argparse
import shutil
import sys
import types
from pathlib import Path
from datetime import datetime, timezone
from stress_observable_gate import load, save, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();work=out/'work';work.mkdir(parents=True,exist_ok=True)
    archive=Path(__file__).resolve().parents[1]/'archive/daytime'
    common=types.ModuleType('common');common.DATA=args.release_root.resolve();common.WORK=common.ROOT=work;common.OUT=out;common.sha=sha;common.save=save;common.check_deadline=lambda:None
    common.state=lambda **kw:save(work/'STATE.json',dict(utc=datetime.now(timezone.utc).isoformat(),**kw))
    common.night=types.SimpleNamespace(predictor=types.SimpleNamespace(load_routes=lambda:{p.name:None for p in sorted((common.DATA/'task4').iterdir()) if p.is_dir()}));sys.modules['common']=common
    shutil.copy2(archive/'task4_prior_structure.py',work/'task4_prior_structure.py');load('task4_prior_structure',archive/'task4_prior_structure.py')
    for name in ['task4_crossmonth_pair_audit','task4_pair_marginal_controls']:
        module=load(name,archive/(name+'.py'));module.O=out/name;module.main()
    save(out/'portable_entrypoint.json',dict(entrypoint_sha256=sha(__file__),scope='Held-out supplied weak-prior prediction only, not real OD truth or official scoring.',official_score=None))


if __name__=='__main__':main()
