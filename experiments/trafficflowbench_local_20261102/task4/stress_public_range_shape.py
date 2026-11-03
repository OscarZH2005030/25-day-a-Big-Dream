"""Reproduce fixed 80 public-range count-shape cases, not official scores."""
import argparse,shutil,sys,types
from pathlib import Path
from datetime import datetime,timezone
from stress_observable_gate import load,save,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--release-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();out=args.output.resolve();work=out/'work';work.mkdir(parents=True,exist_ok=True);archive=Path(__file__).resolve().parents[1]/'archive/daytime'
    common=types.ModuleType('common');common.DATA=args.release_root.resolve();common.WORK=common.ROOT=work;common.OUT=out;common.sha=sha;common.save=save;common.check_deadline=lambda:None
    common.state=lambda **kw:save(work/'STATE.json',dict(utc=datetime.now(timezone.utc).isoformat(),**kw));common.night=types.SimpleNamespace(predictor=types.SimpleNamespace(load_routes=lambda:{p.name:None for p in sorted((common.DATA/'task4').iterdir()) if p.is_dir()}));sys.modules['common']=common
    for name in ['task4_prior_structure','task4_span_shrink_stress','task4_crossmonth_shared_noise_stress','task4_crossmonth_shared_noise_qr','task4_count_source_pair_stress']:
        shutil.copy2(archive/(name+'.py'),work/(name+'.py'));load(name,archive/(name+'.py'))
    module=load('registered_shape_stress',archive/'task4_public_range_shape_stress.py');module.main()
    save(out/'portable_entrypoint.json',dict(entrypoint_sha256=sha(__file__),scope='Artificial count-shape stress at TV 0.14; not hidden OD validation or an official candidate.',official_score=None))


if __name__=='__main__':main()
