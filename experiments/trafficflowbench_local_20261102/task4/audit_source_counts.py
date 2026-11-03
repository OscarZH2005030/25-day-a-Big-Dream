"""Reproduce observable residual alignment, exact pair controls and optional source stress."""
import argparse
import shutil
import sys
import types
from pathlib import Path
from datetime import datetime,timezone
from stress_observable_gate import load,save,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--release-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--with-stress',action='store_true');args=parser.parse_args()
    out=args.output.resolve();work=out/'work';work.mkdir(parents=True,exist_ok=True);here=Path(__file__).resolve().parents[1];archive=here/'archive/daytime'
    common=types.ModuleType('common');common.DATA=args.release_root.resolve();common.WORK=common.ROOT=work;common.OUT=out;common.sha=sha;common.save=save;common.check_deadline=lambda:None
    common.state=lambda **kw:save(work/'STATE.json',dict(utc=datetime.now(timezone.utc).isoformat(),**kw));common.night=types.SimpleNamespace(predictor=types.SimpleNamespace(load_routes=lambda:{p.name:None for p in sorted((common.DATA/'task4').iterdir()) if p.is_dir()}));sys.modules['common']=common
    for name in ['task4_prior_structure','task4_span_shrink_stress','task4_crossmonth_shared_noise_stress','task4_crossmonth_shared_noise_qr']:
        shutil.copy2(archive/(name+'.py'),work/(name+'.py'));load(name,archive/(name+'.py'))
    model=out/'task4_variance_controls_audit/full_train_models.json';model.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(here/'results/task4_variance_controls_audit/full_train_models.json',model)
    for name in ['task4_count_residual_alignment','task4_exact_pair_audit']+(['task4_count_source_pair_stress'] if args.with_stress else []):
        module=load(name,archive/(name+'.py'));module.O=out/name;module.main()
    save(out/'portable_entrypoint.json',dict(entrypoint_sha256=sha(__file__),scope='Public residual/rank/prior diagnostics and optional artificial count-source controls; no hidden OD validation or official candidate.',official_score=None))


if __name__=='__main__':main()
