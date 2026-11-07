"""Reproduce the fresh 480-case OD reprojection stress at seed 2026110643."""
import argparse,shutil,sys,types
from pathlib import Path
import numpy as np
from datetime import datetime,timezone
from stress_observable_gate import load,save,sha


def draw_with_count_error(paths, matrix, level, truth_name, noise, drift, rng):
    truth, priors, counts, destinations = original_draw(
        paths, matrix, level, truth_name, noise, drift, rng)
    # Use an independent stream without changing original scenario draws.
    key = int(rng.bit_generator.state['state']['state']) % (2**64)
    error_rng = np.random.default_rng(key ^ 2026110644)
    bias = error_rng.normal(-.03**2 / 2, .03, counts.shape[1])
    noise = error_rng.normal(-.02**2 / 2, .02, counts.shape)
    return truth, priors, counts * np.exp(bias + noise), destinations

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--release-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();out=args.output.resolve();work=out/'work';work.mkdir(parents=True,exist_ok=True);archive=Path(__file__).resolve().parents[1]/'archive/daytime'
    common=types.ModuleType('common');common.DATA=args.release_root.resolve();common.WORK=common.ROOT=work;common.OUT=out;common.sha=sha;common.save=save;common.check_deadline=lambda:None
    common.state=lambda **kw:save(work/'STATE.json',dict(utc=datetime.now(timezone.utc).isoformat(),**kw));common.night=types.SimpleNamespace(predictor=types.SimpleNamespace(load_routes=lambda:{p.name:None for p in sorted((common.DATA/'task4').iterdir()) if p.is_dir()}));sys.modules['common']=common
    for name in ['task4_prior_structure','task4_span_shrink_stress','task4_crossmonth_shared_noise_stress','task4_crossmonth_shared_noise_qr','task4_count_source_pair_stress','task4_exact_pair_audit','od_cycle_refinement']:
        shutil.copy2(archive/(name+'.py'),work/(name+'.py'));load(name,archive/(name+'.py'))
    module=load('registered_od_cycle_stress',archive/'task4_od_cycle_stress.py');module.O=out/'stress';module.SEED=2026110643
    global original_draw
    original_draw = module.generator.draw
    module.generator.draw = draw_with_count_error
    module.main()
    reprojection = load('fresh_reprojection', archive / 'audit_task4_cycle_projected_reverse.py')
    reprojection.main(module.O / 'all_cases.csv')
    save(out/'portable_entrypoint.json',dict(entrypoint_sha256=sha(__file__),scope='Artificial matching-only risk and invariant test; not a global optimizer, hidden OD validation or an official candidate.',official_score=None))


if __name__=='__main__':main()
