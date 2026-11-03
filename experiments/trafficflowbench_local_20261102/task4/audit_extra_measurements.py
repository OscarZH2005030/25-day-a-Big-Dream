"""Reproduce fixed hypothetical OD measurement audit, not official scores."""
import argparse,shutil,sys,types
from pathlib import Path
from datetime import datetime,timezone
from stress_observable_gate import load,save,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--release-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();out=args.output.resolve();work=out/'work';work.mkdir(parents=True,exist_ok=True);archive=Path(__file__).resolve().parents[1]/'archive/daytime'
    common=types.ModuleType('common');common.DATA=args.release_root.resolve();common.WORK=common.ROOT=work;common.OUT=out;common.sha=sha;common.save=save;common.check_deadline=lambda:None
    common.state=lambda **kw:save(work/'STATE.json',dict(utc=datetime.now(timezone.utc).isoformat(),**kw));common.night=types.SimpleNamespace(predictor=types.SimpleNamespace(load_routes=lambda:{p.name:None for p in sorted((common.DATA/'task4').iterdir()) if p.is_dir()}));sys.modules['common']=common
    for name in ['task4_prior_structure','task4_span_shrink_stress','task4_crossmonth_shared_noise_stress','task4_crossmonth_shared_noise_qr','task4_exact_pair_audit']:
        shutil.copy2(archive/(name+'.py'),work/(name+'.py'));load(name,archive/(name+'.py'))
    import pandas as pd
    baseline = pd.read_csv(Path(__file__).with_name('frozen_reference_path_flow.csv'),
                           dtype={'panel':str,'path_id':str,'departure_time':str})
    anchors = []
    for panel in common.night.predictor.load_routes():
        for split in ['validation','private']:
            folder = common.DATA/'task4'/panel/split
            prior = pd.read_csv(folder/'synthetic_weak_prior.csv',dtype={'departure_time':str})
            frame = baseline[(baseline.panel==panel)&(baseline.departure_time==prior.departure_time.iloc[0])]
            assert len(frame)>0 and not frame.path_id.duplicated().any()
            anchor = work/'task4_metric_invariant'/panel/split/'linear_objective_0_NOT_A_SUBMISSION.parquet'
            anchor.parent.mkdir(parents=True,exist_ok=True)
            frame[['path_id','path_flow']].rename(columns={'path_flow':'baseline_flow'}).to_parquet(anchor,index=False)
            anchors.append(dict(panel=panel,split=split,replicate=0,witness_sha256=sha(anchor)))
    table = out/'task4_metric_invariant/witnesses.csv'
    table.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(anchors).to_csv(table,index=False)
    load('registered_extra_measurement',archive/'task4_extra_measurement_audit.py').main()
    save(out/'portable_entrypoint.json',dict(entrypoint_sha256=sha(__file__),scope='Hypothetical trip-span measurement audit; not hidden OD validation or an official candidate.',official_score=None))


if __name__=='__main__':main()
