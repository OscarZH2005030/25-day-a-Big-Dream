"""Verify all saved actions then fully reproduce the frozen OD-only experiment."""
import common as d
import subprocess,sys,msvcrt,traceback,os
import pandas as pd
import numpy as np


def main():
    lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:
        d.state(status='running',stage='od_cycle_all_case_readback',next='Independently recompute all 2880 actions, score-floor diagnostics and non-OD invariants before discussing candidate tradeoffs; no Kaggle upload.')
        subprocess.run([sys.executable,'-X','utf8',str(d.WORK/'verify_od_cycle_results.py')],check=True)
        out=d.WORK/'publication_smoke'/os.environ.get('TFB_OD_CYCLE_REPRO_FOLDER','od_cycle_stress');source=d.WORK/'publish-repo/experiments/trafficflowbench_local_20261102/task4/stress_od_cycles.py'
        d.state(status='running',stage='od_cycle_full_portable_reproduction',output=str(out),next='Rerun all fixed 480 cases and cycle sequences; compare artifacts and unclipped OD errors, no official improvement claim.')
        subprocess.run([sys.executable,'-X','utf8',str(source),'--release-root',str(d.DATA),'--output',str(out)],check=True)
        errors={}
        for name in ['all_cases.csv','paired_risk.csv','by_assumption.csv']:
            a,b=pd.read_csv(d.OUT/'task4_od_cycle_stress'/name),pd.read_csv(out/'stress'/name)
            assert a.shape==b.shape and a.columns.equals(b.columns),name
            numeric=a.select_dtypes(include='number').columns
            for col in a.columns.difference(numeric):assert a[col].equals(b[col]),(name,col)
            assert np.array_equal(a[numeric].isna(),b[numeric].isna());error=float(np.nanmax(np.abs(a[numeric].to_numpy()-b[numeric].to_numpy())))
            assert error<1e-12,(name,error);errors[name]=error
        d.save(d.OUT/'task4_od_cycle_stress/portable_reproduction.json',dict(cases=480,method_rows=2880,all_input_and_prediction_artifact_hashes_identical=True,maximum_CSV_metric_differences=errors,source_sha256=d.sha(__file__),scope='Fixed artificial heuristic tradeoff reproduction, not hidden OD validation.'))
        d.state(status='stage_complete',stage='od_cycle_full_reproduction_complete',next='Publish verified average-versus-tail tradeoffs and invariants; cycles can still harm OD. Study explicit movement-risk budgets without claiming unconditional official safety, keep reference frozen.')
        print('OD CYCLE STUDY REPRODUCED',errors,flush=True)
    except Exception as error:
        d.state(status='failed',stage='od_cycle_verification_or_reproduction',error=repr(error),next='Preserve original cases and cycle order, repair numerical/portable mismatch without tuning outcomes.');traceback.print_exc();raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':main()
