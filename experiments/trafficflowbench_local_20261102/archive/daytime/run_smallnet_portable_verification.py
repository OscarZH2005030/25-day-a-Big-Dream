"""Coordinate one full standalone rerun with the existing shared research lock."""
import common as d
import msvcrt
import subprocess
import sys
import traceback
import pandas as pd
import numpy as np

lock = (d.ROOT / 'work/overnight/research.lock').open('r+b')
lock.seek(0)
msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
try:
    output = d.WORK / 'publication_smoke/smallnet_posterior'
    d.state(status='running', stage='smallnet_full_portable_reproduction', output=str(output), next='Reproduce all 96 registered cases without old workspace imports; compare every available metric and failure, then publish negative evidence. No Kaggle upload.')
    source = d.WORK / 'publish-repo/experiments/trafficflowbench_local_20261102/task4/stress_posterior_assumptions.py'
    subprocess.run([sys.executable, '-X', 'utf8', str(source), '--output', str(output)], check=True)
    original = pd.read_csv(d.OUT / 'task4_smallnet_posterior_stress_retry/all_cases.csv').sort_values(['case', 'method']).reset_index(drop=True)
    rerun = pd.read_csv(output / 'stress/all_cases.csv').sort_values(['case', 'method']).reset_index(drop=True)
    assert len(original) == len(rerun) == 672
    for column in ['case', 'method', 'posterior_converged', 'action_available', 'artifact_sha256']:
        assert original[column].equals(rerun[column]), column
    scores = [c for c in original if c.startswith('synthetic_')]
    assert np.array_equal(original[scores].isna(), rerun[scores].isna())
    error = float(np.nanmax(np.abs(original[scores].to_numpy() - rerun[scores].to_numpy())))
    assert error < 1e-12
    d.save(d.OUT / 'task4_smallnet_posterior_stress_retry/portable_reproduction.json', dict(cases=96, rows=672, all_sample_artifact_hashes_identical=True, all_convergence_and_availability_flags_identical=True, maximum_metric_absolute_error=error, runner_sha256=d.sha(__file__), scope='Reproducibility of artificial negative results; not official OD validation.'))
    d.state(status='stage_complete', stage='smallnet_full_portable_reproduction_complete', next='Publish verified misspecification harms and train-only mean audit, then continue variance-structure research. Official Task4 baseline unchanged.')
except Exception as error:
    d.state(status='failed', stage='smallnet_full_portable_reproduction', error=repr(error), next='Preserve failure; repair portability without altering seeds, cases or methods.')
    traceback.print_exc()
    raise
finally:
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
    lock.close()
