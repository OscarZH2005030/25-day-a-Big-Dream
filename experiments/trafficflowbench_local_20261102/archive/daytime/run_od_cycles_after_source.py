"""Continue after the known upstream reproduction, never overlap the research lock."""
import common as d
import os
import time
import json
import msvcrt
import subprocess
import sys
import traceback


def main():
    queue=d.WORK/'od_cycle_queue.json';lock=(d.ROOT/'work/overnight/research.lock').open('r+b');acquired=False
    d.save(queue,dict(pid=os.getpid(),status='waiting_for_shared_lock',upstream_required='source_count_three_studies_reproduced',source_sha256=d.sha(__file__),next_scientific_seed=2026110325))
    started=time.monotonic()
    try:
        while not acquired:
            try:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);acquired=True
            except OSError:
                if time.monotonic()-started>1800:
                    d.save(queue,dict(pid=os.getpid(),status='wait_limit_reached',reason='Upstream still holds shared lock; no unknown process touched.'));return
                time.sleep(1)
        upstream=json.loads((d.WORK/'STATE.json').read_text())
        if upstream.get('stage')!='source_count_three_studies_reproduced' or upstream.get('status')!='stage_complete':
            d.save(queue,dict(pid=os.getpid(),status='upstream_not_verified',upstream=upstream,next='Review the existing upstream failure/status; no new experiment started.'));return
        d.save(queue,dict(pid=os.getpid(),status='numerical_controls',upstream_verified=True))
        d.state(status='running',stage='od_cycle_numerical_controls',next='Verify nontrivial cycle update, oracle L1 monotonicity and non-OD invariants before fresh matching-only stress. No Kaggle upload.')
        subprocess.run([sys.executable,'-X','utf8',str(d.WORK/'verify_od_cycles.py')],check=True)
        import task4_od_cycle_stress as study
        d.save(queue,dict(pid=os.getpid(),status='running',seed=study.SEED,numerical_controls_passed=True))
        study.main()
        d.save(queue,dict(pid=os.getpid(),status='complete',seed=study.SEED,numerical_controls_passed=True))
    except Exception as error:
        d.save(queue,dict(pid=os.getpid(),status='failed',error=repr(error)))
        d.state(status='failed',stage='od_cycle_controls_or_stress',error=repr(error),next='Preserve inputs, fixed cycles and failures; repair numerical invariants without changing scientific cases or selecting roads.');traceback.print_exc();raise
    finally:
        if acquired:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
        lock.close()


if __name__=='__main__':main()
