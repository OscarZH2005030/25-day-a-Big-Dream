"""Resume unchanged completed cases after a transient Windows status-file lock."""
import common as d
import task4_smallnet_posterior_stress as study
import json,shutil,msvcrt,traceback
from pathlib import Path
OLD=d.OUT/'task4_smallnet_posterior_stress';NEW=d.OUT/'task4_smallnet_posterior_stress_retry'

def main():
 NEW.mkdir(parents=True,exist_ok=True)
 for name in ['all_cases.csv','convergence_diagnostics.json']:
  if (OLD/name).exists() and not (NEW/name).exists():shutil.copy2(OLD/name,NEW/name)
 previous=json.loads((OLD/'protocol.json').read_text());original_save=d.save
 def checked_save(path,value):
  if Path(path).resolve()==(NEW/'protocol.json').resolve():
   assert {k:v for k,v in value.items() if k not in ['source_sha256','registered_utc']}=={k:v for k,v in previous.items() if k not in ['source_sha256','registered_utc']},'Scientific protocol changed during recovery'
  original_save(path,value)
 d.save=checked_save;study.O=NEW;d.save(NEW/'repair.json',dict(reason='Windows reader temporarily denied atomic status replacement; bounded PermissionError retry and unique temp filenames added.',old_results_protocol_sha256=d.sha(OLD/'protocol.json'),old_source_snapshot_sha256=d.sha(d.WORK/'failure_history/smallnet_state_lock_attempt1/source.py'),new_source_sha256=d.sha(d.WORK/'task4_smallnet_posterior_stress.py'),resume='Completed seven-method cases with saved diagnostics are skipped, all earlier outcomes retained. Incomplete cases rerun from the same case seed.',scientific_protocol_unchanged=True,IO_test_sha256=d.sha(d.OUT/'runtime_save_repair.json')));study.main()

if __name__=='__main__':
 lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:main()
 except Exception as e:d.state(status='failed_or_cutoff',stage='task4_smallnet_resume',error=repr(e),next='Preserve original failure and retry artifacts; do not change seeds or scientific parameters to recover execution.');traceback.print_exc();raise
 finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
