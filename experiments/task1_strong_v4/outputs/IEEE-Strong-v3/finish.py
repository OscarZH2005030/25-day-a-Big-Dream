"""Wait for frozen correction routes, then evaluate and deliver the candidate."""
import datetime as dt
import subprocess
import sys
import time
from residual_core import ROOT,save

while not (ROOT/'frozen-routes.json').exists():
    if (ROOT/'STOP').exists():raise RuntimeError('Stop requested')
    log=ROOT/'controller.log'
    if log.exists() and 'Traceback (most recent call last)' in log.read_text(encoding='utf-8'):
        raise RuntimeError('Training controller failed; inspect its log before continuing')
    time.sleep(3)
for script,phase in [('evaluate_residual.py','evaluating'),('build_candidate.py','building_candidate'),('write_report.py','writing_report')]:
    save(ROOT/'progress.json',dict(phase=phase))
    with (ROOT/script.replace('.py','.log')).open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-X','utf8',str(ROOT/script)],stdout=log,stderr=subprocess.STDOUT,check=True)
save(ROOT/'progress.json',dict(phase='completed',completed_at=dt.datetime.now().astimezone().isoformat()))
