"""Complete inference and delivery after the report-only family-label repair."""
import datetime as dt
import subprocess
import sys
from optimize import ROOT,save

for script,phase in [('build_candidate.py','building_candidate'),('write_report.py','writing_report')]:
    save(ROOT/'progress.json',dict(phase=phase))
    with (ROOT/script.replace('.py','.log')).open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-X','utf8',str(ROOT/script)],stdout=log,stderr=subprocess.STDOUT,check=True)
save(ROOT/'progress.json',dict(phase='completed',completed_at=dt.datetime.now().astimezone().isoformat()))
