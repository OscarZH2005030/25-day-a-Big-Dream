"""Sequential completion controller: local experts, freeze, evaluate, deliver."""
import csv
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time
from optimize import ROOT,save,sha,telemetry,FIELDS
from local_flow_expert import PRIORITY,FOLDER


def main():
    while not (ROOT/'frozen-routes.json').exists():
        if (ROOT/'STOP').exists():raise RuntimeError('Stop requested')
        if (ROOT/'controller.log').exists() and 'Traceback (most recent call last)' in (ROOT/'controller.log').read_text(encoding='utf-8'):
            raise RuntimeError('First-stage controller failed; inspect it before continuing')
        time.sleep(3)
    for panel in PRIORITY:assert (FOLDER/panel/'prepared.json').exists()
    save(FOLDER/'implementation-at-training.json',dict(recorded_at=dt.datetime.now().astimezone().isoformat(),
        source_sha256=sha(ROOT/'local_flow_expert.py'),
        note='Added shared full-grid inference helper after CPU data preparation. Training feature schema and policy unchanged.'))
    started=time.monotonic();rows=[]
    with (FOLDER/'telemetry.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['timestamp','elapsed_seconds','ac_connected']+FIELDS);writer.writeheader()
        for i,panel in enumerate(PRIORITY):
            out=FOLDER/panel
            if (out/'selection.json').exists():continue
            save(ROOT/'progress.json',dict(phase='local_flow_training',panel=panel,index=i+1,total=len(PRIORITY)))
            with (out/'training.log').open('w',encoding='utf-8') as log:
                child=subprocess.Popen([sys.executable,'-X','utf8',str(ROOT/'local_flow_expert.py'),'--worker',panel],stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                try:
                    while child.poll() is None:
                        row=telemetry();row['elapsed_seconds']=time.monotonic()-started;rows.append(row)
                        writer.writerow(row);f.flush()
                        if row['temperature.gpu']>=86 or not row['ac_connected'] or (ROOT/'STOP').exists():raise RuntimeError('Resource watchdog stop')
                        time.sleep(3)
                finally:
                    if child.poll() is None:child.terminate();child.wait(timeout=20)
                if child.returncode:raise RuntimeError('Local expert failed: '+panel)
    local=[json.loads((FOLDER/p/'selection.json').read_text()) for p in PRIORITY]
    manifest=json.loads((ROOT/'frozen-routes.json').read_text());manifest['local_flow_routes']=local
    manifest['final_frozen_at']=dt.datetime.now().astimezone().isoformat()
    manifest['source_sha256']={p.name:sha(p) for p in ROOT.glob('*.py')}
    save(ROOT/'final-model.json',manifest)
    save(FOLDER/'monitor-summary.json',dict(seconds=time.monotonic()-started,peak_vram_mib=max(r['memory.used'] for r in rows),
        peak_power_w=max(r['power.draw'] for r in rows),peak_temperature_c=max(r['temperature.gpu'] for r in rows)))
    for script,phase in [('evaluate.py','evaluating_frozen_model'),('build_candidate.py','building_candidate'),('write_report.py','writing_report')]:
        save(ROOT/'progress.json',dict(phase=phase))
        with (ROOT/(script.replace('.py','.log'))).open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-X','utf8',str(ROOT/script)],stdout=log,stderr=subprocess.STDOUT,check=True)
    save(ROOT/'progress.json',dict(phase='completed',completed_at=dt.datetime.now().astimezone().isoformat()))


if __name__=='__main__':main()
