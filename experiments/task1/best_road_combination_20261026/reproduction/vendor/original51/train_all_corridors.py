"""Sequential per-corridor training over all official train data. No submissions."""
import argparse
import atexit
from training_io import RunLock
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd


def write_json(path,value):
    temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(value,indent=2),encoding='utf-8'); temp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--release-root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    root=Path(a.release_root).resolve(); out=Path(a.output).resolve()
    panels=sorted(q.name for q in (root/'corridors').iterdir() if q.is_dir())
    date_sets=[]
    for panel in panels:
        dates={q.stem[-10:].replace('_','-') for q in (root/'corridors'/panel/'train/mainline_states_masked').rglob('*.parquet')}
        truths={q.stem[-10:].replace('_','-') for q in (root/'corridors'/panel/'train/mainline_states').rglob('*.parquet')}
        if dates!=truths: raise ValueError(f'Masked/truth dates mismatch: {panel}')
        date_sets.append(dates)
    if len(panels)!=10 or not all(x==date_sets[0] for x in date_sets): raise ValueError('Expected 10 aligned corridors')
    dates=sorted(date_sets[0]); dev=int(len(dates)*.8); internal=int(dev*.9)
    plan=dict(release_root=str(root),panels=panels,models='independent per corridor, sequential CUDA training',
              dates=dates,history_dates=dates[:28],training_dates=dates[28:internal],
              internal_validation_dates=dates[internal:dev],holdout_dates=dates[dev:],
              epochs=60,steps_per_epoch=1024,batch_size=2,window=24,patch_nodes=64,width=64,heads=4,layers=3,
              patience=8,seed=2026,physics=False,normalization='per-corridor; supervised training dates only',
              note='Official train only; first 80 percent includes history and internal validation. The final 20 percent overlaps previously inspected January/February; it is not wholly untouched.')
    out.mkdir(parents=True,exist_ok=True)
    guard=RunLock(out/'queue.lock').acquire()
    atexit.register(guard.release)
    if (out/'plan.json').exists():
        if not a.resume: raise ValueError('Existing run; pass --resume')
        if json.loads((out/'plan.json').read_text())!=plan: raise ValueError('Plan mismatch')
    else:
        write_json(out/'plan.json',plan)
        subprocess.run([sys.executable,'-m','pip','freeze'],stdout=(out/'environment_pip_freeze.txt').open('w'),check=True)
        hashes={q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in Path(__file__).parent.glob('*.py')}
        write_json(out/'source_sha256.json',hashes)
    expected_hashes=json.loads((out/'source_sha256.json').read_text())
    for name,sha in expected_hashes.items():
        if hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('Source changed; use the original source to resume')
    status=dict(pid=os.getpid(),state='starting',completed=[],current=None,updated_utc='')
    def update(**kw):
        status.update(kw); status['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(out/'status.json',status)
    # Keep this machine awake while the authorized training process is active.
    if os.name=='nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        for panel in panels:
            folder=out/panel; folder.mkdir(exist_ok=True)
            if (folder/'COMPLETE.json').exists():
                status['completed'].append(panel); continue
            cmd=[sys.executable,'-u',str(Path(__file__).with_name('full_train_worker.py')),
                 '--release-root',str(root),'--output',str(folder),'--history',str(folder/'history.npz'),
                 '--panel',panel,'--train-start',dates[28],'--internal-val-start',dates[internal],
                 '--val-start',dates[dev],'--val-end',(pd.Timestamp(dates[-1])+pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                 '--device','cuda','--epochs','60','--steps-per-epoch','1024','--batch-size','2',
                 '--window','24','--patch-nodes','64','--width','64','--heads','4','--layers','3','--patience','8',
                 '--fd-weight','0','--conservation-weight','0']
            if (folder/'last.pt').exists(): cmd+=['--resume',str(folder/'last.pt')]
            write_json(folder/'command.json',cmd)
            update(state='running',current=panel)
            print(f'START {panel}: {len(status["completed"])+1}/{len(panels)}',flush=True)
            with (folder/'console.log').open('a',encoding='utf-8') as log:
                proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                      cwd=Path(__file__).parent,text=True,encoding='utf-8',errors='replace')
                update(worker_pid=proc.pid)
                try:
                    for line in proc.stdout:
                        log.write(line); log.flush()
                        print(f'[{panel}] {line}',end='',flush=True)
                    code=proc.wait()
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                        proc.wait()
            if code: raise RuntimeError(f'{panel} exited {code}; see {folder / "console.log"}')
            metrics=pd.read_csv(folder/'metrics.csv')
            value=float(metrics.loc[metrics.regime=='macro_mean','S_state'].iloc[0])
            write_json(folder/'COMPLETE.json',dict(panel=panel,S_state=value))
            status['completed'].append(panel); update(current=None)
            print(f'FINISHED {panel}: {value:.7f}',flush=True)
        rows=[]
        for panel in panels:
            df=pd.read_csv(out/panel/'metrics.csv'); df.insert(0,'panel',panel); rows.append(df)
        all_rows=pd.concat(rows,ignore_index=True); all_rows.to_csv(out/'all_metrics.csv',index=False)
        write_json(out/'summary.json',dict(corridors=len(panels),mean_corridor_S_state=float(all_rows.loc[all_rows.regime=='macro_mean','S_state'].mean()),note='Descriptive equal-corridor average of local holdout scores; not an official total competition score.'))
        update(state='complete',current=None)
    except BaseException as error:
        update(state='failed',error=str(error)); raise
    finally:
        if os.name=='nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__': main()
