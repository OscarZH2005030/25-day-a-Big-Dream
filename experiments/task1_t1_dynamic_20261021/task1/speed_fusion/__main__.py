from __future__ import annotations
import argparse
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description='I5 saved-prediction speed fusion; fixed 10/25/50% on 15<gap<=60min only')
    p.add_argument('action',choices=['run','prepare','log'])
    p.add_argument('--output',type=Path);p.add_argument('--run',type=Path)
    p.add_argument('--source-run',type=Path,default=PROJECT/'runs/wave_context_dev_20261018_031857_573')
    p.add_argument('--smoke',action='store_true');p.add_argument('--no-log',action='store_true')
    a=p.parse_args()
    if Path.cwd().resolve()!=PROJECT:p.error(f'Run from {PROJECT}')
    for name in ['output','run','source_run']:
        value=getattr(a,name)
        if value is not None:
            value=value.resolve()
            if not value.is_relative_to(PROJECT):p.error('--'+name.replace('_','-')+' must stay inside project')
            setattr(a,name,value)
    if a.action=='log':
        if a.run is None:p.error('log requires --run')
        from .log import append_log
        print(append_log(a.run,PROJECT/'实验日志.txt'));return
    if a.output is None:p.error('--output required')
    if a.output.exists() and any(a.output.iterdir()):p.error('Choose a new empty output directory')
    from ..data import save_json
    from .runner import run
    try:run(a)
    except Exception as error:
        if not (a.output/'COMPLETE.json').exists():
            save_json(a.output/'FAILED.json',dict(reason=type(error).__name__,message=str(error),suggestion='Inspect saved-source/hash/key mismatch; choose a new output; no model fallback or automatic retraining.'))
        raise


if __name__=='__main__':main()
