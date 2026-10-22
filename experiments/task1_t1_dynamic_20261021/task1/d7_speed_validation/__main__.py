from __future__ import annotations
import argparse
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description='D7 replication of frozen I5 wave-enhanced/no-wave speed fusion')
    p.add_argument('action',choices=['run','prepare','log'])
    p.add_argument('--output',type=Path);p.add_argument('--run',type=Path)
    p.add_argument('--cache-dir',type=Path,default=PROJECT/'cache/d7_speed_validation')
    p.add_argument('--baseline-run',type=Path,default=PROJECT/'runs/corridor_dev_20261016_192125')
    p.add_argument('--hybrid-run',type=Path,default=PROJECT/'runs/grin_hybrid_d7_20261017_143642')
    p.add_argument('--i5-wave-run',type=Path,default=PROJECT/'runs/traffic_wave_dev_20261017_174736_955')
    p.add_argument('--i5-neural-run',type=Path,default=PROJECT/'runs/wave_context_dev_20261018_031857_573')
    p.add_argument('--i5-fusion-run',type=Path,default=PROJECT/'runs/speed_fusion_dev_20261018_143753_506')
    p.add_argument('--device',choices=['cuda','cpu','auto'],default='cuda')
    p.add_argument('--batch-size',type=int,default=2);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--smoke',action='store_true');p.add_argument('--no-log',action='store_true')
    a=p.parse_args()
    if Path.cwd().resolve()!=PROJECT:p.error(f'Run from {PROJECT}')
    for name in ['output','run','cache_dir','baseline_run','hybrid_run','i5_wave_run','i5_neural_run','i5_fusion_run']:
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
            save_json(a.output/'FAILED.json',dict(reason=type(error).__name__,message=str(error),
                suggestion='Inspect source/cache failure and use a new output. Same completed source stages are reusable. No silent device/budget fallback; I5 recipe fixes batch2/threads4.'))
        raise


if __name__=='__main__':main()
