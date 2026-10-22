from __future__ import annotations
import argparse
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description='I5 traffic-wave reconstruction and fixed hybrid feature ablation')
    parser.add_argument('action',choices=['run','prepare','log'])
    parser.add_argument('--output',type=Path);parser.add_argument('--run',type=Path)
    parser.add_argument('--release-root',type=Path,default=PROJECT/'2026-ieee-big-data-traffic-flow-bench/kaggle_public')
    parser.add_argument('--cache-dir',type=Path,default=PROJECT/'cache/traffic_wave')
    parser.add_argument('--baseline-run',type=Path,default=PROJECT/'runs/target_dev_20261016_183242')
    parser.add_argument('--grin-run',type=Path,default=PROJECT/'runs/grin_dev_20261017_015925')
    parser.add_argument('--hybrid-run',type=Path,default=PROJECT/'runs/grin_hybrid_i5_20261017_142725')
    parser.add_argument('--device',choices=['cuda','cpu','auto'],default='cuda')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--no-log',action='store_true')
    args=parser.parse_args()
    if Path.cwd().resolve()!=PROJECT:parser.error(f'Run from {PROJECT}')
    for name in ['output','run','cache_dir','baseline_run','grin_run','hybrid_run']:
        value=getattr(args,name)
        if value is not None:
            value=value.resolve()
            if not value.is_relative_to(PROJECT):parser.error(f'--{name.replace("_","-")} must stay inside project')
            setattr(args,name,value)
    if args.action=='log':
        if not args.run:parser.error('log requires --run')
        from .log import append_log
        print(append_log(args.run,PROJECT/'实验日志.txt'));return
    if not args.output:parser.error('--output is required')
    import torch
    from .runner import run
    try:run(args)
    except torch.cuda.OutOfMemoryError:
        from ..data import save_json
        save_json(args.output/'FAILED.json',dict(reason='CUDA out of memory',suggestion='Close other GPU workloads or explicitly use --device cpu; no silent fallback'))
        raise


if __name__=='__main__':main()
