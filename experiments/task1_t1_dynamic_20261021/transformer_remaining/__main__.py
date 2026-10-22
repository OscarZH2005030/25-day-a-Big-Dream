import argparse
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo
import sys


def main():
    parser=argparse.ArgumentParser(description='Remaining six directions: frozen T1/dynamic-mask three-batch queue')
    parser.add_argument('action',choices=['run','prepare','summary','log','batch'])
    parser.add_argument('--project',default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--release-root');parser.add_argument('--cache-dir');parser.add_argument('--output')
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--batches',nargs='+',choices=['i210','i10','i405'],default=['i210','i10','i405'])
    parser.add_argument('--batch',choices=['i210','i10','i405'])
    parser.add_argument('--batch-action',choices=['run','prepare'],default='run')
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.resume and not args.output:
        parser.error('--resume requires the original top-level --output directory')
    args.project=Path(args.project).resolve();sys.path.insert(0,str(args.project))
    args.release_root=Path(args.release_root).resolve() if args.release_root else args.project/'2026-ieee-big-data-traffic-flow-bench/kaggle_public'
    args.cache_dir=Path(args.cache_dir).resolve() if args.cache_dir else args.project/'cache/transformer_cross_corridor'
    if args.action in ['summary','log','batch'] and not args.output:
        parser.error('--output is required for summary, log and internal batch execution')
    stamp=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d_%H%M%S_%f')
    label='prepare_' if args.action=='prepare' else 'smoke_' if args.smoke else 'dev_'
    args.output=Path(args.output).resolve() if args.output else args.project/'runs'/('remaining_dynamic_'+label+stamp)
    if args.action=='batch':
        if not args.batch: parser.error('--batch is required for internal batch execution')
        from .engine import engine_for
        engine=engine_for(args.batch)
        child=SimpleNamespace(**vars(args));child.action=args.batch_action;child.fusion_mode='rebuild_t0'
        engine.run(child)
    elif args.action in ['summary','log']:
        from .queue import postprocess
        postprocess(args)
    else:
        from .queue import run_queue
        raise SystemExit(run_queue(args))


if __name__=='__main__': main()
