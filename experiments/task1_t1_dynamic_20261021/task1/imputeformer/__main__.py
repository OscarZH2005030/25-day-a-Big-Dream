from __future__ import annotations

import argparse
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('Must be positive')
    return result


def main():
    parser = argparse.ArgumentParser(description='Compact ImputeFormer: I5, fixed June-December fit / January dev')
    parser.add_argument('action', choices=['run', 'prepare', 'evaluate', 'log'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--release-root', type=Path, default=PROJECT/'2026-ieee-big-data-traffic-flow-bench/kaggle_public')
    parser.add_argument('--cache-dir', type=Path, default=PROJECT/'cache/grin')
    parser.add_argument('--baseline-run', type=Path, default=PROJECT/'runs/target_dev_20261016_183242')
    parser.add_argument('--grin-run', type=Path, default=PROJECT/'runs/grin_dev_20261017_015925')
    parser.add_argument('--hybrid-run', type=Path, default=PROJECT/'runs/grin_hybrid_i5_20261017_142725')
    parser.add_argument('--baseline-sample-table', type=Path)
    parser.add_argument('--panel', choices=['D12_I5_N'], default='D12_I5_N')
    parser.add_argument('--fit-start', default='2030-06-01'); parser.add_argument('--fit-end', default='2030-12-31')
    parser.add_argument('--eval-start', default='2031-01-01'); parser.add_argument('--eval-end', default='2031-01-31')
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--max-fit-days', type=positive); parser.add_argument('--max-eval-days', type=positive)
    parser.add_argument('--sample-per-day', type=positive, default=1000)
    parser.add_argument('--device', choices=['cuda','cpu','auto'], default='cuda')
    for name, default in [('threads',4),('batch-size',2),('window',48),('stride',24),('hidden',64),
                          ('embedding-dim',32),('heads',4),('layers',2),('rank',8),('feed-forward',128),
                          ('epochs',20),('steps-per-epoch',100),('patience',5)]:
        parser.add_argument('--'+name, type=positive, default=default)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--dropout', type=float, default=.1)
    parser.add_argument('--drop-observed', type=float, default=.05)
    parser.add_argument('--no-log', action='store_true')
    args = parser.parse_args()
    if Path.cwd().resolve() != PROJECT:
        parser.error(f'Run from {PROJECT}')
    for name in ['output','run','checkpoint','cache_dir','baseline_run','grin_run','hybrid_run','baseline_sample_table']:
        value = getattr(args, name)
        if value is not None:
            value = value.resolve()
            if not value.is_relative_to(PROJECT):
                parser.error(f'--{name.replace("_", "-")} must stay inside project')
            setattr(args, name, value)
    if args.action == 'log':
        if not args.run: parser.error('log requires --run')
        from .log import append_log
        print(append_log(args.run, PROJECT/'实验日志.txt'))
        return
    if not args.output: parser.error('--output is required')
    if args.action == 'evaluate' and not args.checkpoint: parser.error('evaluate requires --checkpoint')
    from .runner import run
    try:
        run(args)
    except __import__('torch').cuda.OutOfMemoryError:
        from ..data import save_json
        save_json(args.output/'FAILED.json', dict(reason='CUDA out of memory', suggestion='Use a new output and --batch-size 1; no silent configuration change'))
        raise


if __name__ == '__main__':
    main()
