from __future__ import annotations

import argparse
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('Must be positive')
    return value


def main():
    parser = argparse.ArgumentParser(description='GRIN: fixed I5 / I405 June-December fit and January development protocol')
    parser.add_argument('action', choices=['run','prepare','evaluate','log'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--run', type=Path, help='Completed run for log retry')
    parser.add_argument('--release-root', type=Path, default=PROJECT/'2026-ieee-big-data-traffic-flow-bench/kaggle_public')
    parser.add_argument('--cache-dir', type=Path, default=PROJECT/'cache/grin')
    parser.add_argument('--baseline-run', type=Path, default=PROJECT/'runs/cross_corridor_review_20261016_210254')
    parser.add_argument('--baseline-sample-table', type=Path)
    parser.add_argument('--panel', default='D12_I5_N')
    parser.add_argument('--fit-start', default='2030-06-01'); parser.add_argument('--fit-end', default='2030-12-31')
    parser.add_argument('--eval-start', default='2031-01-01'); parser.add_argument('--eval-end', default='2031-01-31')
    parser.add_argument('--max-fit-days', type=positive); parser.add_argument('--max-eval-days', type=positive)
    parser.add_argument('--sample-per-day', type=positive, default=1000)
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--device', choices=['auto','cuda','cpu'], default='auto')
    parser.add_argument('--threads', type=positive, default=4)
    parser.add_argument('--hidden', type=positive, default=32)
    parser.add_argument('--batch-size', type=positive, default=4)
    parser.add_argument('--window', type=positive, default=48)
    parser.add_argument('--stride', type=positive, default=24)
    parser.add_argument('--epochs', type=positive, default=20)
    parser.add_argument('--steps-per-epoch', type=positive, default=100)
    parser.add_argument('--patience', type=positive, default=5)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--dropout', type=float, default=.1)
    parser.add_argument('--drop-observed', type=float, default=.05)
    parser.add_argument('--aux-weight', type=float, default=.1)
    parser.add_argument('--no-log', action='store_true')
    args = parser.parse_args()
    if Path.cwd().resolve() != PROJECT:
        parser.error(f'Run from {PROJECT}')
    for name in ['output','checkpoint','run','cache_dir','baseline_run','baseline_sample_table']:
        value = getattr(args,name)
        if value is not None:
            value = value.resolve()
            if not value.is_relative_to(PROJECT):
                parser.error(f'--{name.replace("_","-")} must be inside project')
            setattr(args,name,value)
    if args.action == 'log':
        if not args.run:
            parser.error('log requires --run')
        from .log import append_grin_log
        print(append_grin_log(args.run,PROJECT/'实验日志.txt'))
        return
    if not args.output:
        parser.error('--output is required')
    if args.action == 'evaluate' and not args.checkpoint:
        parser.error('evaluate requires --checkpoint')
    if args.learning_rate <= 0 or not 0 <= args.dropout < 1:
        parser.error('Invalid learning rate/dropout')
    from .runner import run
    run(args)


if __name__ == '__main__':
    main()
