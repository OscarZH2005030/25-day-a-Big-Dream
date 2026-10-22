from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import pipeline
from .audit import audit

PROJECT = Path(__file__).resolve().parents[1]
RELEASE = PROJECT / '2026-ieee-big-data-traffic-flow-bench' / 'kaggle_public'


def positive(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError('Must be positive')
    return n


def main():
    parser = argparse.ArgumentParser(description='TrafficFlowBench Task 1: audit, fit, evaluate, predict')
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ['audit', 'run', 'evaluate', 'fit', 'predict', 'ablate', 'target-compare', 'corridor-validate']:
        p = sub.add_parser(command)
        p.add_argument('--release-root', type=Path, default=RELEASE)
        p.add_argument('--panels', nargs='+', default=['D12_I5_N'], help='Panel IDs, or all')
        p.add_argument('--output', type=Path, required=True)
        if command == 'audit':
            p.add_argument('--max-days', type=positive, default=3, help='Sampled days per split')
            p.set_defaults(func=audit)
        if command in ['run', 'fit', 'ablate', 'target-compare', 'corridor-validate']:
            p.add_argument('--fit-start', default='2030-06-01')
            p.add_argument('--fit-end', default='2030-12-31')
            p.add_argument('--max-fit-days', type=positive)
            p.add_argument('--sample-per-day', type=positive, default=1000)
            p.add_argument('--rounds', type=positive, default=350)
            p.add_argument('--threads', type=positive, default=min(8, os.cpu_count() or 1))
            p.add_argument('--max-gap-minutes', type=positive, default=60)
            p.add_argument('--seed', type=int, default=20261016)
            p.set_defaults(func=pipeline.run if command == 'run' else fit_only)
        if command in ['run', 'evaluate', 'ablate', 'target-compare', 'corridor-validate']:
            p.add_argument('--eval-start', default='2031-01-01')
            p.add_argument('--eval-end', default='2031-01-31')
            p.add_argument('--max-eval-days', type=positive)
        if command in ['evaluate', 'predict']:
            p.add_argument('--models', type=Path, required=True, help='Directory containing per-panel model directories')
            p.set_defaults(func=pipeline.evaluate if command == 'evaluate' else pipeline.predict)
        if command == 'predict':
            p.add_argument('--splits', nargs='+', choices=['validation', 'private'], default=['validation', 'private'])
            p.add_argument('--method', choices=['B0_history', 'B1_interpolation', 'B2_lightgbm'], default='B2_lightgbm')
        if command == 'ablate':
            from .experiments import ablate
            p.add_argument('--variants', nargs='+', choices=['baseline', 'multihop', 'congestion'],
                           default=['multihop', 'congestion'])
            p.add_argument('--cache-dir', type=Path, default=PROJECT / 'cache' / 'task1')
            p.add_argument('--reference-run', type=Path, help='Require control B2 to reproduce this previous run')
            p.set_defaults(func=ablate)
        if command == 'target-compare':
            from .target_experiments import target_compare
            p.add_argument('--cache-dir', type=Path, default=PROJECT / 'cache' / 'task1')
            p.add_argument('--reference-run', type=Path, help='Previous three-feature residual experiment')
            p.add_argument('--no-log', action='store_true', help='Do not append formal results to experiment log')
            p.set_defaults(func=target_compare)
        if command == 'corridor-validate':
            from .corridor_validation import corridor_validate
            p.add_argument('--cache-dir', type=Path, default=PROJECT / 'cache' / 'task1')
            p.add_argument('--no-log', action='store_true')
            p.set_defaults(func=corridor_validate, panels=['D7_I405_N'])
    p = sub.add_parser('fusion-compare', help='Compare fixed multihop fusions using saved predictions, without training')
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--panels', nargs='+', default=['all'], help='Subset of source panels, or all')
    p.add_argument('--cache-dir', type=Path, help='Existing cache; defaults to source configuration')
    p.add_argument('--smoke', action='store_true', help='Verification only; do not append formal log')
    p.add_argument('--no-log', action='store_true')
    from .fusion_experiments import fusion_compare
    p.set_defaults(func=fusion_compare)
    p = sub.add_parser('log-fusion', help='Retry log append for a completed fusion run')
    p.add_argument('--run', type=Path, required=True)
    from .experiment_log import fusion_log_command
    p.set_defaults(func=fusion_log_command, log_path=PROJECT / '实验日志.txt')
    p = sub.add_parser('log-corridor', help='Retry log append for a completed corridor validation')
    p.add_argument('--run', type=Path, required=True)
    from .experiment_log import corridor_log_command
    p.set_defaults(func=corridor_log_command, log_path=PROJECT / '实验日志.txt')
    p = sub.add_parser('log-target', help='Retry log append for a completed target-comparison run')
    p.add_argument('--run', type=Path, required=True)
    from .experiment_log import log_command
    p.set_defaults(func=log_command, log_path=PROJECT / '实验日志.txt')
    args = parser.parse_args()
    # Refuse running with a working directory elsewhere: artifacts stay inside this project.
    if Path.cwd().resolve() != PROJECT:
        parser.error(f'Run from project directory: {PROJECT}')
    if args.command in ['log-target', 'log-fusion', 'log-corridor']:
        args.run = args.run.resolve()
        if not args.run.is_relative_to(PROJECT):
            parser.error('--run must be inside the project directory')
        args.func(args)
        return
    args.output = args.output.resolve()
    if not args.output.is_relative_to(PROJECT):
        parser.error('--output must be inside the project directory')
    if getattr(args, 'cache_dir', None) and not args.cache_dir.resolve().is_relative_to(PROJECT):
        parser.error('--cache-dir must be inside the project directory')
    if hasattr(args, 'source_run') and not args.source_run.resolve().is_relative_to(PROJECT):
        parser.error('--source-run must be inside the project directory')
    if hasattr(args, 'fit_start') and args.fit_start > args.fit_end:
        parser.error('fit-start must not follow fit-end')
    if hasattr(args, 'eval_start') and args.eval_start > args.eval_end:
        parser.error('eval-start must not follow eval-end')
    args.func(args)


def fit_only(args):
    import json
    from .data import save_json
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Fit output directory must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'func'}
    save_json(args.output / 'run_config.json', cfg)
    families = pipeline.panel_families(args.release_root)
    panels = list(families) if args.panels == ['all'] else args.panels
    for panel in panels:
        if panel not in families:
            raise ValueError(f'Unknown panel: {panel}')
        pipeline.fit_panel(args.release_root, panel, args.output / 'models' / panel, cfg)


if __name__ == '__main__':
    main()
