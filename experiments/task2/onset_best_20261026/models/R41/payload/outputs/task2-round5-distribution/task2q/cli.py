"""Bounded engineering commands. There is deliberately no formal-train action."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from .protocol import write_frozen_split

PROJECT = Path(__file__).resolve().parents[1]


def local_output(value, label):
    path = Path(value).resolve()
    if not path.is_relative_to(PROJECT) or path == PROJECT:
        raise ValueError(label + ' must be inside this separate Task 2 project')
    for relative in ['task2q', 'tests', 'verification']:
        protected = PROJECT / relative
        if path == protected or path.is_relative_to(protected):
            raise ValueError('Cannot write outputs into source/verification directories')
    return path


def arguments(argv=None):
    parser = argparse.ArgumentParser(description='Task 2 audit, proxy preparation and bounded smoke only')
    parser.add_argument('action', choices=['audit', 'prepare', 'smoke'])
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, default=PROJECT / 'cache')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--panels', nargs='+')
    parser.add_argument('--max-days', type=int, default=2)
    parser.add_argument('--max-windows', type=int, default=8)
    parser.add_argument('--stride', type=int, default=6, help='Five-minute slots; 6 = every 30 minutes')
    parser.add_argument('--scope', choices=['engineering', 'full-inputs'], default='engineering')
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    args.release_root = args.release_root.resolve()
    if not args.release_root.is_dir() or not (args.release_root / 'task2').is_dir():
        parser.error('--release-root must be the official kaggle_public directory')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    args.output = local_output(args.output or PROJECT / 'runs' / f'{args.action}_{stamp}', 'Output')
    args.cache_dir = local_output(args.cache_dir, 'Cache')
    if args.output.is_relative_to(args.cache_dir) or args.cache_dir.is_relative_to(args.output):
        parser.error('Cache and output directories must be separate')
    for dest in [args.output, args.cache_dir]:
        if dest.is_relative_to(args.release_root) or args.release_root.is_relative_to(dest):
            parser.error('Cannot write inside the released data or its ancestors')
    if args.action != 'prepare' and args.scope != 'engineering':
        parser.error('full-inputs is only a data preparation scope, never training')
    if args.action == 'smoke' and not (1 <= args.steps <= 4 and 1 <= args.max_days <= 2 and 1 <= args.max_windows <= 8):
        parser.error('Smoke is capped at 4 updates, 2 days/role, 8 windows/panel/role')
    if not 1 <= args.threads <= 8 or args.stride < 1:
        parser.error('Use 1..8 threads and a positive stride')
    if args.scope == 'engineering' and (args.max_days < 1 or args.max_windows < 1):
        parser.error('Engineering sample limits must be positive')
    return args


def main(argv=None):
    args = arguments(argv)
    import os
    os.environ.setdefault('OMP_NUM_THREADS', str(args.threads))
    os.environ.setdefault('MKL_NUM_THREADS', str(args.threads))
    import pyarrow
    pyarrow.set_cpu_count(args.threads)
    pyarrow.set_io_thread_count(min(args.threads, 4))
    from .data import PANELS, prepare
    from .audit import audit_release
    panels = args.panels or (['D7_I210_E', 'D7_I405_S'] if args.action == 'smoke' else list(PANELS))
    if len(panels) != len(set(panels)) or set(panels) - set(PANELS):
        raise ValueError('Unknown or repeated Task 2 directional panel')
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('Use a fresh run output; existing content is never overwritten')
    write_frozen_split(args.output / 'frozen_split.json')
    (args.output / 'command.json').write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2) + '\n', encoding='utf-8')
    if args.action == 'audit':
        report = audit_release(args.release_root, output_path=args.output / 'audit.json', panels=panels)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report['passed'] else 1
    if args.action == 'prepare':
        full = args.scope == 'full-inputs'
        result = prepare(args.release_root, args.cache_dir, args.output / 'dataset', panels=panels,
                         roles=('fit', 'tune'), max_days=None if full else args.max_days,
                         max_windows=None if full else args.max_windows, stride=args.stride,
                         engineering=not full)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    from .smoke import run_smoke
    report = run_smoke(args, panels)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
