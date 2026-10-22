"""New-computer entry point; training code is the frozen original engine."""
import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from team_portable import portable_queue


def arguments(argv=None):
    parser = argparse.ArgumentParser(description='Portable six-direction T1/dynamic-mask queue with frozen four-direction reference tables')
    parser.add_argument('action', choices=['run', 'prepare', 'summary', 'log'])
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--release-root', type=Path)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--batch', choices=['all', 'i210', 'i10', 'i405'], default='all')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if (args.resume or args.action in ['summary', 'log']) and args.output is None:
        parser.error('--resume, summary and log require the original top-level --output')
    args.project = args.project.resolve()
    # The modules locate their immutable references relative to this code tree.
    if args.project != Path(__file__).resolve().parent:
        parser.error('--project must be this copied code directory; use --release-root for external data')
    args.release_root = (args.release_root or args.project / '2026-ieee-big-data-traffic-flow-bench/kaggle_public').resolve()
    args.cache_dir = (args.cache_dir or args.project / 'cache/transformer_cross_corridor').resolve()
    stamp = datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d_%H%M%S_%f')
    label = 'prepare' if args.action == 'prepare' else 'smoke' if args.smoke else 'dev'
    args.output = (args.output or args.project / 'runs' / f'team_remaining_{label}_{stamp}').resolve()
    args.batches = ['i210', 'i10', 'i405'] if args.batch == 'all' else [args.batch]
    return args


def main(argv=None):
    args = arguments(argv)
    queue = portable_queue(args.project)
    if args.action in ['summary', 'log']:
        queue.postprocess(args)
        return 0
    if not args.release_root.is_dir():
        raise FileNotFoundError('Official data missing: ' + str(args.release_root) + '; place kaggle_public here or pass --release-root')
    return queue.run_queue(args)


if __name__ == '__main__':
    raise SystemExit(main())
