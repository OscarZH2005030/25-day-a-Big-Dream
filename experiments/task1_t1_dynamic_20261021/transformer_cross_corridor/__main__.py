import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import sys


def main():
    parser = argparse.ArgumentParser(description='D12 I5 T1 versus frozen dynamic missingness replication')
    parser.add_argument('action', choices=['run', 'prepare', 'log'])
    parser.add_argument('--project', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--release-root')
    parser.add_argument('--cache-dir')
    parser.add_argument('--output')
    parser.add_argument('--fusion-mode', choices=['rebuild_t0', 'tree_only'], required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.project = Path(args.project).resolve()
    sys.path.insert(0, str(args.project))
    args.release_root = Path(args.release_root).resolve() if args.release_root else args.project / '2026-ieee-big-data-traffic-flow-bench/kaggle_public'
    args.cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else args.project / 'cache/transformer_cross_corridor'
    stamp = datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d_%H%M%S_%f')
    args.output = Path(args.output).resolve() if args.output else args.project / 'runs' / ('d12_i5_dynamic_' + ('smoke_' if args.smoke else 'dev_') + stamp)
    from .runner import run, log_results
    if args.action == 'log':
        print(log_results(args.output, args.project))
    else:
        run(args)


if __name__ == '__main__':
    main()
