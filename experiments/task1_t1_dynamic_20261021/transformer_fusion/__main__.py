from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import sys


def main():
    parser = argparse.ArgumentParser(description='Prediction-only T0/T1/LightGBM fusion; never trains models')
    parser.add_argument('action', choices=['run', 'prepare', 'log'])
    parser.add_argument('--project', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--source-run')
    parser.add_argument('--cache-dir')
    parser.add_argument('--output')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.project = Path(args.project).resolve()
    sys.path.insert(0, str(args.project))
    args.source_run = Path(args.source_run).resolve() if args.source_run else args.project/'runs/transformer_priorities_pilot_01'
    args.cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else args.project/'cache/transformer_fusion'
    if args.output is None:
        stamp = datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d_%H%M%S_%f')
        args.output = args.project/'runs'/('transformer_fusion_'+('smoke_' if args.smoke else 'dev_')+stamp)
    args.output = Path(args.output).resolve()
    from .runner import run, log_results
    if args.action == 'log':
        print(log_results(args.output, args.project))
    else:
        run(args)


if __name__ == '__main__':
    main()
