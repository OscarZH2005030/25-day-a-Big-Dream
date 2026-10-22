import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from .common import PROJECT,PILOT,inside


def main():
    parser=argparse.ArgumentParser(description='Paired two-corridor Transformer coverage/reliability/delay experiments')
    parser.add_argument('action',choices=['run','prepare','log'])
    parser.add_argument('--release-root',default=str(PROJECT/'2026-ieee-big-data-traffic-flow-bench/kaggle_public'))
    parser.add_argument('--cache-dir',default=str(PROJECT/'cache/transformer_priority'))
    parser.add_argument('--output')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--panels',nargs='+',default=PILOT)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.output is None:
        args.output=str(PROJECT/'runs'/('transformer_priorities_'+('smoke_' if args.smoke else 'pilot_')+datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d_%H%M%S_%f')))
    from .runner import run,log_results
    if args.action=='log':
        print(log_results(inside(args.output)))
    else:
        run(args)


if __name__=='__main__':
    main()
