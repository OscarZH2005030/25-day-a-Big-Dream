import argparse
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import sys


def main():
    parser=argparse.ArgumentParser(description='Two-direction T1 dynamic-mask/regime training ablations')
    parser.add_argument('action',choices=['run','prepare','log'])
    parser.add_argument('--project',default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--source-run');parser.add_argument('--fusion-run');parser.add_argument('--cache-dir');parser.add_argument('--output')
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();args.project=Path(args.project).resolve();sys.path.insert(0,str(args.project))
    for field,default in [('source_run','runs/transformer_priorities_pilot_01'),
                          ('fusion_run','runs/transformer_fusion_dev_20261019_174141_317748'),('cache_dir','cache/transformer_training')]:
        setattr(args,field,Path(getattr(args,field)).resolve() if getattr(args,field) else args.project/default)
    stamp=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d_%H%M%S_%f')
    args.output=Path(args.output).resolve() if args.output else args.project/'runs'/('transformer_training_'+('smoke_' if args.smoke else 'dev_')+stamp)
    from .runner import run,log_results
    if args.action=='log': print(log_results(args.output,args.project))
    else: run(args)


if __name__=='__main__': main()
