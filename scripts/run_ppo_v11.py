#!/usr/bin/env python3
"""Runnable without installation; uses local V11 and copied shared environment."""
import argparse
import json
from pathlib import Path
import sys
from contextlib import redirect_stdout, redirect_stderr
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'victimagent/src')]
from victim_ppo.v11.config import ARMS, load_config
from victim_ppo.v11.runner import preflight,run_cell,aggregate

def main():
    parser=argparse.ArgumentParser(description='V11 200k initial training, causal adaptive evaluation')
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight',action='store_true')
    mode.add_argument('--smoke',action='store_true')
    mode.add_argument('--run',action='store_true')
    mode.add_argument('--known-period',action='store_true')
    mode.add_argument('--direct-known-period',action='store_true',
                      help='User-requested direct adaptive 2022-2025 diagnostic; no model selection claim')
    mode.add_argument('--aggregate',action='store_true')
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--fold',choices=['fold_1','fold_2','fold_3'])
    parser.add_argument('--seed',type=int,choices=range(5),default=0)
    parser.add_argument('--tag',default='v11_200k_1')
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    config=load_config(ROOT/'configs/ppo_v11.yaml')
    if args.preflight:
        result=preflight(ROOT,config)
    elif args.aggregate:
        result=aggregate(ROOT,args.tag)
    else:
        if not args.arm:
            parser.error('--arm required')
        if args.run and not args.fold:
            parser.error('--run requires --fold')
        if args.known_period:
            frozen=ROOT/'reports/v11_ppo'/f'{args.tag}_summary.json'
            if not frozen.exists():
                parser.error('known-period requires completed development summary')
            summary=json.loads(frozen.read_text())
            if summary['complete_cells']!=45:
                parser.error('known-period requires all 45 development records')
        log_directory=ROOT/'reports/v11_ppo/logs'
        log_directory.mkdir(parents=True,exist_ok=True)
        with (log_directory/f'run_{time.time_ns()}.log').open('x',encoding='utf-8') as log:
            with redirect_stdout(log),redirect_stderr(log):
                try:
                    result=run_cell(ROOT,config,args.arm,'known' if (args.known_period or args.direct_known_period) else (args.fold or 'fold_1'),
                                    args.seed,args.tag,smoke=args.smoke,resume=args.resume)
                    if args.direct_known_period:
                        # Separate user authorization/protocol note; do not relabel
                        # the ordinary development selection or any old records.
                        from victim_ppo.v11.runner import write_new
                        from victimagent.v2.registry import sha256_file
                        protocol=ROOT/'reports/v11_ppo'/f'direct_{args.arm}_seed{args.seed}_{args.tag}.json'
                        write_new(protocol,dict(label='user_requested_direct_adaptive_known_period_diagnostic',
                            selection_matrix_completed=False,adaptive_deployment_allowed=True,
                            result_label=result['label'],config_sha256=sha256_file(config.path)))
                except BaseException:
                    import traceback
                    traceback.print_exc()
                    raise
    # Full step/update diagnostics live in append-only records, not console.
    print(json.dumps({k:v for k,v in result.items() if k!='updates'},indent=2,default=str,allow_nan=False))

if __name__=='__main__':
    main()
