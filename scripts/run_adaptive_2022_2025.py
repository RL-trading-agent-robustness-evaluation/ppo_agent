"""User-authorized five-seed direct known-period run and retained summary."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import statistics

ROOT=Path(__file__).resolve().parents[1]
ARM='ppo_v11_vix_mix'

def new_json(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as handle:
        json.dump(payload,handle,indent=2,allow_nan=False)

def summarize(tag):
    records=[]
    for seed in range(5):
        path=ROOT/'experiments/v11_ppo/records'/f'known_{ARM}_known_seed{seed}_{tag}.json'
        record=json.loads(path.read_text())
        if not record['technical_pass']:
            raise RuntimeError(f'seed {seed} failed technical checks')
        before=sum(u['realized_transitions'] for u in record['updates'] if u['cutoff']<='2021-12-31')
        after=sum(u['realized_transitions'] for u in record['updates'] if u['cutoff']>'2021-12-31')
        records.append(dict(seed=seed,metrics=record['metrics'],baselines=record['baselines'],
            training_transitions=before,testing_learning_transitions=after,
            testing_updates=[dict(cutoff=u['cutoff'],activation=u['activation_date'],reason=u['reason'])
                for u in record['updates'] if u['cutoff']>'2021-12-31'],
            reward_nav_error=record['reward_nav_error'],
            record=str(path.relative_to(ROOT)),record_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            responsiveness=record['responsiveness']))
    metric_names=('final_wealth','cagr','sharpe_excess_cash','max_drawdown','total_transaction_cost','total_turnover','average_exposure')
    medians={key:statistics.median(r['metrics'][key] for r in records if r['metrics'][key] is not None) for key in metric_names}
    buy_hold=records[0]['baselines']['buy_and_hold']['final_wealth']
    result=dict(status='PASS',label='user_requested_direct_adaptive_known_period_diagnostic',
        test_period=['2022-01-01','2025-12-31'],seeds=list(range(5)),arm=ARM,
        selection_matrix_completed=False,untouched_test=False,
        updates_allowed_in_testing_and_later_deployment=True,
        medians=medians,buy_and_hold_final_wealth=buy_hold,
        wins_vs_buy_and_hold=sum(r['metrics']['final_wealth']>buy_hold for r in records),
        records=records)
    target=ROOT/'reports/v11_ppo'/f'adaptive_2022_2025_{tag}.json'
    new_json(target,result)
    lines=['# V11 2022–2025 因果動態評估','',
        '**使用者直接授權之 adaptive known-period diagnostic；不是 untouched test，未使用該期間調參。**','',
        f"五 seeds 的 median final wealth：**{medians['final_wealth']:.6f}**；可執行 SPY buy-and-hold：**{buy_hold:.6f}**。",'',
        f"{result['wins_vs_buy_and_hold']}/5 seeds 勝過 buy-and-hold。所有 seeds 使用相同固定 reward、VIX 時鐘、200k 初始化與 32,768-step fine-tuning。",'',
        '| Seed | Final wealth | CAGR | Max drawdown | Trading costs | 2022–2025 updates |',
        '|---:|---:|---:|---:|---:|---:|']
    for row in records:
        m=row['metrics']
        lines.append(f"| {row['seed']} | {m['final_wealth']:.6f} | {m['cagr']:.2%} | {m['max_drawdown']:.2%} | ${m['total_transaction_cost']:,.2f} | {len(row['testing_updates'])} |")
    lines+=['','模型在 2010–2021 chronological prefix 上初始化並依 VIX 更新；測試帳戶於 2022 開始。測試中 action 先決定，訓練只使用截至該日已觀測資料，新模型從下一 close 決策生效。scaler 不 refit，績效只使用淨 ledger NAV。','',
        '每個 run 保留 `pre_2022_model.zip`、初始測試前 freeze record、逐次 updates、完整 scoring trajectory、final model 與 deployment checkpoint。後續 deployment 可繼續同一更新政策；需將新增市場資料連接至保存的 account/optimizer/clock，不能 reset 持倉或 scaler。','',
        '這是單一凍結方案的直接評估，尚未完成 45-cell development comparison；結果不能證明混合 reward 或動態排程的單獨因果效果。所有 seed 與 baseline 詳細資料見同名 JSON。']
    md=target.with_suffix('.md')
    with md.open('x',encoding='utf-8') as handle:
        handle.write('\n'.join(lines)+'\n')
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',default='adaptive_20261007_1')
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.workers not in (1,2,3):raise ValueError('workers must be 1–3')
    if not args.tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.tag):raise ValueError('unsafe tag')
    target=ROOT/'reports/v11_ppo'/f'adaptive_protocol_{args.tag}.json'
    new_json(target,dict(created_utc=datetime.now(timezone.utc).isoformat(),
        user_authorization='Run all five seeds on 2022–2025; dynamic updates allowed in testing and later deployment.',
        seeds=list(range(5)),arm=ARM,initialization_requested=200000,fine_tune_steps=32768,
        config_sha256=hashlib.sha256((ROOT/'configs/ppo_v11.yaml').read_bytes()).hexdigest(),
        source_hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (ROOT/'src/victim_ppo/v11').glob('*.py')},
        untouched_test=False,selection_matrix_completed=False))
    pending=list(range(5));running={};completed=[]
    while pending or running:
        while pending and len(running)<args.workers:
            seed=pending.pop(0)
            path=ROOT/'reports/v11_ppo/logs'/f'adaptive_seed{seed}_{args.tag}_console.log'
            path.parent.mkdir(parents=True,exist_ok=True)
            handle=path.open('x',encoding='utf-8')
            flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
            proc=subprocess.Popen([sys.executable,str(ROOT/'scripts/run_ppo_v11.py'),
                '--direct-known-period','--arm',ARM,'--seed',str(seed),'--tag',args.tag],
                cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,creationflags=flags)
            running[seed]=(proc,handle)
            print(f'Started seed {seed}, PID {proc.pid}',flush=True)
        for seed,(proc,handle) in list(running.items()):
            if proc.poll() is not None:
                handle.close();del running[seed]
                if proc.returncode:
                    print(f'Seed {seed} failed; preserving other in-flight runs.',flush=True)
                    pending=[]
                    for other,(child,stream) in running.items():
                        child.wait();stream.close()
                    raise RuntimeError(f'seed {seed} exit {proc.returncode}; see retained logs')
                completed.append(seed)
                print(f'Completed seed {seed}; finished seeds {completed}',flush=True)
        time.sleep(1)
    result=summarize(args.tag)
    print(json.dumps(dict(status=result['status'],medians=result['medians'],
                         wins_vs_buy_and_hold=result['wins_vs_buy_and_hold']),indent=2),flush=True)

if __name__=='__main__':main()
