"""Read-only V11 comparison; computes two benchmarks and no model actions."""
import json
import math
import hashlib
import statistics
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent/'ppo_v11'
for path in (ROOT/'src', PARENT/'src', PARENT/'victimagent/src'):
    sys.path.insert(0, str(path))
from victim_ppo.v11.config import load_config
from victim_ppo.v11.data import MarketData
from victimagent.v8.study import _environment
from victimagent.v4.evaluation import rollout, metrics
from v12_benchmark import run_monthly

TAG = 'adaptive_20261007_1'
VIX_SHA256 = '675f9bcb86ee88799f8f7a8b7ee9c30f7a0be26a26df8f9549bddf9e84c4a598'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=str, allow_nan=False)

def jsonl(path, rows):
    with path.open('x', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str, allow_nan=False)+'\n')

def close(a,b):
    return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8)

def run():
    report = ROOT/'reports/v12_benchmark/comparison_2022_2025.json'
    if report.exists():
        raise FileExistsError('completed report exists; preserve it')
    source = {}
    frozen = json.loads((ROOT/'configs/benchmark_v12.json').read_text(encoding='utf-8'))
    expected = dict(version='v12spec1_benchmark_only',initial_cash=1000000.,start='2022-01-01',
        end='2025-12-31',purchase_months=48,cost_bps=10.,budget_includes_cost=True,
        share_mode='fractional',external_deposits=False,source_tag=TAG,
        existing_seeds=[0,1,2,3,4],training_authorized=False,
        future_initial_requested_transitions=200000,future_initial_expected_transitions=200704,
        future_fine_tune_transitions=32768,future_seeds=[0,1,2,3])
    if frozen != expected:
        raise ValueError('frozen V12 config changed')
    records, ledger_rows = [], []
    config = load_config(PARENT/'configs/ppo_v11.yaml')
    expected_data = {key:config.raw[key+'_sha256'] for key in ('market','events')}
    expected_data['vix'] = VIX_SHA256
    for key in ('market','events','vix'):
        p = PARENT/config.raw[key]
        source[str(p)] = sha(p)
        assert source[str(p)] == expected_data[key], key
    for seed in range(5):
        label = f'ppo_v11_vix_mix_known_seed{seed}_{TAG}'
        p = PARENT/f'experiments/v11_ppo/records/known_{label}.json'
        record = json.loads(p.read_text(encoding='utf-8'))
        assert record['technical_pass'] and record['status']=='PASS'
        assert record['seed']==seed and record['evaluation_learning']=='causal_adaptive'
        for key in ('market','events','vix'):
            assert record['audit']['data_hashes'][key] == expected_data[key]
        artifact = PARENT/f'artifacts/ppo_v11/known/{label}'
        trajectory = artifact/'validation.jsonl'
        assert sha(trajectory) == record['trajectory_sha256']
        assert sha(artifact/'final_model.zip') == record['final_checkpoint_sha256']
        rows = [json.loads(line) for line in trajectory.read_text(encoding='utf-8').splitlines()]
        recomputed = metrics(rows)
        for key in ('steps','final_wealth','cagr','max_drawdown','total_transaction_cost'):
            assert close(recomputed[key],record['metrics'][key]), (seed,key)
        assert rows[0]['start_nav']==1_000_000.
        assert all(math.isfinite(r['end_nav']) and r['end_nav']>0 for r in rows)
        assert all(close(rows[i]['start_nav'],rows[i-1]['end_nav']) for i in range(1,len(rows)))
        assert abs(sum(r['reward'] for r in rows)-math.log(recomputed['final_wealth']))<1e-10
        for p in (p,trajectory,artifact/'final_model.zip',artifact/'pre_2022_model.zip'):
            source[str(p)] = sha(p)
        records.append(record)
        ledger_rows.append(rows)
    data = MarketData(PARENT,config,end='2025-12-31')
    bundle,_ = data.bundle('2025-12-31',start='2022-01-01')
    lump_rows,lump,_ = rollout(_environment(bundle,wrap=False),lambda o,i: 2 if i==0 else None)
    monthly_rows,purchases,monthly = run_monthly(bundle)
    assert len(purchases)==48 and close(monthly['principal_budget_spent'],1_000_000.)
    assert purchases[0]['execution_date']=='2022-01-04'
    assert monthly['number_of_trades']==48 and close(monthly['total_transaction_cost'],1_000_000*.001/1.001)
    calendar = [(r['decision_date'],r['execution_date'],r['valuation_date']) for r in monthly_rows]
    for record,rows in zip(records,ledger_rows):
        assert calendar==[(r['decision_date'],r['execution_date'],r['valuation_date']) for r in rows]
        for key in ('steps','final_wealth','cagr','max_drawdown','total_transaction_cost'):
            assert close(lump[key],record['baselines']['buy_and_hold'][key]),key
    for row in monthly_rows:
        assert row['cash_available']>=0 and close(row['end_nav'],row['cash_available']+
            row['shares']*row['true_valuation_close']+row['dividend_receivable'])
        assert close(row['trade']['cash_pre_trade']-row['trade']['cash_post_trade'],row['purchase_budget'])
    out = ROOT/'reports/v12_benchmark'
    jsonl(out/'monthly_ledger.jsonl',monthly_rows)
    jsonl(out/'lump_sum_ledger.jsonl',lump_rows)
    write(out/'monthly_purchases.json',purchases)
    write(out/'baseline_metrics.json',dict(monthly=monthly,lump_sum=lump))
    keys = ('final_wealth','cagr','max_drawdown','total_transaction_cost','average_exposure')
    models=[]
    for record in records:
        m = record['metrics']
        models.append(dict(seed=record['seed'],technical_pass=True,
            metrics={k:m[k] for k in keys},final_nav=m['final_wealth']*1_000_000,
            differences={name:{k:m[k]-b[k] for k in keys} for name,b in [('monthly',monthly),('lump_sum',lump)]}))
    medians={k:statistics.median(m['metrics'][k] for m in models) for k in keys}
    wins={name:sum(m['metrics']['final_wealth']>b['final_wealth'] for m in models)
        for name,b in [('monthly',monthly),('lump_sum',lump)]}
    for path in (PARENT/'configs/ppo_v11.yaml', PARENT/'SOURCE_MANIFEST.json'):
        if path.exists(): source[str(path)]=sha(path)
    for base in (ROOT/'src',ROOT/'scripts',ROOT/'tests',ROOT/'configs',ROOT/'docs',
                 PARENT/'src',PARENT/'victimagent/src'):
        for p in base.rglob('*'):
            if p.is_file() and p.suffix in ('.py','.md','.json','.yaml'):
                source[str(p)]=sha(p)
    # Confirm the source artifacts were not altered during execution.
    assert all(sha(Path(p))==h for p,h in source.items())
    output_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    result=dict(version='v12spec1_benchmark_only',status='PASS',technical_pass=True,
        created_utc=datetime.now(timezone.utc).isoformat(),
        label='adaptive known-period diagnostic',new_model_runs=0,new_training_transitions=0,
        source_tag=TAG,benchmarks={'monthly':monthly,'lump_sum':lump},models=models,
        medians=medians,wins=wins,source_hashes=source,output_hashes=output_hashes,
        checks={'data_hashes':True,'five_records_technical_pass':True,'checkpoint_hashes':True,
            'trajectory_hashes':True,'recomputed_metrics':True,'identical_valuation_calendar':True,
            'lump_sum_reproduction':True,'48_budgets_and_1000000_total':True,
            'cash_nonnegative_and_nav_conservation':True,'reward_nav_identity':True},
        limitation='Monthly fractional shares versus original integer shares; different exposure schedules. No training improvement claim or formal adoption.')
    write(report,result)
    lines=['# V12：2022–2025 定額 B&H 與既有 V11 比較','',
        '**adaptive known-period diagnostic；沒有重新訓練、載入或執行模型。**',
        '', '五個既有種子均通過技術檢查；原 ledger 重算指標、資料及 checkpoint hashes、估值日曆皆一致。',
        '', '| 策略 | Final wealth | CAGR | 最大回撤 | 成本 USD | 平均曝險 |',
        '|---|---:|---:|---:|---:|---:|']
    def line(name,m):
        return f"| {name} | {m['final_wealth']:.7f} | {m['cagr']:.4%} | {m['max_drawdown']:.4%} | {m['total_transaction_cost']:,.2f} | {m['average_exposure']:.2%} |"
    lines += [line('每月定額 B&H',monthly),line('原 lump-sum B&H',lump)]
    lines += [line(f"V11 seed {m['seed']}",m['metrics']) for m in models]
    lines += [line('V11 五種子中位數（逐指標）',medians),'',
        f"模型勝出：相對定額 B&H **{wins['monthly']}/5**；相對原 B&H **{wins['lump_sum']}/5**。",'',
        '| Seed | Wealth 差：定額 | Wealth 差：原 B&H | CAGR 差：定額 pp | CAGR 差：原 B&H pp | MDD 差：定額 pp | MDD 差：原 B&H pp | 成本差：定額 USD | 成本差：原 B&H USD |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in models:
        a,b=m['differences']['monthly'],m['differences']['lump_sum']
        lines.append(f"| {m['seed']} | {a['final_wealth']:+.7f} | {b['final_wealth']:+.7f} | {100*a['cagr']:+.4f} | {100*b['cagr']:+.4f} | {100*a['max_drawdown']:+.4f} | {100*b['max_drawdown']:+.4f} | {a['total_transaction_cost']:+,.2f} | {b['total_transaction_cost']:+,.2f} |")
    lines += ['', 'MDD 是負值；差值為模型減基準，正值代表較小的回撤。Wealth 單位為初始資金倍數。', '',
        '比較期間：2022-01-03 初始收盤至 2025-12-31 收盤，共 1,002 transitions；首筆月買 2022-01-04，最後一期 2025-12-01。',
        f"48 次含成本本金支出合計 ${monthly['principal_budget_spent']:,.2f}；利息 ${monthly['total_cash_interest']:,.2f}、股息 ${monthly['total_dividends']:,.2f} 保留現金。終值 ${monthly['final_nav']:,.2f}；現金 ${monthly['final_cash']:,.2f}；應收股息 ${monthly['final_receivable']:,.2f}。",'',
        '定額基準使用 fractional shares，原模型及 lump-sum B&H 使用整股，這是已登記的執行差異。定額投入降低早期曝險；報酬與回撤差異不能單獨歸因於擇時能力。未新增終點賣出。', '',
        '本次只測量既有模型相對新基準的表現，不能證明模型改善，也不能通過三 folds × 五 seeds 的正式採用門檻。未調參或啟動未來四種子訓練；是否保留此基準或進一步訓練由使用者決定。','',
        '研究支持：定額投資的固定金額原理見 [Investor.gov](https://www.investor.gov/introduction-investing/investing-basics/glossary/dollar-cost-averaging)；持有現金的機會成本與風險取捨見 [Vanguard 2023](https://corporate.vanguard.com/content/dam/corp/research/pdf/cost_averaging_invest_now_or_temporarily_hold_your_cash.pdf)。兩者不證明 48 個月最佳。', '',
        '完整數值與 provenance：[JSON](comparison_2022_2025.json)；[月買明細](monthly_purchases.json)；[定額 ledger](monthly_ledger.jsonl)；[原 B&H ledger](lump_sum_ledger.jsonl)。']
    (out/'comparison_2022_2025.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='PASS',monthly=monthly,medians=medians,wins=wins),allow_nan=False))

if __name__=='__main__':
    out=ROOT/'reports/v12_benchmark'
    out.mkdir(parents=True,exist_ok=True)
    attempt=out/f'attempt_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")}.json'
    try:
        run()
        write(attempt,dict(status='PASS'))
    except Exception:
        write(attempt,dict(status='FAIL',traceback=traceback.format_exc()))
        raise
