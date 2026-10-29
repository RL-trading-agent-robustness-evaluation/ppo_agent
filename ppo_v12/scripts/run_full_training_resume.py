"""Single supervisor: verify/smoke, five isolated V12 models, version comparison."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
ENGINE=ROOT/'training'
PREVIOUS=ROOT.parent/'ppo_v11'
OLD=ROOT.parent/'ppo_v10'
ARM='ppo_v12_vix_mix'
TAG='full500k_20261008_3'
OUT=ROOT/'reports/full_training'/TAG
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def new_json(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2,default=str,allow_nan=False)
def status(obj):
    p=OUT/'status.json'
    tmp=OUT/'status.tmp'
    tmp.write_text(json.dumps(dict(updated_utc=datetime.now(timezone.utc).isoformat(),**obj),indent=2),encoding='utf-8')
    tmp.replace(p)
def command(*args):
    return [sys.executable,'-B',str(ENGINE/'scripts/run_ppo_v11.py'),*args]
def launch(args,log):
    stream=log.open('x',encoding='utf-8')
    try:
        child=subprocess.Popen(args,cwd=ENGINE,stdout=stream,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    except BaseException:
        stream.close();raise
    return child,stream
def summarize():
    sys.path[:0]=[str(ENGINE/'src'),str(ENGINE/'victimagent/src')]
    from victimagent.v4.evaluation import metrics
    baseline_report=ROOT/'reports/v12_benchmark/comparison_2022_2025.json'
    b=json.loads(baseline_report.read_text(encoding='utf-8'))
    provenance={str(baseline_report):sha(baseline_report)}
    required=('final_wealth','cagr','max_drawdown','total_transaction_cost','average_exposure')
    models=[];pairs=[]
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    assert all(sha(Path(p))==h for p,h in protocol['source_hashes'].items())
    for seed in range(5):
        label=f'{ARM}_known_seed{seed}_{TAG}'
        record_path=ENGINE/f'experiments/v11_ppo/records/known_{label}.json'
        r=json.loads(record_path.read_text(encoding='utf-8'))
        assert r['technical_pass'] and r['status']=='PASS' and r['config_version']=='v12train1_500k'
        assert r['evaluation_learning']=='causal_adaptive' and r['seed']==seed
        old_path=PREVIOUS/f'experiments/v11_ppo/records/known_ppo_v11_vix_mix_known_seed{seed}_adaptive_20261007_1.json'
        previous=json.loads(old_path.read_text(encoding='utf-8'))
        assert previous['technical_pass'] and r['audit']['data_hashes']==previous['audit']['data_hashes']
        folder=ENGINE/f'artifacts/ppo_v12/known/{label}'
        ledger=folder/'validation.jsonl'
        assert sha(ledger)==r['trajectory_sha256'] and sha(folder/'final_model.zip')==r['final_checkpoint_sha256']
        rows=[json.loads(line) for line in ledger.read_text(encoding='utf-8').splitlines()]
        recomputed=metrics(rows)
        assert recomputed['steps']==1002 and recomputed['start_date']=='2022-01-03' and recomputed['end_date']=='2025-12-31'
        for key in required:
            assert math.isfinite(r['metrics'][key]) and math.isclose(recomputed[key],r['metrics'][key],rel_tol=1e-10,abs_tol=1e-8)
        old_ledger=PREVIOUS/f'artifacts/ppo_v11/known/ppo_v11_vix_mix_known_seed{seed}_adaptive_20261007_1/validation.jsonl'
        assert sha(old_ledger)==previous['trajectory_sha256']
        old_rows=[json.loads(line) for line in old_ledger.read_text(encoding='utf-8').splitlines()]
        assert [(x['decision_date'],x['valuation_date']) for x in rows]==[(x['decision_date'],x['valuation_date']) for x in old_rows]
        assert rows[0]['start_nav']==1e6 and abs(sum(x['reward'] for x in rows)-math.log(r['metrics']['final_wealth']))<1e-10
        assert all(math.isclose(rows[i]['start_nav'],rows[i-1]['end_nav'],rel_tol=1e-12,abs_tol=1e-6) for i in range(1,len(rows)))
        assert r['updates'][0]['requested_transitions']==500000 and r['updates'][0]['realized_transitions']==501760
        assert all(u['realized_transitions']==32768 for u in r['updates'][1:])
        assert len(r['updates'])==34 and r['total_transitions']==1583104
        assert all(math.isclose(r['baselines']['buy_and_hold'][key],b['benchmarks']['lump_sum'][key],rel_tol=1e-10,abs_tol=1e-8) for key in required)
        for u in r['updates']:
            stage=folder/f"update_{u['ordinal']:04d}"
            assert sha(stage/'model.zip')==u['checkpoint_sha256']
            assert sha(stage/'windows.csv')==u['windows_sha256']
            assert sha(stage/'training_steps.jsonl')==u['training_steps_sha256']
        deployment_folders=sorted(p for p in folder.glob('resume_deployment*') if p.is_dir())
        assert deployment_folders,'deployment state missing'
        models.append(dict(seed=seed,technical_pass=True,metrics=r['metrics'],record=str(record_path),
            final_checkpoint=str(folder/'final_model.zip'),deployment_state=str(deployment_folders[-1]),
            total_transitions=r['total_transitions'],updates=len(r['updates'])-1,
            seconds=r['seconds']))
        pairs.append(dict(seed=seed,delta_log_wealth=math.log(r['metrics']['final_wealth']/previous['metrics']['final_wealth']),
            differences={key:r['metrics'][key]-previous['metrics'][key] for key in required}))
        for p in (record_path,old_path,ledger,folder/'final_model.zip'):
            provenance[str(p)]=sha(p)
    historical_path=OLD/'reports/v10_ppo/comparison_2022_2025/COMPARISON.json'
    h=json.loads(historical_path.read_text(encoding='utf-8'))
    assert h['status']=='PASS' and h['evaluation_window']==dict(start='2022-01-03',end='2025-12-31',steps=1002)
    historical={}
    for name,records in h['per_seed'].items():
        assert len(records)==5 and sorted(r['seed'] for r in records)==list(range(5))
        for r in records:
            assert r.get('technical_pass',r.get('status')=='PASS')
            assert all(math.isfinite(r[k]) for k in required)
            p=OLD/r['trajectory_path']
            assert sha(p)==r['trajectory_sha256']
            provenance[str(p)]=sha(p)
        historical[name]=dict(records=records,medians={k:statistics.median(r[k] for r in records) for k in required})
    provenance[str(historical_path)]=sha(historical_path)
    medians={k:statistics.median(m['metrics'][k] for m in models) for k in required}
    wins={name:sum(m['metrics']['final_wealth']>bm['final_wealth'] for m in models) for name,bm in b['benchmarks'].items()}
    result=dict(status='PASS',technical_pass=True,label='adaptive known-period training-budget comparison',
        formal_adoption='NOT_EVALUATED',untouched_test=False,seeds=list(range(5)),
        initial_requested=500000,initial_actual=501760,fine_tune=32768,
        total_transitions=5*1583104,models=models,medians=medians,benchmarks=b['benchmarks'],
        wins_vs_baselines=wins,v11_200k=b['models'],v11_medians=b['medians'],paired_v11_differences=pairs,
        paired_v11_positive=sum(p['delta_log_wealth']>0 for p in pairs),
        median_paired_log_wealth=statistics.median(p['delta_log_wealth'] for p in pairs),
        historical=historical,historical_caveats=h['comparison_caveats'],source_hashes=provenance)
    new_json(OUT/'COMPARISON.json',result)
    lines=['# V12 500k 完整五模型：2022–2025 版本比較','',
        '**adaptive known-period training-budget comparison；不是 untouched test。正式採用 NOT_EVALUATED。**','',
        '每 seed 初始化 500,000 requested / 501,760 actual，33 次 32,768-step 更新，全程 1,583,104 transitions；沒有依已知期間挑 seed 或 checkpoint。','',
        '| 策略／版本 | Wealth | CAGR | MDD | 成本 USD | 平均曝險 |',
        '|---|---:|---:|---:|---:|---:|']
    def line(name,m): return f"| {name} | {m['final_wealth']:.7f} | {m['cagr']:.4%} | {m['max_drawdown']:.4%} | {m['total_transaction_cost']:,.2f} | {m['average_exposure']:.2%} |"
    lines += [line('定額 B&H',b['benchmarks']['monthly']),line('一次投入 B&H',b['benchmarks']['lump_sum'])]
    lines += [line(f"V12 500k seed {m['seed']}",m['metrics']) for m in models]
    lines += [line('V12 500k 中位數',medians),line('V11 200k 中位數',b['medians'])]
    lines += [line(name+' 中位數',row['medians']) for name,row in historical.items()]
    lines += ['',f"新模型勝月買 B&H {wins['monthly']}/5；勝一次投入 B&H {wins['lump_sum']}/5；勝同 seed V11 {result['paired_v11_positive']}/5。",'',
        '| Seed | 相對 V11 log-wealth 差 | Wealth 差 | CAGR 差 pp | MDD 差 pp | 成本差 USD |',
        '|---|---:|---:|---:|---:|---:|']
    for p in pairs:
        d=p['differences'];lines.append(f"| {p['seed']} | {p['delta_log_wealth']:+.6f} | {d['final_wealth']:+.6f} | {100*d['cagr']:+.4f} | {100*d['max_drawdown']:+.4f} | {d['total_transaction_cost']:+,.2f} |")
    lines += ['', 'MDD 差為新模型減旧模型，正值代表較小回撤。各版本與每種子原始結果、基準差異及 provenance 詳見 JSON。', '',
        'V11 對照使用同 seeds、reward／更新政策／初始資金／市場資料與估值日曆；改變的是初始化步數與學習率 schedule 的持續時間。五 paired 結果不等於正式三fold採用通過。', '',
        '舊版本限制：V4 的 features、資料 hash、訓練 cutoff 與執行契約不同；V9 訓練至2018、V10是訓練至2014的pilot；舊版本為 frozen evaluation，V11/V12為因果adaptive。只能做描述比較，不能當成單一算法效果。', '',
        '月買採 fractional shares，其他基準與RL採整股；月買平均曝險較低，回撤／收益不能只歸因於擇時。2022–2025 已開啟，不據此調參或選最佳模型。', '',
        '部署 checkpoint 保存帳戶、optimizer、scaler及更新時鐘；本次不自動部署。全部 seeds 都交付，不指定最佳 seed。', '',
        '[完整結果及每個模型路徑](COMPARISON.json)；[凍結 protocol](protocol.json)。']
    with (OUT/'COMPARISON.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    return result
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--summarize',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.summarize:
        print(json.dumps(summarize()['medians']));return
    assert not (OUT/'protocol.json').exists(),'supervisor already started: refuse duplicate'
    tracked=[ENGINE/'configs/ppo_v11.yaml',ENGINE/'SOURCE_MANIFEST.json',ENGINE/'ENGINE_CHANGES.json',
        ROOT/'docs/PPO_V12_FULL_TRAINING_ADDENDUM_ZH_HANT.md',Path(__file__)]
    tracked += list((ENGINE/'src/victim_ppo/v11').glob('*.py'))
    new_json(OUT/'protocol.json',dict(user_authorization='User authorized full training, additional seeds/timesteps, and direct comparisons to new B&H and previous RL versions.',
        tag=TAG,supervisor_pid=os.getpid(),created_utc=datetime.now(timezone.utc).isoformat(),
        seeds=list(range(5)),initial_requested=500000,initial_actual=501760,fine_tune=32768,
        expected_total_per_seed=1583104,workers=2,arm=ARM,adaptive_testing=True,
        source_hashes={str(p):sha(p) for p in tracked},formal_selection_matrix=False))
    smoke_evidence=[]
    for arm in ('ppo_v12_vix_nav',ARM,'ppo_v12_fixed_nav'):
        path=ENGINE/f'experiments/v11_ppo/records/smoke_{arm}_fold_1_seed0_full500k_20261008_1.json'
        record=json.loads(path.read_text(encoding='utf-8'))
        assert record['technical_pass'] and record['config_version']=='v12train1_500k'
        assert record['audit']['config_sha256']==sha(ENGINE/'configs/ppo_v11.yaml')
        for rel,h in record['audit']['implementation_hashes'].items():
            assert sha(ENGINE/rel)==h
        smoke_evidence.append(dict(record=str(path),sha256=sha(path)))
    new_json(OUT/'RESTART.json',dict(authorization='User explicitly authorized resuming training after the scheduled user-requested pause; no saved initialization checkpoint exists, so initialization restarts.',
        interrupted_tag='full500k_20261008_2',fresh_tag=TAG,
        unchanged_initial_requested=500000,unchanged_fine_tune=32768,
        reused_passed_smokes=smoke_evidence,preflight_rerun=True,
        reason='Resume authorized after user pause; no recoverable initialization checkpoint, preserve both earlier attempts and all partial artifacts.'))
    running={};completed=[];failure=None
    try:
        for name,cmd in [('preflight',command('--preflight'))]:
            status(dict(phase='preflight_and_smoke',current=name,supervisor_pid=os.getpid()))
            child,stream=launch(cmd,OUT/f'{name}.log')
            code=child.wait();stream.close()
            if code: raise RuntimeError(f'{name} failed with {code}; retained log')
        pending=list(range(5))
        while pending or running:
            while pending and len(running)<2 and not failure:
                seed=pending.pop(0)
                child,stream=launch(command('--direct-known-period','--arm',ARM,'--seed',str(seed),'--tag',TAG),OUT/f'seed{seed}.log')
                running[seed]=(child,stream)
                print(f'Started seed {seed} PID {child.pid}',flush=True)
            for seed,(child,stream) in list(running.items()):
                code=child.poll()
                if code is not None:
                    stream.close();del running[seed]
                    if code:
                        failure=f'seed {seed} exit {code}';pending=[]
                    else:completed.append(seed)
            status(dict(phase='training',supervisor_pid=os.getpid(),completed=completed,pending=pending,
                running={seed:child.pid for seed,(child,_) in running.items()},failure=failure))
            time.sleep(5)
        if failure:raise RuntimeError(failure)
        status(dict(phase='verifying_and_summarizing',completed=completed))
        r=summarize()
        status(dict(phase='complete',technical_pass=True,completed=completed,medians=r['medians'],wins=r['wins_vs_baselines']))
        with (ROOT/'dev/devlog.md').open('a',encoding='utf-8') as f:
            f.write('\n\n## V12 500k 完整五模型訓練完成\n\n'+
                f"- 全部 seeds 0–4 technical_pass；各 1,583,104 transitions。比較：reports/full_training/{TAG}/COMPARISON.md。\n"+
                f"- 中位數 {json.dumps(r['medians'])}；基準勝數 {json.dumps(r['wins_vs_baselines'])}；同 seed V11 正差 {r['paired_v11_positive']}/5。\n"+
                '- adaptive known-period budget comparison，正式採用 NOT_EVALUATED；不自動部署。\n')
    except BaseException:
        new_json(OUT/f'failure_{time.time_ns()}.json',dict(traceback=traceback.format_exc(),completed=completed,
            running={seed:child.pid for seed,(child,_) in running.items()}))
        status(dict(phase='failed',completed=completed,failure=traceback.format_exc()))
        raise
if __name__=='__main__':main()
