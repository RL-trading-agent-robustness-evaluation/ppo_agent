"""Summarize existing V9 RGN results using the user's comparison-table criteria."""
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent/'reports'/'RGN_ATTACK_REPORT_CRITERIA_V1'
ARM = 'v9_gate_relative_reward'


def load(relative):
    path = ROOT/'outputs'/relative
    return json.loads(path.read_text(encoding='utf-8')), {
        'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def summary(clean, attacked, bh):
    assert len(clean)==len(attacked) and all(w>0 for w in clean+attacked)
    paired = [100*(a/c-1) for c,a in zip(clean,attacked)]
    return {'median_final_wealth':median(attacked),
            'median_paired_relative_wealth_change_pct':median(paired),
            'harmed_seeds':sum(a<c-1e-12 for c,a in zip(clean,attacked)),
            'unchanged_seeds':sum(abs(a-c)<=1e-12 for c,a in zip(clean,attacked)),
            'improved_seeds':sum(a>c+1e-12 for c,a in zip(clean,attacked)),
            'wins_vs_buy_and_hold':sum(a>bh for a in attacked), 'denominator':len(clean)}


def table(lines, fold, cells):
    clean = [r['clean_wealth'] for r in cells]
    bh = cells[0]['buy_and_hold']
    lines += ['| 版本 | 期末財富中位數 | 相對未攻擊財富變化¹ | 受傷 seeds² | 贏 B&H |',
              '|---|---:|---:|---:|---:|',
              f'| SPY B&H | {bh:.4f} 倍 | — | — | — |',
              f'| PPO V9 未攻擊 | {median(clean):.4f} 倍 | — | — | {sum(w>bh for w in clean)}/5 |']
    outputs=[]
    for sigma in (0.05,0.1,0.2):
        attacked=[r['attacked_wealth'][str(sigma)] for r in cells]
        s=summary(clean,attacked,bh)
        outputs.append({'fold':fold,'sigma':sigma,'noise_seed':123,**s})
        lines.append(f"| RGN σ={sigma}，noise seed 123 | {s['median_final_wealth']:.4f} 倍 | {s['median_paired_relative_wealth_change_pct']:+.2f}% | {s['harmed_seeds']}/5 | {s['wins_vs_buy_and_hold']}/5 |")
    return outputs


def main():
    first,p1=load('v9_gaussian_20261019_02/COMPARISON.json')
    repeated,p2=load('v9_gaussian_10noise_20261019_01/COMPARISON.json')
    sweep,p3=load('v9_small_sigma_20261029_01/COMPARISON.json')
    trace_path=ROOT/'outputs/v9_pipeline_trace_20261020_02/TRACE_REPORT.json'
    trace=json.loads(trace_path.read_text(encoding='utf-8'))
    sources=[p1,p2,p3,{'path':str(trace_path),'sha256':hashlib.sha256(trace_path.read_bytes()).hexdigest()}]
    assert all(r.get('status')=='COMPLETE' for r in (first,repeated,sweep,trace))
    assert all(r['clean']['technical_pass'] and r['perturbed']['technical_pass'] for r in repeated['cells'])
    assert all(r['clean']['technical_pass'] and r['noisy']['technical_pass'] for r in sweep['cells'])
    folds={}
    for fold in ('fold_1','fold_2','fold_3'):
        rows=[]
        for seed in range(5):
            original=next(r for r in first['cells'] if r['arm']==ARM and r['fold']==fold and r['seed']==seed)
            attacked={'0.05':original['perturbed']['metrics']['final_wealth']}
            for sigma in (0.1,0.2):
                r=next(r for r in sweep['cells'] if r['fold']==fold and r['model_seed']==seed and r['sigma']==sigma)
                assert r['checkpoint_sha256']==original['checkpoint_sha256']
                assert r['clean']['metrics']['final_wealth']==original['clean']['metrics']['final_wealth']
                attacked[str(sigma)]=r['noisy']['metrics']['final_wealth']
            rows.append({'model_seed':seed,'checkpoint_sha256':original['checkpoint_sha256'],
                         'clean_wealth':original['clean']['metrics']['final_wealth'],
                         'buy_and_hold':original['buy_and_hold'],'attacked_wealth':attacked})
        folds[fold]=rows
    lines=['# PPO V9：隨機高斯噪音（RGN）攻擊報告', '',
        '## 主要比較：2019–2021', '',
        '依使用者提供表格的四項標準重整既有結果：期末財富中位數、相對未攻擊財富變化、受傷 seeds、贏 B&H。',
        '使用同一批 **5 個凍結 PPO V9 模型 seeds 0–4**。圖例的 15 個 DQN seeds 並非本次樣本，不能套用其分母。',
        '模型組別為 `v9_gate_relative_reward`；各噪音強度使用相同 noise seed 123，便於配對比較。沒有新訓練或挑選最佳／最差噪音結果。','']
    grouped=table(lines,'fold_3',folds['fold_3'])
    lines+=['','¹ 先對每個模型計算 `100 × (W_RGN / W_clean − 1)`，再取五個模型的中位數。',
        '不是「兩組財富中位數的比值」，也不是總報酬率的百分點差。若為 +0.00%，不代表每個模型均無變化。',
        '² 受傷表示同一模型的攻擊後期末財富較未攻擊低（數值容差 1e-12）；不代表統計顯著。', '',
        '### 2019–2021：逐模型財富（倍）','',
        '| 模型 seed | 未攻擊 | σ=0.05 | σ=0.1 | σ=0.2 |','|---|---:|---:|---:|---:|']
    for r in folds['fold_3']:
        a=r['attacked_wealth']
        lines.append(f"| {r['model_seed']} | {r['clean_wealth']:.6f} | {a['0.05']:.6f} | {a['0.1']:.6f} | {a['0.2']:.6f} |")
    lines+=['','## 其他驗證時段','', '### 2015–2016','']
    grouped+=table(lines,'fold_1',folds['fold_1'])
    lines+=['','### 2017–2018','']
    grouped+=table(lines,'fold_2',folds['fold_2'])
    lines+=['','## σ=0.05：十個 noise seeds 的補充驗證','',
        '噪音 seeds 為 123–132；每個 fold 共 5 個模型 × 10 個噪音實現 = 50 次配對評估。',
        '這不是 50 個獨立訓練 seeds；下表以「評估次數」計數，不能與主要表格的 5 個模型分母混用。','',
        '| 時段 | 攻擊財富中位數（50 次） | 配對財富變化中位數 | 受傷評估次數 | 贏 B&H 評估次數 |',
        '|---|---:|---:|---:|---:|']
    noise_groups=[]
    for fold,label in [('fold_1','2015–2016'),('fold_2','2017–2018'),('fold_3','2019–2021')]:
        rows=[r for r in repeated['cells'] if r['arm']==ARM and r['fold']==fold]
        assert len(rows)==50 and len({(r['seed'],r['noise_seed']) for r in rows})==50
        s=summary([r['clean']['metrics']['final_wealth'] for r in rows],
                  [r['perturbed']['metrics']['final_wealth'] for r in rows],rows[0]['buy_and_hold'])
        noise_groups.append({'fold':fold,**s})
        lines.append(f"| {label} | {s['median_final_wealth']:.4f} 倍 | {s['median_paired_relative_wealth_change_pct']:+.2f}% | {s['harmed_seeds']}/50 | {s['wins_vs_buy_and_hold']}/50 |")
    keys=['steps','input_changed_steps','probability_changed_steps','same_state_action_flips',
          'flips_masked_by_gate','flips_masked_by_five_day_lock','flips_reaching_executed_target']
    audit={k:sum(c[k] for c in trace['cells']) for k in keys}
    lines+=['','## 實驗設定與完整性','',
        '- 資料：真實 CRSP SPY 資料；各 fold scaler 只依訓練期擬合；2022 之後資料未使用。',
        '- RGN：對 11 個標準化價格衍生特徵加獨立零均值 Gaussian noise；σ=0.05、0.1、0.2 為訓練標準差單位。',
        '- 保護欄位：利率、觀察價格、SMA-200 gate flag、真實成交／估值價格、股息和成本規則。',
        '- 帳戶與五日規則狀態依各軌跡自然演進，沒有直接加噪音；交易不同後可間接改變。',
        '- deterministic policy；完整時段評估；模型、optimizer 與 scaler 在評估前後不變。',
        '- clean replay 與原始軌跡 hash 一致；reward 加總符合 log NAV 恆等式；評估自然終止。',
        '- σ=0.1／0.2 只測一個 noise seed，是小規模強度診斷；σ=0.05 才有十個 noise seeds。',
        '- 本次沿用目前 feature-only threat model；完整 Attack Spec V1 原文未取得，因此未宣稱正式規格符合。','',
        '## 為何財富中位數可能完全不變？','',
        '已完成 folds 2–3 共 100 次 σ=0.05 評估的端到端追蹤。在同一帳戶／規則狀態下只置換乾淨特徵，隔離特徵噪音對動作的直接影響：','',
        f"- {audit['steps']:,} 個決策的 policy input 與動作機率都確實改變。",
        f"- {audit['same_state_action_flips']:,} 次動作翻轉中，{audit['flips_masked_by_gate']:,} 次被 gate 抑制，{audit['flips_masked_by_five_day_lock']} 次被五日限制擋下，只有 {audit['flips_reaching_executed_target']} 次到達執行目標。",
        '- 因此，沒有發現噪音檔案未傳入 policy 的問題；結果反映 neural policy 加上 gate／持有規則的整體系統。','',
        '## 結論','',
        '在目前測試的 feature-only PPO V9 設定下，RGN 未呈現跨時段一致的績效破壞效果。',
        '2015–2016 確實有退化；2017–2018 與 2019–2021 的財富中位數則多半不變或改善。',
        '這不等於每個 seed 都沒受影響，也不構成「統計上無顯著效果」或「所有 Gaussian 攻擊均無效」的證明。',
        '尚未測試直接擾動原始價格、重算 gate、較廣泛噪音強度，以及利用 policy 梯度選擇方向的攻擊。',
        '未來比較應保留相同 checkpoint、日期、攻擊欄位和明確可比的擾動預算，並同時報告配對財富損失和實際交易改變。','',
        '## 可稽核來源','']
    for source in sources:
        lines.append(f"- `{source['path']}`；SHA-256 `{source['sha256']}`")
    OUT.mkdir(parents=True,exist_ok=False)
    with (OUT/'REPORT.md').open('x',encoding='utf-8') as handle:
        handle.write('\n'.join(lines)+'\n')
    with (OUT/'REPORT.json').open('x',encoding='utf-8') as handle:
        json.dump({'status':'COMPLETE','algorithm':'PPO V9','arm':ARM,
            'definition_relative_change':'median over matched model pairs of 100*(W_attack/W_clean-1)',
            'harm_tolerance':1e-12,'primary_fold':'fold_3','model_seeds':list(range(5)),
            'single_noise_seed_results':grouped,'repeated_noise_results':noise_groups,
            'per_model':folds,'trace_audit':audit,'sources':sources},handle,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps(grouped[:3],indent=2))
    print(OUT/'REPORT.md')


if __name__=='__main__':
    main()
