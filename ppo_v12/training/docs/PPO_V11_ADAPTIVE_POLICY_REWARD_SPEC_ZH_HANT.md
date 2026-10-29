> V12 training override: see ../../../docs/PPO_V12_FULL_TRAINING_ADDENDUM_ZH_HANT.md. Initial request 500000 / actual 501760; five seeds 0–4. All other V11 economics remain unchanged.

# PPO V11 規格：近期加權、VIX 動態更新與三因子獎勵

**版本：v11spec2_200k；日期：2026-10-07；狀態：已實作並完成單元測試及三-arm smoke；正式矩陣尚未執行。** 本文件與配套 [代理 Prompt](prompts/PPO_V11_IMPLEMENTATION_PROMPT_ZH_HANT.md) 為實作契約；實際驗證範圍見 [驗證報告](../reports/V11_IMPLEMENTATION_CHECKS.md)。參數是預先登記的工程預設，不是論文證實的最優解。

沿用 [V10 規格](PPO_V10_NAV_TREND_REWARD_SPEC_ZH_HANT.md) 的 11 節結構與數值採用門檻。V11 明確取代 V10 的資料起點、episode 長度、離線訓練與 frozen-policy evaluation 契約；其餘環境規則保留。不得修改、覆寫或重新標記 V9/V10 紀錄。若實作發現本文件內部矛盾或未明示的 parent 契約衝突，停止該項工作並回報，不得自行改參數。

## 1. 研究問題

> 在固定 SPY 環境、SMA-200 gate 與五交易日規則下，以近期加權歷史 episodes 進行 chronological PPO fine-tuning，並由 VIX 決定更新間隔，是否改善因果 walk-forward 的扣成本財富？在相同動態更新政策下，NAV、SPY-relative 與正報酬 streak 的固定混合獎勵是否優於 NAV-only？

拆成兩個預先固定的對照：reward 效果與更新排程效果。V11 不以舊 V10 的績效作 matched control。所有正式績效只來自真實 ledger；shaping bonus 不加入資產、現金或 PnL。

## 2. V9/V10 中保留與明確變更的項目

### 2.1 保留的共同契約

- V9 parent commit：`f4988db5c98d9248533c4ca852cc48ca3f6b7aad`。實作前核對實際 submodule；任何必要更新須保留 parent 與新 implementation commit。
- 重用 `SingleAssetTradingEnvV4`、`TradeDecisionEnv(..., "v3_min_hold_five")`、`TrendGateWrapper`、原始 ledger 與 SPY synthetic total-return reference，不重寫其經濟語義。
- SPY、20 維 observation、long-only、無槓桿、目標權重 `{0.0, 0.5, 1.0}`、初始資金 `$1,000,000`、交易成本 `10 bps`。
- close `t` 決策、true raw open `t+1` 執行、close `t+1` 估值並交付 reward 與 observation。
- observed total-return feature price 高於其 SMA-200 時要求全額 SPY；否則通過 PPO action。五交易日規則仍可延後 gate 要求。
- PPO：SB3 `2.9.0`、`MlpPolicy`、CPU、單一 training environment；`n_steps=2048`、`batch_size=64`、`n_epochs=10`、`gamma=0.99`、`gae_lambda=0.95`、`clip_range=0.2`、`normalize_advantage=true`、`ent_coef=0`、`vf_coef=0.5`、`max_grad_norm=0.5`、兩層 `[64,64]`、Tanh。
- seeds `[0,1,2,3,4]`；全部結果與失敗保留，禁止挑最佳 seed。

### 2.2 V11 明確變更

初始學習資料範圍為 `2010-01-01` 至 `2021-12-31`，實際使用範圍內有效 SPY sessions。早於 2010 的資料僅可建立 causal market indicators；不可擬合 scaler、抽樣 episodes 或貢獻 reward transitions。2010 起的第一個有效 session 可作 episode 初始 observation，但首個 reward 的決策及估值兩端都必須在 2010 起。

「初始訓練截止 2021」不表示可在早期模擬讀取完整 2010–2021。每個 cutoff 只看當時 prefix。各 development fold 亦不得使用其 training boundary 之後資料預訓練。

固定 63 個 reward transitions 的訓練 episode，需 64 個連續有效 session observations。第 63 個 transition 作 time-limit truncation；使用 SB3 的 terminal observation/bootstrap 機制，不將人為窗口終點視為市場終結。窗口的最終 valuation session 不得超過 cutoff。訓練 episodes 每次 reset 使用既有初始帳戶狀態及 `streak=0`，不沿用實際交易帳戶。

## 3. 主要變更：訓練政策、時鐘與混合 reward

### 3.1 Chronological 初始化及 fine-tuning

累積 252 個具備有效 causal features 的 2010 起 SPY sessions 後，在第 252 個 session 收盤初始化模型。此前使用 deterministic gate-only baseline；baseline 定義沿用既有實作。scaler 僅擬合這 252 個 sessions，之後永久固定，並保存 hash。不足 252 個 sessions 不得初始化。

初始化請求 `200,000` transitions；完整 2048-step rollouts 預期實際 `200,704`。初始化沿用 learning rate 線性 `3e-4 → 3e-5`，以該次初始化實際步數定義進度。fine-tuning 每次恰為 `32,768 = 16 × 2048` transitions，learning rate 固定 `3e-5`；不得因 SB3 全域進度重啟線性 schedule。

每次更新保留 policy、value network 與 optimizer state；收集該更新期間當前 policy 的 fresh rollout，允許 PPO 對該 rollout 做既定 epochs。不能保存並 replay 舊政策的 rollout/advantages。抽樣的是歷史市場窗口，動作與 reward 都重新模擬。更新期間 cutoff 固定，不因訓練計算讀入新日期。

每個 fold/seed/arm 使用獨立模型、optimizer、scaler 和 RNG，不跨 folds 搬運 checkpoint。training RNG 與 deterministic evaluation 分離；episode 抽樣與 rollout 的種子衍生規則在實作 config 登記並保存 RNG state，讓 checkpoint resume 可重現。

### 3.2 近期資料權重

在 cutoff `c` 列舉所有合法 63-transition 窗口，結束日為 `e_i`。`a_i` 是 `e_i` 到 `c` 的 SPY session index 差，最新合法窗口的 age 為 0；`N` 為合法窗口數。

```text
q_i = 2 ** (-a_i / 126)
p_i = 0.80 * q_i / sum(q_j) + 0.20 / N
```

每次 episode reset 依 `p_i` 有放回抽樣。所有配對 arms 使用同一候選窗口及機率；不同 policy 仍產生各自 rollout。`N=0` 是技術失敗。此設計改變日期出現機率，不加權 PPO loss，也不對過去實際交易 actions 做行為複製。

半衰期 `126` sessions 約六個月；兩個結束日相差 126 sessions 的 exponential component 權重比為 2:1，加入 uniform 後總機率比不再恰為 2:1。63-transition 窗口會平滑實際 observation 日期權重，因此不能宣稱每個 observation 本身有精確 126-session 半衰期。報告實際 sampled endpoint ages 與 transition dates 分布。

例如模擬 `2018-10-10` 收盤更新時，合法窗口的最後估值日最多為該日；當年 10 月 11 日及以後的資料不可影響窗口、scaler、reward 或模型。

### 3.3 VIX 動態時鐘與缺值

以 SPY session calendar 對齊 VIX，保留原始來源與 observation date。已存在候選資料 `data/raw/VIXCLS_1990_2026.csv`（欄位 `observation_date,VIXCLS`）僅是本地候選，實作必須確認來源、有效值與可得時點並記錄 hash，不可假設整份檔案都可供當下使用。VIX 使用 index level，例如 20，不是 0.20。

對每個 SPY session，缺值可沿時間正向填補，最多連續 3 個 SPY sessions；不允許 backfill。非有限或非正值按無效資料處理。第 4 個缺值 session 或無歷史值可填時停止該 run。週末與 SPY 休市日不增加缺值計數。以完成填補的連續五個前置 SPY sessions 計算平均；填補值保留旗標，不以跳過缺值的方式偷偷延長視窗。

對收盤 decision date `t`：

```text
vix_mean[t] = mean(aligned_vix[t-5], ..., aligned_vix[t-1])
raw_period[t] = 126 * (20 / vix_mean[t]) ** 2
P[t] = min(252, max(21, floor(raw_period[t] + 0.5)))
elapsed[t] = session_index[t] - session_index[last_completed_update]
update_due[t] = elapsed[t] >= P[t]
```

初始化視為一次 completed update。每日重新計算 `P[t]`；VIX 升高可提前觸發，下降可延後，但最多 252 sessions 必須更新。每個 session 最多一次更新，不補做多次落後更新；無獨立 emergency trigger。

| 五日平均 VIX | 更新間隔 sessions | 相對於正常期 |
|---:|---:|---|
| 10 | 252 | 上限 |
| 15 | 224 | 較慢 |
| 20 | 126 | 約六個月 |
| 25 | 81 | 較快 |
| 30 | 56 | 約正常期的 44% |
| 40 | 32 | 約正常期的 25% |
| 50 | 21 | 下限 |

固定排程 control 使用 `P=126`，其他資料、模型、抽樣與每次預算相同；仍保存 VIX 診斷並遵守相同 data preflight。

### 3.4 收盤與啟用順序

1. close `t` 完成既有執行路徑的估值與 reward，以啟用中的舊 checkpoint 做 close `t` 決策，該 action 在 open `t+1` 執行。
2. 檢查截至 `t-1` 的 VIX 與 elapsed；若到期，使用完整觀測至 close `t` 的歷史，在獨立 training environment 更新。
3. 新 checkpoint 於 close `t+1` 決策才啟用，其首個可受影響成交為 open `t+2`。不得回頭修改 close `t` 的 action。
4. model swap 不 reset evaluation environment、NAV、持倉、交易成本、五日規則或技術指標；失敗更新停止 run，保存失敗前軌跡，不靜默回退並繼續計分。

這是模擬計算可在下一次收盤前完成的假設，不是實盤運算時延承諾。記錄 cutoff、completed-update date、activation date 與首個影響成交日。最終計分 session 不啟動無法影響任何後續決策的更新。訓練階段轉入 validation 時保留模型、optimizer 與 last-update clock；只按既有 evaluation 契約建立新的計分帳戶。

### 3.5 三因子獎勵

每個 transition 都由同一 V4 ledger 的淨 NAV 及既有 synthetic close-to-close SPY reference 計算：

```text
r_nav[t] = log(NAV_close[t+1] / NAV_close[t])
r_spy[t] = log((SPY_raw_close[t+1] + cash_exdate[t+1]) / SPY_raw_close[t])
r_relative[t] = r_nav[t] - r_spy[t]
streak[t] = streak[t-1] + 1 if r_nav[t] > 0 else 0
b[t] = 0.10 * max(min(streak[t], 5) - 1, 0) * max(r_nav[t], 0)
r_trend[t] = r_nav[t] + b[t]
r_train[t] = 0.30*r_nav[t] + 0.20*r_relative[t] + 0.50*r_trend[t]
           = r_nav[t] - 0.20*r_spy[t] + 0.50*b[t]
```

每個 training episode reset 重設 streak；`r_nav=0` 也中斷 streak。負報酬不產生 bonus。不使用 market momentum 替代此 streak，也不改 SMA-200 gate。synthetic SPY reference 不等於 executable buy-and-hold ledger。

| 正 streak 長度 | 原始 V10 bonus / 正 r_nav | 混合 reward 的加權 bonus / 正 r_nav |
|---:|---:|---:|
| 1 | 0% | 0% |
| 2 | 10% | 5% |
| 3 | 20% | 10% |
| 4 | 30% | 15% |
| ≥5 | 40% | 20% |

50% 是 NAV-plus-streak component 的係數，不代表 50% 的 realized reward 來自 bonus。使用相同 log-return 單位，不另作逐成分標準化或 adaptive weights。reward 可為負，亦不因 VIX 改變。

在相同固定日期及 action-independent 市場路徑下，減去 SPY 項是與 action 無關的項；不能宣稱其直接改變精確目標下的最優政策。它仍可能影響有限預算 PPO 的 value fitting 與 learning dynamics。streak 是 action-dependent shaping，可能改變目標；不是具有一般 policy-invariance 保證的 potential-based shaping。

## 4. 實驗 arms 與因果對照

| Arm | 更新排程 | Training reward | 計分 reward |
|---|---|---|---|
| `ppo_v11_vix_nav` | VIX 動態 | `r_nav` | `r_nav` |
| `ppo_v11_vix_mix` | VIX 動態 | `r_train` | `r_nav` |
| `ppo_v11_fixed_nav` | 固定 126 sessions | `r_nav` | `r_nav` |

reward 對照為 `vix_mix` 相對 `vix_nav`，共用每個 fold 的 cutoff/update schedule，僅訓練 reward 不同。排程對照為 `vix_nav` 相對 `fixed_nav`；初始化、每次訓練步數與抽樣相同，但更新次數及總計算量可能不同。不能將該差異稱為純粹 compute-matched 效果。`vix_nav` 結果可供兩對照共用，不重訓來挑較好結果。

這些對照不單獨識別近期加權相對 uniform sampling 的效果。若要研究該效果，須另版預先登記對照，不能把 V11 改善全部歸因於 recency。

## 5. Wrapper、控制器行為與必要欄位

實作職責分離：causal prefix/window sampler、VIX update scheduler、chronological training controller、training-only mixed reward wrapper，以及獨立 scoring environment。既有 20 維 observation 不加入 VIX、reward 或 streak。不得嵌套舊 relative wrapper 後再把其輸出當作 `r_nav`，導致重複扣 SPY。

每個訓練 step 至少保存：`nav_log_reward`、`spy_log_return`、`relative_log_reward`、`positive_return_streak`、`positive_streak_bonus`、`weighted_streak_bonus`、`trend_component_reward`、`training_reward`、`reward_mode`，以及既有 `agent_action`、`gate_forced_full`、`gated_action`、`executed_action`。

NAV arms 可計算相同診斷，但 `training_reward=r_nav`、delivered `weighted_streak_bonus=0`；診斷值不代表 bonus 已用於訓練。scoring environment 的 delivered reward 永遠為 `r_nav`；adaptive learning 在隔離的 training clone 使用對應 arm reward，不將 scoring ledger 複製到 training episode。

每次更新至少保存：trigger date、cutoff、activation date、last-update date、elapsed、raw/filled VIX 五日值與平均、requested/rounded/clipped period、update reason、候選窗口及機率 hash、sampled ages、模型與 optimizer 前後 hash、scaler hash、requested/realized transitions、learning rate、RNG state reference、耗時與 failure status。

## 6. Fold、seed 與正式矩陣

| Fold | 初始 development training prefix | Adaptive validation |
|---|---|---|
| `fold_1` | 2010-01-01 至 2014-12-31 | 2015-01-01 至 2016-12-31 |
| `fold_2` | 2010-01-01 至 2016-12-31 | 2017-01-01 至 2018-12-31 |
| `fold_3` | 2010-01-01 至 2018-12-31 | 2019-01-01 至 2021-12-31 |

每個 fold 都從相同初始化流程獨立循時推進到 train boundary，再進入 validation。scaler 擬合初始 252-session prefix，不是整個 fold train span。validation 決策為 deterministic，但允許依固定演算法使用已觀測 validation prefix fine-tune。超參數、採用規則與 reward weights 不隨 validation 結果修改。

正式矩陣 `3 folds × 5 seeds × 3 arms = 45` 個獨立 chronological trajectories，包含 reward 主比較 30 cells 與額外 fixed-nav 15 cells；不是 45 個單次 200k 訓練。各 trajectory 總 transitions 為 `200,704 + 32,768 × completed_fine_tunes`，報告 training/validation 階段分項。

技術 smoke：fold_1、seed 0、三 arms，初始化請求 20,000（預期 20,480）及一次完整 32,768-step 更新，以人工短時鐘觸發並明確標記 `smoke`，不能加入正式結果。保留相同模型及 wrapper；smoke 專用時鐘不可進入 formal config。

正式全期模型以 2010–2021 prefix 同樣循時建立，不以 development 中最佳 checkpoint 代替；保留全部五 seeds。2022–2025 為 `adaptive_known_period_confirmation_not_untouched_test`，可循時使用已觀測期間資料更新，scaler 不變。參數與最終選擇须先凍結；本文件交付不啟動該期間。

## 7. 驗證、主要比較與採用規則

### 7.1 成對對照與決策

```text
delta_reward = log(W_vix_mix / W_vix_nav)
delta_schedule = log(W_vix_nav / W_fixed_nav)
```

每個對照報告全部 15 paired deltas、各 fold median、win count、pooled distribution、三個 fold medians 算術平均。每個對照獨立套用原 V10 四門檻：

1. 至少兩個 fold 的 paired median delta 嚴格大於 0。
2. 15 pairs 中至少 10 個 delta 嚴格大於 0。
3. treatment 的三-fold median maximum-drawdown magnitude 比 control 惡化不超過 `2.0` percentage points。
4. treatment 的三-fold median transaction cost 不高於 control 的 `110%`。

聚合口徑固定為：先於各 fold 對五 seeds 取 MDD magnitude 或 transaction cost 的 median，再對三個 fold medians 取 median，於 treatment/control 間比較。cost 使用同初始資金下 ledger 的總 dollar cost。control cost 為 0 時，treatment 也須為 0，不以除零或 epsilon 掩蓋。此為將 V10 的三-fold median 用語明確化，不調整數值門檻。

任一技術失敗使該對照判定 `INCOMPLETE`，不得移除 seed 或視作平手後聲稱通過。兩項比較都通過才可採用完整 `vix_mix` 方案；任一未過時保留比較結論，不自動選用未檢驗的 `fixed_mix` 或宣告替代 production 模型。

### 7.2 必報指標

保留 V10：final wealth、CAGR、lagged-cash excess return 與 Sharpe（numerator/denominator）、MDD、annualized volatility、average exposure、turnover、trade count、transaction cost；action distribution、gate-forced/override share、five-day-rule blocked share；cash、executable SPY buy-and-hold、constant-50、gate-only baselines。所有 baselines 使用相同計分 sessions 與成本契約。

另外報告 streak 分布、bonus-positive 比例、原始/加權 bonus 總和與相對 `sum(abs(r_nav))` 比例、三 reward component 的 signed sum 與 absolute sum、更新次數/interval 分布、VIX 缺值填補、fresh rollout 數、總 transitions、CPU time、sampled age 分布。分母為 0 時標為 undefined 並保存原值。

### 7.3 敏感度與 known period

敏感度僅於 development data，且與 formal 主結果分開：one-factor-at-a-time 比較 half-life `63/126/252`、recent mixture `0.60/0.80/0.90`、VIX exponent `1/2/3`、reward weights `(0.40,0.20,0.40)/(0.30,0.20,0.50)/(0.20,0.20,0.60)`；其他參數固定。此為附加探索，不是自動搜尋或本次執行要求。不得將觀察後改參數的結果沿用 v11spec2_200k；修訂需新版本及紀錄。不得使用 2022–2025 選參數、挑 seed、改門檻或反覆挑 checkpoint。

## 8. 正式訓練前必須通過的測試

1. **日期/窗口：** 2010 前無 learning transitions/scaler rows；63 transitions 對應 64 sessions；cutoff 前完整窗口才合法；252-session initialization boundary 正確。
2. **未來 sentinel：** 修改 cutoff 後價格、VIX、公司行動或現金資料，不得改較早 observations、sampling probabilities、scheduler、rewards 或 checkpoints；新增未来資料不得改初始 scaler。
3. **抽樣：** 所有 `p_i>0`、總和與 1 的誤差 `≤1e-12`；exponential component 在 age 相差 126 時比為 2；相同 RNG state 抽樣可重現，記錄實際 endpoint/transition ages。
4. **VIX：** 上表數值、half-up rounding、21/252 clamp、五個前置 sessions、缺值最多三次、第四次停止、無 backfill、daily shortening/lengthening、每 session 最多一次、固定 control 126。
5. **時序：** close t action 不受該日更新影響；close t+1 首次採新模型，open t+2 首次受其成交影響；model swap 不改 ledger、hold state；train/validation boundary 不重設 update clock。
6. **reward：** 正/正/正/負/正的 streak 為 `1,2,3,0,1`；zero 中斷、cap/reset 正確；weighted bonus 上限為 `0.20*max(r_nav,0)`；四項 reward 與展開公式逐步一致。
7. **ledger 與 identity：** 相同 action path 下 NAV、成交、cost、observations、termination/truncation 一致；完整 episode `sum(r_nav)=log(NAV_end/NAV_start)`；mix episode `sum(r_train)=log(NAV_end/NAV_start)-0.20*sum(r_spy)+0.50*sum(b)`，絕對誤差 `≤1e-10`。
8. **隔離與 PPO：** scoring reward 永遠等於 NAV；learning clone 不改 scoring environment；optimizer 在更新/resume 後延續、scaler hash 固定、舊 rollout 不 replay、fresh rollout 與 LR 正確；finite observation/loss、state-responsiveness、checkpoint reload、時間截斷 bootstrap 與自然 scoring 終止。
9. **聚合與失敗：** 15 pairs 完整性、零成本比較、四門檻等號邊界、不同 arm 的共享 control、technical failure 的 `INCOMPLETE`，以及 append-only resume 不覆寫。

此處規定未来實作的必要測試，不表示本次文件交付已執行 PPO smoke 或驗證模型。

## 9. Provenance 與輸出要求

每個 trajectory 與 update 使用 append-only records；保存 spec/Prompt SHA-256、V11 code/config commit、V9 parent commit、實際 victimagent commit、package versions、arm/fold/seed、market/events/VIX/session-calendar/window-manifest/scaler hashes、資料來源及 availability lag、requested/realized transitions、全部公式與凍結值、checkpoint/optimizer/RNG hashes、training/scoring trajectories、stdout/stderr、全部 tests、metrics 與 failure records。

V11 新 namespace：`experiments/v11_ppo/records/`、`experiments/v11_ppo/attempts/`、`artifacts/ppo_v11/`、`reports/v11_ppo/`、`tensorboard/ppo_v11/`。新增 VIX/日期契約使舊 processed-data hashes 不足以驗證 V11；保留 parent hashes，另建 V11 manifest，禁止為符合舊 hash 修改資料。

中斷恢復須一起還原 model、optimizer、scaler、RNG、scheduler、scoring ledger、hold state 與 pending checkpoint/action；保存 resume provenance。缺失任一必要狀態時，不把半恢復 run 混入 formal matrix。

## 10. 結果詮釋限制與參數依據

### 10.1 參數—理由—來源—限制

| 參數/原則 | 本規格採用理由 | 來源 | 限制與證據等級 |
|---|---|---|---|
| VIX 作 adaptation proxy | VIX 反映 SPX options 隱含的未來 30 日預期波動，與 SPY 市場風險相關 | [Cboe VIX FAQ](https://www.cboe.com/tradable_products/vix/faqs) | 支持 proxy 定義，不證明高 VIX 必有 policy drift 或 fine-tuning 必然有益 |
| 歷史 discounting + 保留長期資料 | 非定態環境中近期資料更能反映當前分布；uniform 部分避免完全失去早期經驗 | [Globally Nonstationary Bandits, JMLR 2024](https://jmlr.org/papers/volume25/21-0916/21-0916.pdf) | discounting/windowing 是相關原理；bandit 理論保證不可直接移植至 PPO |
| half-life 126、80/20 mixture、63-transition episode | 約六個月 recency 配合正常更新週期；20% 歷史底座；63 約一季且初始一年可形成足夠窗口 | 上述非定態原理＋工程設計 | 所有數值皆預設，未以本專案實驗證實；短 episode 改變訓練目標的有效 horizon |
| `126*(20/VIX)^2` | 若每日變異負荷與 volatility squared 成比例，固定累積變異負荷會導出 inverse-square 時鐘 | 工程推導，VIX 單位依 Cboe 定義 | VIX 是年化 implied volatility proxy；此非精確 realized variance 模型或實證最優排程 |
| VIX reference 20；21–252 bounds；五日平均；三日填補 | 20 作固定標尺；一月至一年限制計算/陳舊風險；五日平滑避免單日跳動；lag 保守處理 publication 時點 | 工程預設 | 不是估計的全樣本均值，不是論文推導門檻；平滑與 lag 會延遲反應 |
| fresh rollout、warm-start PPO、clip 0.2 | PPO 從當前政策收集資料，再對當批資料做多次 constrained surrogate 更新；warm-start 減少重建成本 | [Schulman et al., PPO, 2017](https://arxiv.org/abs/1707.06347) | 不保證金融非定態環境收斂或 KL 永遠受限；32,768 與 LR 為本案工程預算 |
| NAV/relative/trend = 30/20/50 | 保留淨財富主體、有限 benchmark-relative learning influence；trend component 最大，同時沿用 capped bonus | V9/V10 契約＋使用者偏好 | 非學術最優權重；展開後 bonus 最大只占正 NAV 的 20%，非 50% |
| beta 0.10、cap 5 | 保留 V10 一交易週的有限 streak bonus，避免固定微小正收益套利與無界累積 | 既有 V10 工程契約 | 五日 streak 不等同長期 market momentum，也不代表必然經濟優勢 |
| 不宣稱 shaping policy invariance | 一般保持最優政策的 shaping 有特定 potential-based 結構；本 streak bonus 未滿足其保證 | [Ng, Harada & Russell, 1999](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf) | 僅解釋理論限制；不證明本 bonus 必然傷害或改善策略 |

上述來源採意譯；數值來源清楚標記為使用者決策、parent 契約或工程預設。文獻查核日期為 2026-10-07。

### 10.2 解讀邊界

- development folds 已反覆使用；adaptive validation 中模型也學習其已觀測資料。這是 prequential/walk-forward development evidence，不是 frozen-policy holdout。
- 2022–2025 已開啟，且允許 causal online learning；只能稱 adaptive known-period confirmation，不是 untouched test。
- gate 可能主導曝險；必須分開报告 gate 強制與 agent 自主 action，不能將全部收益歸給 PPO。
- streak 未納入既有 observation，agent 不直接看到 bonus 狀態；critic 的狀態別名與小 bonus 的學習效果須如實討論，不能暗中增加 observation 維度。
- inverse-square 時鐘、recency mixture 及固定權重是可反駁的設計假設。更快更新可能造成估計噪音、過擬合、遺忘與計算成本上升。
- 四門檻是工程採用規則，不是統計顯著性，也不是穩定勝過 executable SPY buy-and-hold 的證明。

## 11. 實作者簡要檢查表

- [ ] 讀取 parent 契約，僅套用本文件明示的 V11 改動；新 namespace，不覆寫舊紀錄。
- [ ] 凍結 2010 起點、252-session initialization、63-transition episodes、126 half-life、80/20 mixture。
- [ ] 凍結 VIX 時鐘、publication lag、fill 上限及 activation 順序。
- [ ] 凍結 30/20/50 weights、beta 0.10、cap 5；不另改 normalization 或 reward。
- [ ] 隔離 training clone 與 scoring ledger，保留 optimizer、scaler、scheduler 與帳戶狀態。
- [ ] 依第 8 節通過必要 tests，再做標記清楚的三-arm smoke。
- [ ] formal 前保存 config、spec、code 與資料 hashes，確認 45 trajectories 的資源估計。
- [ ] 報告兩個成對對照、全部 seeds、四個原門檻與額外 compute，不自動組合未測 arm。
- [ ] 在已凍結最終 protocol 後，另行安排 adaptive known-period confirmation；本次文件工作不執行。
- [ ] 最终交付實作狀態、驗證證據、已知限制與尚未執行項目，不把文件檢查寫成模型測試通過。


**2026-10-07 使用者修訂：** 本版本初始化 requested transitions 改為 200,000，SB3 實際 200,704；每次 fine-tuning 仍為 32,768。原 V10 目錄中的 v11spec1 文件保留為歷史版本。
