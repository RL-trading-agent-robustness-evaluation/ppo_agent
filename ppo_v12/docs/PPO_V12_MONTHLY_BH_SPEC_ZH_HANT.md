# PPO V12：每月定額 B&H 與既有模型比較規格

版本 `v12spec1_benchmark_only`；2026-10-08。數值為預先登記的工程預設；本次不啟動或更新 PPO。配套：[代理 Prompt](prompts/PPO_V12_IMPLEMENTATION_PROMPT_ZH_HANT.md)。沿用 V11 的 11 節結構及正式採用門檻；本次新增基準比較，並非新的學習政策。

## 1. 研究問題

在相同 2022–2025 估值期間、起始資金與成本下，既有五個 V11 adaptive 模型相對「從起始現金逐月等額買入」SPY 的 NAV 表現如何？同時保留原 lump-sum B&H，避免改換基準被誤認為 RL 改善。目的為使用者決定是否保留新基準或進一步訓練提供資料，不自動決定採用或訓練。

## 2. V11 中保留與明確變更的項目

保留 `ppo_v11`、模型、ledger、既有結果。V12 直接讀取其資料及程式，不複製大型模型或資料；provenance 保存實際依賴 hashes。既有完整契約另存 [V11 參考快照](reference/PPO_V11_ADAPTIVE_POLICY_REWARD_SPEC_ZH_HANT.md)，內容不修改。V12 本規格優先於快照中本次工作範圍與訓練啟動要求。

- RL 初始資金 $1,000,000，全部立即可用；SPY、long-only、無槓桿、動作 `{0,0.5,1}`、10 bps、SMA-200 gate、五交易日限制保持原樣。
- close t 決策、true raw open t+1 成交、close t+1 估值；不新增終點賣出。
- 僅新增獨立月買基準 ledger 與報告。基準使用 fractional shares；既有 RL 與 lump-sum B&H 使用整股，報告必須揭露差異。
- 定額基準不套用 gate 或最短持有規則；股票買入後持有，沒有賣出、外部入金或績效回填。
- 不改 observation、reward reference、PPO loss、scaler 或既有模型。不載入模型進行推論；只驗證 checkpoints hashes。

## 3. 主要變更：定額投入、訓練政策與 reward

### 3.1 定額現金預算

設定 C0=1,000,000，M=48，c=10/10000。每期支出預算 B=C0/M，包含成本；末期使用 C0 減前 47 期已分配預算，吸收浮點尾差。對當期真實開盤價 O：

\[
Q_k=\frac{B_k}{(1+c)O_k},\quad N_k=Q_k O_k=\frac{B_k}{1+c},\quad F_k=cN_k
\]

現金減少 Bk，股份增加 Qk；交易後開盤 NAV 等於交易前 NAV 減 Fk。48 期本金預算總和 C0，總成本 C0×c/(1+c)，沒有把成本另行加到本金之上。若現金不足，不借款、不縮減固定金額，停止並保存失敗。

### 3.2 排程、現金與事件

2022-01 至 2025-12 每月第一個可執行 session 開盤購買。初始帳戶在 2022-01-03 收盤，該日開盤已不可交易，因此第一期 2022-01-04；其他月份為第一個資料中可交易 session。每筆使用前一 session 收盤作決策日期。日曆與金額事先固定，不依未來價格或 VIX 改動。缺少整個月份或不具可交易 open 時停止。

既有 V4 在 close t 可用現金以 t 的 lagged DGS3MO 對至下一 session 的日曆天數做 ACT/365 複利。下一 open 先處理 splits、ex-date 股息應收、payment-date 結清，之後交易；股息權利使用 ex-date 開盤買入前的股份。應收股息計入 NAV，但不生息、不作可用現金。利息、股息結清保留現金，不擴大固定購買預算。終點 NAV=可用現金+股份×raw close+應收股息。

### 3.3 凍結 V11 學習契約（供未來階段）

初始資料為 2010-01-01 至 2021-12-31；pre-2010 僅暖機。每個 cutoff 只用已觀測 prefix；252 個有效 sessions 後初始化，scaler 只擬合當時 252 sessions 並永久凍結。warm-start PPO 每次保留模型與 optimizer、重新收集當前政策 rollout，不 replay 舊 transitions、不重設實際投資組合。63 transitions episodes 使用 64 個 sessions，完全在 cutoff 以前；抽樣：

\[
p_i=0.8\frac{2^{-a_i/126}}{\sum_j2^{-a_j/126}}+0.2/N.
\]

VIX 使用截至前一交易日的五個值平均；最多 forward-fill 三 sessions，超過停止：

\[
P_t=\operatorname{clip}(\lfloor126(20/\bar V_t)^2+0.5\rfloor,21,252).
\]

VIX 15/20/30/40/50 對應 224/126/56/32/21 sessions。沿用 V11 freeze-action、訓練與延後啟用時序，不能讓更新影響已決定的當期成交。2021 是初始學習截止，不是永久學習截止；testing 與部署仍只用已觀測資料因果更新。

NAV reward 為真實 ledger close-to-close log return，SPY synthetic reference 保持不變。正 NAV reward 才增加 streak，否則歸零；episode reset streak=0。beta=.10、cap=5：

\[
r_{relative}=r_{NAV}-r_{SPY},\quad b=.10\max(\min(streak,5)-1,0)\max(r_{NAV},0)
\]
\[
r_{trend}=r_{NAV}+b,\quad r_{train}=.30r_{NAV}+.20r_{relative}+.50r_{trend}=r_{NAV}-.20r_{SPY}+.50b.
\]

50% 是 trend component 係數，weighted bonus 最大為正 NAV reward 的 20%。SPY 項不依賴動作，不能宣稱其直接改變精確目標的最優政策。定額 B&H 不成為此 SPY reference。

未來初始化請求 200,000，2048-step rollout 預期實際 200,704；每次 fine-tune 32,768。初始化 lr 3e-4 線性降至 3e-5，之後固定 3e-5。PPO 其餘設定使用 V11 config 快照：MlpPolicy、CPU、n_steps=2048、batch=64、epochs=10、gamma=.99、GAE=.95、clip=.2、normalize_advantage=true、ent=0、vf=.5、grad_norm=.5、[64,64] Tanh。這些數值僅保留，並不授權本次訓練。

## 4. 實驗 arms 與因果對照

本次兩基準：`monthly_cash_budget_bh`（新增）及 `buy_and_hold`（原始整股 executable B&H）。對照既有 `ppo_v11_vix_mix` seeds 0–4，來源 tag `adaptive_20261007_1`。不重跑推論，不重訓，不用最佳 seed。資料、事件、1,002 transitions 與日曆必須一致；原 B&H 指標須重現各 V11 record。

基準 deterministic，無須為同一基準執行五次。比較的是不同投資曝險配置，不能推導 reward 或更新排程的因果效果。

## 5. Wrapper、控制器行為與必要欄位

V12 `MonthlyEnvironment` 沿用 V4 transition 時序，將帳戶替換成獨立 `FractionalBudgetLedger`。沿用現金、股息應收及結清，覆寫 fractional shares 驗證、split 和固定金額買入，不改 parent source。原模型 account features 只接受整股，因此基準不組合或使用 policy observation，僅回傳未使用的零向量以滿足環境介面；不將 fractional 持倉取整偽裝為模型輸入。外部提供已登記 budget，只有排程日呼叫購買 hook，其他日 hold shares。

每 session 保存 decision/execution/valuation 日期、start/end NAV、價格、股份、可用現金、應收股息、利息、事件、notional、成本、預算、net return、reward。每筆購買保存月份、預算、交易前後現金及股份、實際股數與價格。不得把當期 raw open 作為前日決策特徵。

## 6. Fold、seed 與正式矩陣

本次只比較既有 seeds `[0,1,2,3,4]`，期間 2022–2025，標為 **adaptive known-period diagnostic**，不叫 untouched test。此期間已開啟，不可用於參數搜尋。

後續 200k、四 seeds `[0,1,2,3]` 僅為待指示的 pilot；四種子不能替代正式採用矩陣。若日後啟動正式 reward 比較，保留三 folds × 五 seeds × NAV/mixed arms，同一動態排程；另以 NAV 比 fixed-126 與 VIX 排程。fold_1 train 至 2014、validation 2015–16；fold_2 train 至 2016、validation 2017–18；fold_3 train 至 2018、validation 2019–21；training 起點全部 2010-01-01。validation 允許因果線上更新；額外更新計算量需揭露。

## 7. 驗證、主要比較與採用規則

只以真實 ledger NAV 計算 final wealth、CAGR、最大回撤、成本、平均曝險。列出兩基準、全部五 seeds、逐指標中位數、模型減基準差值及 final wealth 勝出數；中位數不是可交易的合成帳戶。MDD 為負值，差值正數代表模型回撤較小。初始現金一直在帳戶內，無外部 flow，不另用入金調整來增加收益。

正式門檻完整保留：至少兩個 fold 的 paired median 為正，15 pairs 至少 10 勝，drawdown 惡化不超過 2 percentage points，成本增加不超過 10%；每個 reward/schedule 對照獨立判定，不混入 V10 結果。門檻是工程判斷，不是顯著性證明。本次缺正式 folds/controls，正式採用狀態 **NOT_EVALUATED**；即使五模型都勝新基準，也不算通過。

## 8. 正式訓練前必須通過的測試

本次執行前測試：48 月排程、第一筆可成交日、非交易日順延、整月缺值停止、本金尾差及成本；固定現金隨價格改變股數；合成常數/變價價格路徑精確 NAV；cash/股份/NAV 守恆與不足現金停止；lagged 利息、股息 entitlement/付款、fractional split；未來價格 sentinel 不影響先前交易。

整體驗收：資料 hashes、五紀錄 technical_pass、ledger/checkpoint hashes、有限且正 NAV、逐步 NAV telescoping 與 log-reward identity、原 B&H 重現、相同估值日曆、完整失敗紀錄。V11 模型僅讀取 hash，不 reload 或推論，因此本次不新增 PPO checkpoint roundtrip；既有驗證 provenance 保留。日後訓練須另外重跑原 V11 日期邊界、scaler、VIX 缺值/排程、抽樣、streak/reset、reward identities、同 action ledger 與 checkpoint reload 驗收。

## 9. Provenance 與輸出要求

`configs/benchmark_v12.json` 凍結本次資金、日期、成本、48 月與 seeds；保存 V11 config 與規格參考快照，V11 devlog 原文前綴延續於 V12 devlog。基準入口讀取相鄰 V11 依賴，不修改其檔案、不產生 parent bytecode。

比較報告 JSON 保存 market/events/VIX、V11 config、source modules、五個原始 records、validation ledgers、pre-2022/final checkpoints、V12 source/spec/Prompt 的 SHA-256。執行前後核對依賴 hashes。輸出月買 ledger、原 B&H ledger、48 筆購買、完整 metrics、Markdown/JSON 比較、單元測試 log 及每次 attempt；已有完成產物不得覆寫。失敗保留 logs/traceback 與已產生資料，不刪種子。未來訓練才需新更新 cutoff、VIX/原因、抽樣分布、模型前後 hashes、actual transitions 與失敗紀錄。

## 10. 結果詮釋限制與參數依據

| 參數／設計 | 理由 | 來源 | 限制 |
|---|---|---|---|
| 等額按月購買 | 分散買入時點，低價買較多股份 | [Investor.gov](https://www.investor.gov/introduction-investing/investing-basics/glossary/dollar-cost-averaging) | 不證明較高收益，不消除終點風險 |
| 起始現金逐步投入 | 與一次投入比較時揭露現金的機會成本 | [Finlay & Zorn, Vanguard 2023](https://corporate.vanguard.com/content/dam/corp/research/pdf/cost_averaging_invest_now_or_temporarily_hold_your_cash.pdf) | 該研究不證明本案 48 月最佳，不能移植勝率 |
| 48 月、$1m、月初、10 bps | 使用者選擇與既有交易契約，固定可審核日曆 | 本次預先登記的工程預設 | 不是研究最優值，不依 2022–2025 選參數 |
| fractional、費用包含於 budget | 金額精確相同且不借款 | 帳戶守恆及工程契約 | 與舊整股 RL/B&H 有 rounding 差異 |
| 126 半衰期、80/20 抽樣 | 近期偏重同時保留歷史 | [非定態學習研究，JMLR 2024](https://jmlr.org/papers/volume25/21-0916/21-0916.pdf) | 方向性原理，不可把 bandit 保證移植 PPO；數值為工程預設 |
| VIX、126×(20/VIX)^2、21–252 | 波動代理與較高波動較快更新 | [Cboe VIX FAQ](https://www.cboe.com/tradable_products/vix/faqs) | Cboe 支持 VIX 定義，不支持此公式的最優性 |
| fresh rollouts、warm-start PPO | 從目前政策收集資料並作 clipped surrogate 更新 | [Schulman et al. 2017](https://arxiv.org/abs/1707.06347) | 非定態金融更新時鐘不是 PPO 論文證實的最佳值 |
| .30/.20/.50、beta=.10、cap=5 | 保留既有 trend 偏重設計 | 工程預設；[Ng et al. 1999](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf) 提供 shaping 判準 | streak bonus 不是已證明的 potential shaping，無一般最優政策不變保證 |

不同初期曝險可能同時改變回撤與報酬；單一已知期間與五種子不足以判定普遍優劣。降低比較基準收益不代表 agent 改善。敏感度分析僅可使用 development data，本次不執行。

## 11. 實作者簡要檢查表

- [ ] 獨立 V12，保留 parent、資料及所有五 seeds。
- [ ] 48 期含成本預算合計 $1m，沒有外部入金或額外再投入利息／股息。
- [ ] 同時重現原 B&H、計算月買 B&H，估值期間完全一致。
- [ ] 通過合成測試、原績效重算與 hashes，保留失敗紀錄。
- [ ] 輸出所有結果、差值、勝數、曝險與限制，正式採用 NOT_EVALUATED。
- [ ] 更新 devlog，交付本規格及 Prompt。
- [ ] 停止，不訓練、不更新、不調參；未來四 seeds 訓練等待使用者後續指示。
