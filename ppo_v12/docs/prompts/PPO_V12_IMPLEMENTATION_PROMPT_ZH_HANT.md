# V12 代理實作 Prompt：基準比較優先，禁止自動訓練

你正在 `C:\03_College\Grad_proj` 工作。任務是維護獨立 `ppo_v12`，以每月定額 SPY B&H 比較既有 V11 五模型。先讀 [V12 規格](../PPO_V12_MONTHLY_BH_SPEC_ZH_HANT.md)、[V11 快照](../reference/PPO_V11_ADAPTIVE_POLICY_REWARD_SPEC_ZH_HANT.md) 與 V12 devlog。V12 規格優先於快照的本次執行範圍。

## 本次凍結契約

- 現金 $1,000,000 起始全部在帳戶內；RL 已有模型初始資金全部可用，沒有外部入金。
- 2022-01 至 2025-12 每月第一個可執行 open，48 次等額含成本購買；每期 budget=1m/48，末期補浮點尾差；cost=10bps，shares=budget/((1+.001)×raw open)。第一筆 2022-01-04，由 2022-01-03 close 排程。
- 月買採 fractional shares、買後持有，原 lump-sum 與 RL 保留整股；兩種 B&H 都須呈現，揭露差異。
- 使用既有 V4 cash ACT/365、lagged rate、split、股息應收及付款時序。利息/股息留現金，不擴大購買本金。只以 raw close 終點估值，不強制賣出。
- 同一資料、事件、估值日曆、成本、初始 NAV；來源 V11 tag `adaptive_20261007_1`，既有 seeds 0–4 全部保留。

## 執行步驟

1. 檢查 parent 資料/config hashes、五 records technical_pass、trajectory/checkpoint hashes；重算既有 ledger 的 NAV metrics。不能執行 PPO、載入模型推論或重新蒐集 action path。
2. 實作／檢查隔離 fractional budget ledger，沿用原 event 時序；非排程日 hold。不得修改 V11 source、模型、結果或 ledger。入口使用 `python -B` 避免 parent bytecode 寫入。
3. 執行 `python -B -m unittest discover -s ppo_v12/tests -v`；驗證排程/尾差/成本、合成 NAV、現金與股份守恆、事件、lagged interest、未來資料 sentinel、缺值／資金不足停止，以及基準不使用 policy observation。
4. 驗證通過後執行 `python -B ppo_v12/scripts/compare_existing.py`，僅計算兩基準、讀取五模型結果。禁止覆寫已完成報告；重複檢查應讀既有產物或使用新的獨立輸出 namespace。
5. 核對 48 筆支出共 $1m、初末日期、1,002 transitions、原 B&H 重現、NAV及現金守恆、全部指標有限。保存完整 hashes、attempt、交易與 session ledger。
6. 報告兩基準與五模型 final wealth、CAGR、最大回撤、成本、平均曝險；列逐種子差值、勝數、中位數。MDD 差以模型減基準，正值為較小回撤。
7. 更新 `dev/devlog.md`，提供規格、Prompt、驗證與報告連結，**完成本次比較後停止**。

## 禁止事項與停止條件

不得新訓練、fine-tune、模型推論、部署、調參、只挑最佳 seed、刪失敗、把定額 B&H 插入 reward/observation/PPO loss、把已知 2022–2025 稱 untouched test，或把改基準稱模型改善。

若資料/hash/期間不同、任一 seed 缺失或 technical_pass=false、NAV不有限、原 B&H 無法重現、排程缺月、現金不足或守恆失敗：停止比較、保存 traceback 及已產生 artifacts、明確報告問題，不用替代數據補洞。

本次為 adaptive known-period diagnostic，正式三fold採用狀態 NOT_EVALUATED。不依本次結果自動 abort、採用或啟動訓練；由使用者判斷。

## 必要輸出

繁體中文 11 節規格及本 Prompt；月買及原 B&H daily ledger；48 筆購買；兩基準 metrics；五 seeds 與差值、中位數、勝數的 Markdown/JSON；單元測試 log、來源/輸出 hashes、所有 attempts；延續歷史的 devlog。

## 未來訓練階段（本次未授權）

只有使用者後續明確要求才可執行。預設 pilot seeds 0–3；初始化請求 200,000，2048-step rollout 預期 200,704；每次更新 32,768，總量必須揭露，不能稱全程僅 200k。保留 2010 起 chronological 訓練、初始 2021 截止、63-transition 窗口、126 半衰期 80/20 抽樣、VIX 動態時鐘、frozen scaler、warm-start fresh rollouts、30/20/50 reward、beta=.1、cap=5、V11 PPO 超參數與既有 gate／成本／時序。測試與後續部署允許已觀測資料因果更新，不能使用未來資訊或重置實際帳戶。

四種子只是 pilot，不替代正式三 folds × 五 seeds：至少兩 folds 正 paired medians、15 pairs 至少 10 勝、MDD 惡化 <=2pp、成本增加 <=10%；reward 與排程對照分別判定，不混 V10 結果。更換基準仍不改學習目標；若要變更訓練方法，先建立新的明確研究契約。2022–2025 不可用於選參數。
