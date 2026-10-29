# PPO V10 規格：NAV 與連續正報酬趨勢獎勵

**狀態：已凍結，可供正式實作與訓練使用。** 這是 PPO V9 的最小變更延伸。V10 保留 V9 的資料、三個驗證 fold、環境、觀察值、動作、交易成本、SMA-200 trend gate、五交易日規則、PPO 超參數、評估方式及稽核要求，只替換 treatment arm 的訓練獎勵。其他組員或代理若發現本文件與凍結的 V9 契約衝突，應停止執行並回報，不得自行推測或修改。

## 1. 研究問題

> 在固定 V9 trend gate 與 PPO 設定的前提下，於真實 NAV log return 之外，對「連續為正的投資組合報酬」提供逐步增加且有上限的訓練 bonus，是否能改善扣除交易成本後的驗證期財富，同時不造成不可接受的回撤或交易成本？

V10 測試的是 **reward shaping**。它不宣稱這個 bonus 是新的經濟利潤，也不把 bonus 加入 NAV、PnL 或正式績效指標。

## 2. V9 中完全保留的項目

- 固定 `victimagent` V9 commit：`f4988db5c98d9248533c4ca852cc48ca3f6b7aad`。若實作 V10 wrapper 必須更新該 repository，應建立明確的新 commit 並在所有紀錄中保存 V9 parent commit 與 V10 implementation commit，不得覆寫 V9。
- 使用 V9 的 `v9_bundle`、`SingleAssetTradingEnvV4`、`TradeDecisionEnv(..., "v3_min_hold_five")`、`TrendGateWrapper` 與相同資料建置流程。
- SPY、1993 warm-up、1994-02-01 起的 development data、20 維 observation、每個 fold 僅用 train 資料擬合的 scaler。
- 動作仍為目標權重 `{0.0, 0.5, 1.0}`，long-only、無槓桿、初始資金 `$1,000,000`、交易成本 `10 bps`。
- 決策時鐘不變：close `t` 決策、true raw open `t+1` 執行、close `t+1` 產生下一 observation 與 reward。
- SMA-200 trend gate 不變：observed total-return feature price 高於其 200-session SMA 時，要求 100% SPY；否則讓 PPO action 通過。五交易日規則仍可延後 gate 的要求。
- PPO 架構與超參數、每次正式 run 的 requested `500,000` transitions、SB3 實際 `501,760` transitions、deterministic validation、所有 seed 均保留等規則不變。
- 驗證及報告一律使用真實 NAV log reward。2022–2025 仍是已開啟的 known-period confirmation，不是 untouched test，也不得用於 V10 選擇或調參。

## 3. 唯一主要變更：連續正報酬 bonus

### 3.1 基礎 NAV reward

每一步的真實 NAV reward 為：

```text
r_nav[t] = log(NAV_close[t+1] / NAV_close[t])
```

NAV 必須來自原始 V4 ledger，並已反映持倉損益、現金利息、股利、公司行動及交易成本。

### 3.2 正報酬 streak

`streak[t]` 表示截至當前 reward step 為止，連續出現 `r_nav > 0` 的步數：

```text
if r_nav[t] > 0:
    streak[t] = streak[t-1] + 1
else:
    streak[t] = 0
```

- 每次 environment reset 時，`streak = 0`。
- `r_nav == 0` 視為 streak 中斷。
- streak 只能使用當下已經產生的 `r_nav[t]` 與過去狀態，不得讀取未來價格或 future return。
- streak 以投資組合的 **net NAV log return** 判斷，不使用 SPY return、未扣成本報酬、訓練 loss 或 value estimate。

### 3.3 V10 treatment 的訓練 reward

凍結值：

```text
beta = 0.10
streak_cap = 5
```

先計算額外 bonus：

```text
bonus[t] = beta * max(min(streak[t], streak_cap) - 1, 0) * max(r_nav[t], 0)
```

再得到 treatment training reward：

```text
r_train[t] = r_nav[t] + bonus[t]
```

因此：

| 當前連續正報酬步數 | 正報酬的 reward 倍率 |
|---:|---:|
| 1 | 1.0× |
| 2 | 1.1× |
| 3 | 1.2× |
| 4 | 1.3× |
| 5 或以上 | 1.4× |

若 `r_nav <= 0`，`bonus = 0`，training reward 完全等於真實 NAV reward。使用與正報酬幅度成比例且有上限的 bonus，可避免只因極小的正報酬就取得固定高額獎勵，也避免無限延長 streak 造成 reward 爆增。

`beta` 與 `streak_cap` 已凍結。正式結果出現後不得調整並沿用 V10 名稱；任何不同數值都必須建立新的版本或明確標示為探索性實驗。

## 4. 實驗 arms 與因果對照

正式研究只需要兩個 arms，兩者都啟用完全相同的 V9 trend gate：

| PPO arm | Trend gate | 訓練 reward | 驗證 reward |
|---|---|---|---|
| `ppo_v10_gate_nav` | 開啟 | `r_nav` | `r_nav` |
| `ppo_v10_gate_nav_streak` | 開啟 | `r_nav + bonus` | `r_nav` |

兩個 arms 只允許在 training reward transformation 上不同。模型架構、初始化 seed、fold、訓練步數、rollout、optimizer、scaler、資料、執行規則及評估流程必須相同。

V9 的 `ppo_v9_gate_relative` 結果可作歷史描述，但不得與 V10 treatment 直接配對來宣稱 streak bonus 的因果效果。V10 不使用 SPY-relative training reward，避免同時改變兩種 reward shaping。

## 5. Wrapper 行為與必要欄位

實作使用獨立的 `PositiveStreakRewardWrapper`，放置於 V9 gate 之外，且只在 treatment 的 **training environment** 使用。wrapper 不得更改價格、觀察值、action、gate、成交、ledger、NAV、成本、終止條件或 episode 長度。

每一步至少保留以下 `info`：

```text
nav_log_reward
positive_return_streak
positive_streak_bonus
training_reward
agent_action
gate_forced_full
gated_action
executed_action
```

control arm 也應輸出同名欄位，其中 `positive_streak_bonus = 0`、`training_reward = nav_log_reward`，以便逐步配對與稽核。

評估 environment 禁止套用 `PositiveStreakRewardWrapper`。評估輸出的 reward 必須直接等於 `nav_log_reward`。

## 6. Fold、seed 與正式矩陣

沿用三個 V8/V9 expanding-window validation folds：

| Fold | Training | Validation |
|---|---|---|
| `fold_1` | 1994-02-01 至 2014-12-31 | 2015-01-01 至 2016-12-31 |
| `fold_2` | 1994-02-01 至 2016-12-31 | 2017-01-01 至 2018-12-31 |
| `fold_3` | 1994-02-01 至 2018-12-31 | 2019-01-01 至 2021-12-31 |

正式矩陣：

```text
3 folds × 5 seeds × 2 arms = 30 independent PPO trainings
seeds = [0, 1, 2, 3, 4]
requested transitions per run = 500,000
expected SB3 transitions per run = 501,760
```

每個 fold/seed 的兩個 arms 必須成對執行並保留所有結果，不得挑選最佳 seed。若資源只允許較短實驗，必須標示為 `pilot`，不得與正式 V9/V10 結果混合或用於正式結論。

## 7. 驗證、主要比較與採用規則

### 7.1 主要 reward-shaping 比較

對每個成對 fold/seed，以扣除成本後的 deterministic validation final wealth 計算：

```text
delta_reward = log(W_gate_nav_streak / W_gate_nav)
```

必須報告全部 15 個 paired deltas、每個 fold 的 median、每個 fold 的 win count、pooled distribution，以及三個 fold median 的算術平均。不得只報告最佳 seed。

### 7.2 必報績效與風險指標

- final wealth、CAGR、annualized excess return relative to lagged cash；
- Sharpe relative to lagged cash，並保留 numerator 與 denominator；
- maximum drawdown、annualized volatility、average exposure；
- total turnover、trade count、transaction cost；
- action distribution、gate-forced share、gate-override share、five-day-rule blocked share；
- streak 長度分布、`bonus > 0` 的 step 比例、bonus 總和、`sum(bonus) / sum(abs(r_nav))`；
- executable cash、SPY buy-and-hold、constant-50 與 deterministic gate-only baselines。

### 7.3 預先固定的採用規則

V10 treatment 只有在以下條件全部成立時才可取代 control：

1. 三個 fold 中至少兩個 fold 的 paired median `delta_reward > 0`；
2. 15 個 paired runs 中至少 10 個 `delta_reward > 0`；
3. treatment 的三-fold median maximum-drawdown magnitude 不得比 control 惡化超過 `2.0` percentage points；
4. treatment 的三-fold median transaction cost 不得高於 control 超過 `10%`。

這是已凍結的工程採用規則，不是統計顯著性或經濟優越性的證明。任何門檻變更都必須建立新的版本，不得沿用 V10 名稱。

## 8. 正式訓練前必須通過的測試

1. **公式單元測試：** 人工輸入正、正、正、負、正的 `r_nav`，確認 streak 為 `1,2,3,0,1`，bonus 與上表一致。
2. **零與負報酬：** `r_nav <= 0` 時 bonus 必須為 0，並立即重設 streak。
3. **上限測試：** streak 超過 5 後倍率仍固定為 1.4×。
4. **reset 測試：** episode reset 後 streak 不得延續。
5. **因果性測試：** 修改未來價格不得改變較早 step 的 observation、gate flag、NAV reward、streak 或 bonus。
6. **同 action path 測試：** control 與 treatment 使用相同 action sequence 時，NAV、成交、成本、observation 與 termination 必須逐步完全相同；只有 training reward 與 bonus diagnostic 可不同。
7. **reward identities：** 完整 episode 應滿足：

   ```text
   sum(nav_log_reward) = log(NAV_end / NAV_start)
   sum(training_reward) = log(NAV_end / NAV_start) + sum(positive_streak_bonus)
   ```

   絕對誤差上限為 `1e-10`。
8. **evaluation 測試：** 兩 arms 的 evaluation reward 都必須等於真實 NAV log reward，且不含 bonus。
9. **PPO smoke：** 兩 arms 都必須通過 finite observation/loss、checkpoint reload、model/scaler immutability、自然終止與 state-responsiveness checks。

## 9. Provenance 與輸出要求

每次 run 使用 append-only 紀錄，至少包含：

- V10 spec SHA-256、PPO code commit、V9 parent commit、V10 wrapper implementation commit；
- config、fold、processed market、events 與 scaler SHA-256；
- arm、fold、seed、requested/realized transitions、套件版本；
- `beta`、`streak_cap` 與完整 reward 公式字串；
- checkpoint、training log、TensorBoard log、validation trajectory SHA-256；
- 完整 stdout/stderr、技術檢查、全部績效與 streak diagnostics。

使用新的 V10 artifact namespace，例如：

```text
experiments/v10_ppo_records/
experiments/v10_ppo_attempts/
artifacts/ppo_v10/
reports/v10_ppo/
tensorboard/ppo_v10/
```

不得覆寫、移動或重新標記 V9 的 records、checkpoints、reports 或 known-period artifacts。

## 10. 結果詮釋限制

- V10 validation 使用的是已反覆用於開發的 pre-2022 folds，因此只能視為 development evidence。
- 2022–2025 已在專案層級開啟，只能稱為 known-period confirmation。V10 必須先依上述規則完成選擇、凍結 final protocol，並取得明確授權後才能執行一次。
- training reward 的 bonus 不是可交易收益。正式圖表與績效比較只能使用真實 ledger NAV。
- 即使 treatment 勝過 control，也只支持「此 streak reward 在固定 V9 gate 與 PPO protocol 下有幫助」。它不能單獨證明模型穩定勝過 SPY buy-and-hold。
- 若 bonus 增加報酬但同時明顯提高回撤、換手或成本，必須完整揭露，不得只呈現 final wealth。

## 11. 實作者簡要檢查表

- [ ] 凍結 `beta=0.10`、`streak_cap=5` 或在任何 run 前完成修訂。
- [ ] 僅新增 reward wrapper 與 V10 runner/config/reporting，不修改 V4 ledger 或 V9 gate。
- [ ] control 與 treatment 共用完全相同的 PPO 設定。
- [ ] smoke 通過後才啟動 30-cell 正式矩陣。
- [ ] validation 永遠使用 unshaped NAV reward。
- [ ] 保留全部 seed、失敗紀錄及 hashes。
- [ ] 不使用 2022–2025 調整 V10。

