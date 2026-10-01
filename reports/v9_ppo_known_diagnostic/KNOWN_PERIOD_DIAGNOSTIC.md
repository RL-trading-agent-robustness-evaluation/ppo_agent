# PPO V9 2022–2025 known-period diagnostic

> 本結果使用 PPO V9 fold-3 checkpoints（train 截至 2018），在已開封 2022–2025 做延伸診斷。它不是 untouched test，也不是 train 截至 2021 的最終模型確認。各模型訓練窗口不同的比較僅為描述性比較。

固定標籤：`known_period_diagnostic_fold3_not_final_test`

實際期間：2022-01-03 至 2025-12-31，1002 steps

| Seed | Final wealth | CAGR | Excess return | Sharpe | Volatility | Max DD | Exposure | Turnover | Cost | log gap vs B&H |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.4919× | 10.54% | 7.08% | 0.471 | 15.05% | -24.29% | 89.88% | 23.08× | $20,743 | +0.0059 |
| 1 | 1.1672× | 3.95% | 0.80% | 0.056 | 14.30% | -33.94% | 87.35% | 27.99× | $21,452 | -0.2395 |
| 2 | 1.5213× | 11.08% | 7.80% | 0.473 | 16.50% | -20.38% | 90.70% | 19.12× | $18,358 | +0.0254 |
| 3 | 1.2430× | 5.60% | 2.31% | 0.168 | 13.81% | -26.88% | 86.21% | 23.07× | $19,659 | -0.1766 |
| 4 | 1.3449× | 7.71% | 4.39% | 0.304 | 14.46% | -27.77% | 89.24% | 26.03× | $22,466 | -0.0978 |

Wins versus executable SPY buy-and-hold: **2/5**

## Five-seed distribution

| Metric | Median | IQR | Min | Max |
|---|---:|---:|---:|---:|
| final_wealth | 1.344896 | 0.248844 | 1.167182 | 1.521299 |
| cagr | 0.077056 | 0.049395 | 0.039487 | 0.110829 |
| annualized_mean_excess_return | 0.043871 | 0.047698 | 0.007982 | 0.077960 |
| sharpe_excess_cash | 0.303560 | 0.303225 | 0.055872 | 0.472819 |
| annualized_volatility | 0.144621 | 0.007562 | 0.138131 | 0.164959 |
| max_drawdown | -0.268827 | 0.034824 | -0.339411 | -0.203821 |
| average_exposure | 0.892361 | 0.025301 | 0.862075 | 0.907045 |
| total_turnover | 23.078183 | 2.954544 | 19.124966 | 27.990810 |
| total_transaction_cost | 20743.037145 | 1792.856185 | 18358.093110 | 22466.339060 |
| log_wealth_gap_vs_buy_and_hold | -0.097821 | 0.182481 | -0.239545 | 0.025427 |

## Technical checks

- All five original records: formal PASS, 501,760 actual steps.
- All checkpoint, scaler, config, processed-data and trajectory hashes are in the JSON.
- Model parameters, optimizer and scaler remained unchanged for every rollout.
- Reward–NAV identity passed within `1e-10`; evaluation used true NAV reward.
- Executable buy-and-hold final wealth: `1.48310441640342`.

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_ppo_v9_known_diagnostic.py
```
