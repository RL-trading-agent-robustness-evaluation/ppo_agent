# PPO V4 / V9 / V10：2022–2025 描述性比較

> 本比較使用已開封的 2022–2025 known period，不是 untouched test，也不是最終模型確認。模型的訓練窗口、特徵、訓練步數與執行規則不同，因此只能作描述性比較，不可據此選模型、seed、checkpoint 或 reward。

固定標籤：`known_period_descriptive_comparison_v4_v9_v10_not_final_test`

共同評估期間：2022-01-03 至 2025-12-31，1002 steps。

所有數值均為五個 seeds 的中位數。

| Model | Final wealth | CAGR | Excess return | Sharpe | Volatility | Max DD | Exposure | Turnover | Cost | Wins vs B&H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V4 | 1.1942× | 4.55% | 1.60% | 0.101 | 15.81% | -24.90% | 84.57% | 20.10× | $21,489 | 1/5 |
| V9 relative fold 3 | 1.3449× | 7.71% | 4.39% | 0.304 | 14.46% | -26.88% | 89.24% | 23.08× | $20,743 | 2/5 |
| V10 streak pilot | 1.4948× | 10.59% | 7.22% | 0.462 | 15.73% | -21.87% | 90.33% | 23.98× | $20,954 | 3/5 |
| SPY buy-and-hold | 1.4831× | 10.38% | 7.35% | 0.417 | 17.63% | -24.71% | 97.59% | 1.00× | $999 | — |

## Comparability limits

- All 2022-2025 results are known-period diagnostics, not untouched tests.
- V4 uses 10 features, a 2010-2018 training window, 200k steps, and its V4 execution contract.
- V9 uses fold 3, a 1994-2018 training window, 500k requested steps, and the V9 gate/min-hold contract.
- V10 is a 200k fold-1 development pilot trained through 2014 with the V9 environment and streak-shaped training reward.
- The comparison is descriptive and must not be used for model, seed, checkpoint, or reward selection.

完整 per-seed 成績、IQR、min/max、hash 與 technical checks 請見 JSON。
