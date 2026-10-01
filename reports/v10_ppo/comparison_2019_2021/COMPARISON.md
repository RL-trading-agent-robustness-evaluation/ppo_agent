# PPO model comparison, 2019–2021

All figures are medians across five seeds. Evaluation uses true NAV and 10 bps costs.

| Model | Final wealth | CAGR | Sharpe | Volatility | Max drawdown | Exposure | Turnover | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v4 | 1.9004× | 23.91% | 1.246 | 19.14% | -27.02% | 86.42% | 20.01× | $23,338 |
| v9_control | 1.8121× | 21.96% | 1.008 | 20.93% | -32.78% | 94.27% | 8.05× | $8,595 |
| v9_relative | 1.9405× | 24.77% | 1.163 | 18.35% | -29.04% | 95.06% | 7.07× | $7,734 |
| v10_streak | 1.7841× | 21.32% | 1.076 | 18.72% | -29.58% | 95.39% | 9.06× | $10,298 |
| SPY buy-and-hold | 1.9802× | 25.62% | 1.133 | 21.40% | -33.13% | 97.91% | 1.00× | $999 |

## Comparability

- V4 uses the historical 10-feature V4 protocol and 200k steps.
- V9 control and relative use the 20-feature V9 fold-3 protocol, train through 2018, and 500k steps.
- V10 streak uses the 20-feature V9 environment, its fold-1 scaler, train through 2014, and 200k steps.
- This is a descriptive common-period comparison, not a matched causal reward ablation.
