# PPO V9 validation summary

Known period accessed: **no**. Final arm was preregistered as `v9_gate_relative_reward`; validation does not select an arm.

| Arm | Fold | Median log(W/B&H) | Wins vs B&H | Median wealth | Sharpe | Max DD | Exposure | Turnover | Cost | Gate forced | Gate override | 5-day blocked |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v9_trend_gate | fold_1 | -0.0403 | 0/5 | 1.0917 | 0.409 | -13.87% | 0.903 | 8.01 | 7885.22 | 79.72% | 60.24% | 14.91% |
| v9_trend_gate | fold_2 | -0.0272 | 2/5 | 1.1202 | 0.396 | -18.90% | 0.963 | 6.02 | 7185.81 | 92.02% | 78.64% | 7.78% |
| v9_trend_gate | fold_3 | -0.0887 | 0/5 | 1.8121 | 1.008 | -32.78% | 0.943 | 8.05 | 8595.02 | 88.62% | 50.93% | 6.61% |
| v9_gate_relative_reward | fold_1 | -0.0490 | 1/5 | 1.0823 | 0.380 | -14.40% | 0.917 | 9.00 | 8549.12 | 79.72% | 66.20% | 14.91% |
| v9_gate_relative_reward | fold_2 | +0.0135 | 4/5 | 1.1667 | 0.570 | -17.87% | 0.966 | 5.03 | 5903.78 | 92.02% | 65.47% | 7.19% |
| v9_gate_relative_reward | fold_3 | -0.0203 | 0/5 | 1.9405 | 1.163 | -29.04% | 0.951 | 7.07 | 7733.52 | 88.62% | 34.66% | 8.60% |

## Paired reward contrast

| Fold | Median log(W relative/W NAV) | Positive pairs |
|---|---:|---:|
| fold_1 | -0.0163 | 2/5 |
| fold_2 | +0.0407 | 4/5 |
| fold_3 | +0.0218 | 3/5 |

Technical failures: 0; non-responsive: 0; near-constant policies: 1.

DQN values in the JSON are descriptive references from the pinned victimagent report, not paired PPO comparisons.
