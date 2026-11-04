# V9 Gaussian feature-noise diagnostic

Noise sigma=0.05, noise seed=123; 11 standardized price features.
Interest rate, trend gate, execution prices, costs and ledger use the clean contract. No retraining.
One noise realization; formal attack-spec alignment has not been verified. Development folds only; known period not accessed.

All 30 clean trajectories exactly reproduce their original hashes and wealth. All noisy evaluations pass state and reward checks.

| Arm | Fold | B&H return | Clean median return | Noisy median return | Clean wins | Noisy wins | Median paired change (pp) |
|---|---|---:|---:|---:|---:|---:|---:|
| v9_trend_gate | fold_1 | 13.66% | 9.17% | 5.65% | 0/5 | 0/5 | -1.610 |
| v9_trend_gate | fold_2 | 15.11% | 12.02% | 12.02% | 2/5 | 2/5 | +0.000 |
| v9_trend_gate | fold_3 | 98.02% | 81.21% | 81.21% | 0/5 | 0/5 | +0.000 |
| v9_gate_relative_reward | fold_1 | 13.66% | 8.23% | 6.34% | 1/5 | 1/5 | +0.000 |
| v9_gate_relative_reward | fold_2 | 15.11% | 16.67% | 16.67% | 4/5 | 5/5 | +0.000 |
| v9_gate_relative_reward | fold_3 | 98.02% | 94.05% | 94.05% | 0/5 | 0/5 | +0.000 |

## Every checkpoint

| Arm | Fold | Seed | Clean return | Noisy return | B&H return | Change (pp) |
|---|---|---:|---:|---:|---:|---:|
| v9_trend_gate | fold_1 | 0 | 6.613% | 5.003% | 13.661% | -1.610 |
| v9_trend_gate | fold_1 | 1 | 10.002% | 5.653% | 13.661% | -4.349 |
| v9_trend_gate | fold_1 | 2 | 9.172% | 11.157% | 13.661% | +1.985 |
| v9_trend_gate | fold_1 | 3 | 5.962% | 4.027% | 13.661% | -1.935 |
| v9_trend_gate | fold_1 | 4 | 12.887% | 12.887% | 13.661% | +0.000 |
| v9_gate_relative_reward | fold_1 | 0 | 15.678% | 15.678% | 13.661% | +0.000 |
| v9_gate_relative_reward | fold_1 | 1 | 8.229% | 6.221% | 13.661% | -2.007 |
| v9_gate_relative_reward | fold_1 | 2 | -0.103% | 1.116% | 13.661% | +1.219 |
| v9_gate_relative_reward | fold_1 | 3 | 6.343% | 6.343% | 13.661% | +0.000 |
| v9_gate_relative_reward | fold_1 | 4 | 10.520% | 10.520% | 13.661% | +0.000 |
| v9_trend_gate | fold_2 | 0 | 12.018% | 12.018% | 15.106% | +0.000 |
| v9_trend_gate | fold_2 | 1 | 24.844% | 24.844% | 15.106% | +0.000 |
| v9_trend_gate | fold_2 | 2 | 15.477% | 15.477% | 15.106% | +0.000 |
| v9_trend_gate | fold_2 | 3 | 10.718% | 10.718% | 15.106% | +0.000 |
| v9_trend_gate | fold_2 | 4 | 9.156% | 9.156% | 15.106% | +0.000 |
| v9_gate_relative_reward | fold_2 | 0 | 16.668% | 16.668% | 15.106% | +0.000 |
| v9_gate_relative_reward | fold_2 | 1 | 16.161% | 16.161% | 15.106% | +0.000 |
| v9_gate_relative_reward | fold_2 | 2 | 17.239% | 17.239% | 15.106% | +0.000 |
| v9_gate_relative_reward | fold_2 | 3 | 18.012% | 18.012% | 15.106% | +0.000 |
| v9_gate_relative_reward | fold_2 | 4 | 14.892% | 16.327% | 15.106% | +1.435 |
| v9_trend_gate | fold_3 | 0 | 70.688% | 70.688% | 98.018% | +0.000 |
| v9_trend_gate | fold_3 | 1 | 79.105% | 77.663% | 98.018% | -1.442 |
| v9_trend_gate | fold_3 | 2 | 81.214% | 81.214% | 98.018% | +0.000 |
| v9_trend_gate | fold_3 | 3 | 83.254% | 83.254% | 98.018% | +0.000 |
| v9_trend_gate | fold_3 | 4 | 89.855% | 91.562% | 98.018% | +1.707 |
| v9_gate_relative_reward | fold_3 | 0 | 96.235% | 96.235% | 98.018% | +0.000 |
| v9_gate_relative_reward | fold_3 | 1 | 97.067% | 97.067% | 98.018% | +0.000 |
| v9_gate_relative_reward | fold_3 | 2 | 78.957% | 78.957% | 98.018% | +0.000 |
| v9_gate_relative_reward | fold_3 | 3 | 71.181% | 84.903% | 98.018% | +13.722 |
| v9_gate_relative_reward | fold_3 | 4 | 94.046% | 94.046% | 98.018% | +0.000 |

Full CAGR, Sharpe, drawdown, exposure, turnover, costs and action distributions are in COMPARISON.json.
Noise may improve or worsen returns. Random Gaussian noise is a robustness diagnostic, not a targeted optimization attack.
The trend gate masks agent actions in uptrend sessions, so these results do not test gate or raw-price perturbation.
