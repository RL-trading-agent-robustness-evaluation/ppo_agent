# V9 Gaussian noise: ten seeds

Noise seeds 123–132; sigma=0.05 on 11 standardized price-derived features. Same frozen 30 checkpoints; no retraining.
Clean gate, rates and execution market. All current clean replays match original trajectory hashes.
Seed 123 is reused from the verified prior diagnostic; seeds 124–132 are new. Formal attack-spec alignment remains unverified.

| Group | Fold | B&H return | Clean median return | Noisy median return | Clean wins | Noisy wins | Median paired change (pp) | Worst / best change (pp) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v9_gate_relative_reward | fold_1 | 13.66% | 8.23% | 6.53% | 1/5 | 10/50 | +0.000 | -4.174 / +5.268 |
| v9_gate_relative_reward | fold_2 | 15.11% | 16.67% | 16.67% | 4/5 | 43/50 | +0.000 | -2.537 / +1.852 |
| v9_gate_relative_reward | fold_3 | 98.02% | 94.05% | 94.05% | 0/5 | 1/50 | +0.000 | -2.902 / +20.293 |
| v9_trend_gate | fold_1 | 13.66% | 9.17% | 7.76% | 0/5 | 0/50 | +0.000 | -7.575 / +1.985 |
| v9_trend_gate | fold_2 | 15.11% | 12.02% | 12.89% | 2/5 | 20/50 | +0.000 | -11.219 / +10.986 |
| v9_trend_gate | fold_3 | 98.02% | 81.21% | 81.21% | 0/5 | 0/50 | +0.000 | -5.293 / +3.424 |

Noisy medians pool 5 models x 10 noise realizations. Win counts are repeated evaluations, not independent training-seed evidence.

| Group | Fold | Model seed | Clean return | Noisy return range | Wins across noise seeds |
|---|---|---:|---:|---:|---:|
| v9_gate_relative_reward | fold_1 | 0 | 15.678% | 15.678% to 15.738% | 10/10 |
| v9_gate_relative_reward | fold_1 | 1 | 8.229% | 4.055% to 12.022% | 0/10 |
| v9_gate_relative_reward | fold_1 | 2 | -0.103% | -0.708% to 5.165% | 0/10 |
| v9_gate_relative_reward | fold_1 | 3 | 6.343% | 5.948% to 6.712% | 0/10 |
| v9_gate_relative_reward | fold_1 | 4 | 10.520% | 8.221% to 10.520% | 0/10 |
| v9_gate_relative_reward | fold_2 | 0 | 16.668% | 16.668% to 16.668% | 10/10 |
| v9_gate_relative_reward | fold_2 | 1 | 16.161% | 16.161% to 16.161% | 10/10 |
| v9_gate_relative_reward | fold_2 | 2 | 17.239% | 17.239% to 17.239% | 10/10 |
| v9_gate_relative_reward | fold_2 | 3 | 18.012% | 15.475% to 19.864% | 10/10 |
| v9_gate_relative_reward | fold_2 | 4 | 14.892% | 14.892% to 16.327% | 3/10 |
| v9_gate_relative_reward | fold_3 | 0 | 96.235% | 93.334% to 100.481% | 1/10 |
| v9_gate_relative_reward | fold_3 | 1 | 97.067% | 97.067% to 97.067% | 0/10 |
| v9_gate_relative_reward | fold_3 | 2 | 78.957% | 78.957% to 78.957% | 0/10 |
| v9_gate_relative_reward | fold_3 | 3 | 71.181% | 71.181% to 91.474% | 0/10 |
| v9_gate_relative_reward | fold_3 | 4 | 94.046% | 92.915% to 94.046% | 0/10 |
| v9_trend_gate | fold_1 | 0 | 6.613% | 2.781% to 8.096% | 0/10 |
| v9_trend_gate | fold_1 | 1 | 10.002% | 2.427% to 11.062% | 0/10 |
| v9_trend_gate | fold_1 | 2 | 9.172% | 8.346% to 11.157% | 0/10 |
| v9_trend_gate | fold_1 | 3 | 5.962% | 4.027% to 6.659% | 0/10 |
| v9_trend_gate | fold_1 | 4 | 12.887% | 12.887% to 12.887% | 0/10 |
| v9_trend_gate | fold_2 | 0 | 12.018% | 12.018% to 12.149% | 0/10 |
| v9_trend_gate | fold_2 | 1 | 24.844% | 13.625% to 24.844% | 7/10 |
| v9_trend_gate | fold_2 | 2 | 15.477% | 15.477% to 17.405% | 10/10 |
| v9_trend_gate | fold_2 | 3 | 10.718% | 10.718% to 16.989% | 1/10 |
| v9_trend_gate | fold_2 | 4 | 9.156% | 9.156% to 20.142% | 2/10 |
| v9_trend_gate | fold_3 | 0 | 70.688% | 67.583% to 72.867% | 0/10 |
| v9_trend_gate | fold_3 | 1 | 79.105% | 77.663% to 79.105% | 0/10 |
| v9_trend_gate | fold_3 | 2 | 81.214% | 79.395% to 81.214% | 0/10 |
| v9_trend_gate | fold_3 | 3 | 83.254% | 83.254% to 86.678% | 0/10 |
| v9_trend_gate | fold_3 | 4 | 89.855% | 84.561% to 91.651% | 0/10 |
