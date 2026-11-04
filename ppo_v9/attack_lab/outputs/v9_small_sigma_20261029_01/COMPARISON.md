# Small V9 Gaussian sigma sweep

Relative-reward V9 only; five frozen model seeds per fold, noise seed 123. Same Gaussian directions across strengths.
Noise on eleven standardized market features. Clean gate/rate/ledger; no retraining; no known-period access.
All clean trajectories reproduce original hashes; noisy evaluations preserve model/optimizer/scaler and reward invariants.
This is a one-noise-seed diagnostic, not a robustness conclusion or verified formal attack-spec evaluation.

| Fold | Sigma | B&H return | Clean median | Noisy median | Wins vs B&H | Median paired change (pp) |
|---|---:|---:|---:|---:|---:|---:|
| fold_1 | 0.1 | 13.66% | 8.23% | 4.74% | 1/5 | -1.598 |
| fold_1 | 0.2 | 13.66% | 8.23% | 5.55% | 1/5 | -2.299 |
| fold_2 | 0.1 | 15.11% | 16.67% | 16.67% | 5/5 | +0.000 |
| fold_2 | 0.2 | 15.11% | 16.67% | 17.24% | 5/5 | +0.000 |
| fold_3 | 0.1 | 98.02% | 94.05% | 94.05% | 0/5 | +0.000 |
| fold_3 | 0.2 | 98.02% | 94.05% | 94.05% | 0/5 | +0.000 |

| Fold | Model seed | Sigma | Clean return | Noisy return | Change (pp) |
|---|---:|---:|---:|---:|---:|
| fold_1 | 0 | 0.1 | 15.678% | 15.678% | +0.000 |
| fold_1 | 0 | 0.2 | 15.678% | 15.738% | +0.060 |
| fold_1 | 1 | 0.1 | 8.229% | 4.712% | -3.516 |
| fold_1 | 1 | 0.2 | 8.229% | 0.108% | -8.120 |
| fold_1 | 2 | 0.1 | -0.103% | 3.828% | +3.930 |
| fold_1 | 2 | 0.2 | -0.103% | 5.546% | +5.648 |
| fold_1 | 3 | 0.1 | 6.343% | 4.745% | -1.598 |
| fold_1 | 3 | 0.2 | 6.343% | 2.089% | -4.254 |
| fold_1 | 4 | 0.1 | 10.520% | 7.331% | -3.189 |
| fold_1 | 4 | 0.2 | 10.520% | 8.220% | -2.299 |
| fold_2 | 0 | 0.1 | 16.668% | 16.668% | +0.000 |
| fold_2 | 0 | 0.2 | 16.668% | 16.668% | +0.000 |
| fold_2 | 1 | 0.1 | 16.161% | 16.161% | +0.000 |
| fold_2 | 1 | 0.2 | 16.161% | 21.789% | +5.629 |
| fold_2 | 2 | 0.1 | 17.239% | 17.239% | +0.000 |
| fold_2 | 2 | 0.2 | 17.239% | 17.239% | +0.000 |
| fold_2 | 3 | 0.1 | 18.012% | 18.012% | +0.000 |
| fold_2 | 3 | 0.2 | 18.012% | 18.012% | +0.000 |
| fold_2 | 4 | 0.1 | 14.892% | 16.327% | +1.435 |
| fold_2 | 4 | 0.2 | 14.892% | 16.327% | +1.435 |
| fold_3 | 0 | 0.1 | 96.235% | 96.235% | +0.000 |
| fold_3 | 0 | 0.2 | 96.235% | 96.235% | +0.000 |
| fold_3 | 1 | 0.1 | 97.067% | 97.067% | +0.000 |
| fold_3 | 1 | 0.2 | 97.067% | 97.067% | +0.000 |
| fold_3 | 2 | 0.1 | 78.957% | 78.957% | +0.000 |
| fold_3 | 2 | 0.2 | 78.957% | 78.957% | +0.000 |
| fold_3 | 3 | 0.1 | 71.181% | 80.163% | +8.982 |
| fold_3 | 3 | 0.2 | 71.181% | 75.857% | +4.676 |
| fold_3 | 4 | 0.1 | 94.046% | 94.046% | +0.000 |
| fold_3 | 4 | 0.2 | 94.046% | 94.046% | +0.000 |
