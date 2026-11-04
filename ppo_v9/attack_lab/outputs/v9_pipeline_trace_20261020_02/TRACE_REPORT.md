# Policy-input and execution pipeline audit

100 attacked rollouts: relative-reward models, folds 2/3, model seeds 0–4, noise seeds 123–132.
Every policy input is checked against the saved CSV at float32 precision; every probability argmax against model.predict.
All actual traces must reproduce saved action, gate, executed target, true price, trade notional and NAV paths exactly.
Same-state counterfactual replaces only market features, preserving the attacked rollout account and rule states. It is a one-step diagnostic, not a second simulated portfolio.

| Fold | Model seed | Steps | Changed inputs | Changed probabilities | Feature-induced action flips | Gate masked | Lock masked | Reach target | Actual trade changes vs clean | Same final wealth /10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fold_2 | 0 | 5010 | 5010 | 5010 | 24 | 24 | 0 | 0 | 0 | 10/10 |
| fold_2 | 1 | 5010 | 5010 | 5010 | 132 | 125 | 7 | 0 | 0 | 10/10 |
| fold_2 | 2 | 5010 | 5010 | 5010 | 113 | 105 | 8 | 0 | 0 | 10/10 |
| fold_2 | 3 | 5010 | 5010 | 5010 | 114 | 108 | 3 | 3 | 19 | 7/10 |
| fold_2 | 4 | 5010 | 5010 | 5010 | 206 | 191 | 12 | 3 | 41 | 7/10 |
| fold_3 | 0 | 7560 | 7560 | 7560 | 268 | 257 | 7 | 4 | 51 | 7/10 |
| fold_3 | 1 | 7560 | 7560 | 7560 | 131 | 127 | 4 | 0 | 0 | 10/10 |
| fold_3 | 2 | 7560 | 7560 | 7560 | 113 | 107 | 6 | 0 | 0 | 10/10 |
| fold_3 | 3 | 7560 | 7560 | 7560 | 287 | 270 | 9 | 8 | 151 | 5/10 |
| fold_3 | 4 | 7560 | 7560 | 7560 | 148 | 142 | 4 | 2 | 50 | 8/10 |

Counts sum across ten noise seeds; repeated dates are not independent market samples.
Gate/lock columns classify same-state action flips. Actual trade changes compare full attacked and clean rollouts and can include later account feedback.
Daily CSVs include policy probabilities, actions, gate, lock, rejection reason, trades and NAV. NPZs contain actual 20-D inputs and their same-state clean-feature counterfactuals.
