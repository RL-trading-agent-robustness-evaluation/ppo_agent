# Attack performance benchmarks for PPO V9: literature review

Research date: 2026-10-29. Scope: primary research on attacks against RL trading/portfolio agents, with broader PPO/RL observation-attack research as methodological context. This is a focused review, not an exhaustive systematic review or meta-analysis. No new experiments were run.

## Finding

The reviewed evidence does **not establish a universal 30% performance-drop requirement**, or a defensible numerical expectation for Gaussian noise against our PPO V9 configuration. Published outcomes depend on the attack surface, perturbation budget, objective, transaction costs, market/task, victim, and measurement. A small or zero effect is a reportable result; it is neither proof of general robustness nor an invalid experiment merely because it misses 30%.

We found a plausible source of confusion, but cannot establish the historical provenance of the project's original 30% expectation. No reviewed source justifies treating that number as a universal success threshold.

## 1. The paper containing a prominent 30% claim

Chen et al., *Adversarial Attacks Against Reinforcement Learning-Based Portfolio Management Strategy* (IEEE Access, 2021), describes its proposed mechanisms as over 30% more effective **relative to FGSM/I-FGSM**. This compares attack methods; it does not prescribe a 30% loss from the clean baseline. [Publisher abstract](https://ieeexplore.ieee.org/abstract/document/9386090), [authors' institutional record](https://scholar.nycu.edu.tw/en/publications/adversarial-attacks-against-reinforcement-learning-based-portfoli/).

The accepted manuscript defines accumulated portfolio value (APV) as `Pt/P0` in Eq. 29, p. 12. Selected matched values from Tables 7–8, p. 17, are:

| Enhanced EIIE setting | Clean APV | RL-attack APV | Derived relative wealth loss |
|---|---:|---:|---:|
| S&P portfolio, 2013 | 1.7502 | 0.7976 | 54.43% |
| S&P portfolio, 2014 | 2.6251 | 0.0086 | 99.67% |
| FX, 2011–2012 | 0.9390 | 0.9363 | 0.29% |
| FX, 2012–2013 | 1.0823 | 1.0435 | 3.58% |

These use optimized RL attacks and 0.25% transaction costs, not Gaussian noise. The main TWSE design perturbs bid/ask volumes; the daily-data extensions do not establish a directly comparable feature-noise budget. These rows are periods, not independent training seeds. The accessed manuscript is accepted but unedited; its FX table repeats two rows, so we do not recompute its aggregate. [Primary manuscript, §§V–VI and Tables 7–8](https://www.researchgate.net/publication/350361055_Adversarial_Attacks_Against_Reinforcement_Learning-Based_Portfolio_Management_Strategy/fulltext/605be2c1a6fdccbfea049a2c/Adversarial-Attacks-Against-Reinforcement-Learning-Based-Portfolio-Management-Strategy.pdf).

## 2. A useful PPO comparison: random versus optimized perturbations

Zhang et al., *Robust Deep Reinforcement Learning against Adversarial Perturbations on State Observations* (NeurIPS 2020), Table 1, p. 8, reports mean episode rewards over 50 episodes:

| Vanilla PPO task | Normalized-observation L∞ budget | Clean reward | Random attack reward | Strong attack reward |
|---|---:|---:|---:|---:|
| Hopper | 0.075 | 3167.6 ± 541.6 | 2915.2 ± 677.7 | 733.8 ± 44.6, RS+MAD |
| Walker2d | 0.05 | 4619.5 ± 38.2 | 4480.0 ± 465.3 | 913.7 ± 54.3, RS |

Derived declines of the reported means are approximately **8.0% versus 76.8%**, and **3.0% versus 80.2%**, respectively. These are reward changes, not financial wealth losses; error bars are standard deviations, not confidence intervals. Separately, the paper evaluates at least 15 training runs per PPO setting. Do not relabel the table's 50 episodes as 50 model seeds. [Primary paper, Table 1 and §4](https://proceedings.neurips.cc/paper_files/paper/2020/file/f0eb6568ea114ba6e293f903c34d7488-Paper.pdf).

The authors' implementation samples the random attack uniformly in `[-epsilon, epsilon]`, not from a Gaussian. This supports separating random and optimized attacks, but supplies no matched Gaussian benchmark for trading. [Author code, `apply_attack`, lines 629–632](https://github.com/huanzhang12/SA_PPO/blob/master/src/policy_gradients/agent.py#L629-L632).

## 3. Trading attacks can report large percentages on different denominators

Ataiefard and Hemmati, *Gray-box Adversarial Attack of Deep Reinforcement Learning-based Trading Agents* (2023 preprint), trains an A2C adversary that trades inside the ABIDES simulated market. This changes the market through interaction rather than adding independent noise to a frozen observation file. Its initial capital equals the victim's capital, making the attack budget unlike our sigma budget.

Table III reports cumulative-return reduction (`CR`) from **63.68% to 95.36%** across nine trader/market pairs. Eq. 6 defines this as an average of `(P_clean-P_attack)/P_clean` over 50 episodes, where P denotes returns. This cannot automatically be read as a percentage of terminal wealth. The reported **214.17%** reward reduction is from mean reward **0.623 to −0.711**, demonstrating how crossing zero produces percentages above 100%. The paper also reports per-step return reduction separately. Its 50 episodes do not establish 50 independent training seeds. [Primary manuscript, §§III–IV, Eqs. 6–7, Tables I/III/IV](https://arxiv.org/html/2309.14615v1).

## 4. Other relevant evidence and reporting limits

Piazza et al., *Adversarial Attacks on Deep Algorithmic Trading Policies* (CAMLIS 2021), attacks Basic DQN and TensorTrade DQN using observation delays and targeted/untargeted FGSM and C&W. The perturbation experiment modifies the newest observation tuple with plausibility constraints. Table 4 counts attack opportunities and action-change failures; Figures 3–8 show performance/reward/net-worth effects. The authors explicitly distinguish reward damage from net-worth damage. We did not convert plotted curves into purported exact wealth-drop percentages. This is evidence for reporting both behavioral and financial outcomes, not a 30% threshold. [Primary proceedings paper, §§5.2–5.3, pp. 8–11](https://ceur-ws.org/Vol-3095/paper6.pdf).

Huang et al., *Adversarial Attacks on Neural Network Policies* (2017), evaluates FGSM against DQN, TRPO and A3C Atari policies across perturbation norms and strengths. It trains five policies but retains at most three satisfying a performance-selection rule, and averages ten rollouts per result. These selected-policy game-score results cannot be transferred directly to all-seed trading performance or Gaussian corruption. [Primary paper, §§5.1–5.3, pp. 5–7](https://rll.berkeley.edu/adversarial/arXiv2017_AdversarialAttacks.pdf).

## 5. What this means for our experiment

Our existing [PPO V9 report](C:/03_College/Grad_proj/ppo_v9/reports/RGN_ATTACK_REPORT_CRITERIA_V1/REPORT.md) evaluates noise on 11 standardized price features while keeping execution prices, the trend gate input, and the interest-rate feature clean. It therefore answers a narrower question than a market-manipulation attack. The latest headline uses five frozen models and one common noise seed, with additional repeated-noise evidence elsewhere in the report. Its scope must remain explicit.

There is no defensible pooled 'usual attack drop' across the studies above: the denominators and experiments differ. In particular, large optimized-attack outcomes should not become an expected minimum for an independent random-noise baseline. We also did not identify a directly matched published experiment with our feature representation, frozen PPO architecture, gate/lock rules, data folds, and Gaussian budgets.

### Keep the financial measurements unambiguous

For each model i and attack replicate j, let W be terminal wealth divided by initial wealth, and R = W − 1. Report:

| Quantity | Definition | Interpretation |
|---|---|---|
| Paired wealth change | `100 × (W_attack / W_clean − 1)` | Negative means wealth damage; primary effect size |
| Return change | `100 × (R_attack − R_clean)` | Percentage points, not relative percent |
| Harm frequency | Proportion with `W_attack < W_clean − tolerance` | Any measurable harm; not statistical significance |
| Material-harm frequency | Proportion whose paired wealth loss exceeds a preregistered threshold | Practical damage under an explicitly chosen criterion |
| B&H comparison | Wealth and win count against the same clean B&H baseline | Trading competitiveness, separate from attack efficacy |
| Mechanism measures | Input changes, raw action changes, executed trades, gate/lock suppression | Shows where perturbations stop affecting behavior |

Illustration: wealth 2.00 → 1.80 is a 10% wealth decline, a return change of −20 percentage points (100% → 80%), and a 20% reduction in profit. Calling all three 'performance dropped 20%' would obscure the result.

### Recommended prospective protocol — our proposal, not a literature rule

1. Retire 30% as a literature-backed pass/fail rule unless an exact, applicable source is produced. Preserve any previously registered threshold in historical reporting; label revisions explicitly and apply them prospectively.
2. Keep clean, random, and optimized attack arms on the same frozen checkpoints, data periods, cost model, and permitted features. Include a range of justified budgets and report complete curves rather than picking the most damaging test result.
3. Match budgets explicitly. An unbounded Gaussian sigma is not an L∞ cap. For a bounded comparison, define clipping/projection and match permitted support, attacked dimensions, frequency, and normalization. Retain the original unbounded-Gaussian experiment as a separate baseline; record realized perturbation norms.
4. Report paired effects per model first, then aggregates with uncertainty. Multiple noise seeds on one model measure attack randomness; they do not create extra independent trained models. Use model-level or hierarchical uncertainty, keep folds visible, and acknowledge that five models give limited precision. Market-time dependence also prevents treating every trading day as an independent sample.
5. Decide the smallest practically meaningful loss before the next confirmatory evaluation, based on the project's economic objective. Compare optimized attacks against random perturbations at the matched budget and test whether damage is reproducible. Keep statistical evidence, economic materiality, and B&H wins as separate conclusions.

The immediate research conclusion is that the 30% expectation lacks support as a universal benchmark in the reviewed evidence. The next experiment should estimate the attack effect under a fixed, defensible threat model rather than tune until an arbitrary loss target appears.
