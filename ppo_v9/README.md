# PPO V9 random Gaussian noise experiments

This package contains the attack implementation, saved experiment summaries, a literature review, and clean/perturbed trade logs for the 2023–2025 diagnostic. Models are frozen during evaluation.

## Reports

- [Attack comparison across 2015–2021](reports/RGN_ATTACK_REPORT_CRITERIA_V1/REPORT.md)
- [Attack benchmark literature review](reports/ATTACK_BENCHMARK_LITERATURE_REVIEW.md)
- [2023–2025 paired results](attack_lab/outputs/rgn_2023_2025_sigma02/comparison.json)
- [Policy-to-execution trace](attack_lab/outputs/v9_pipeline_trace_20261020_02/TRACE_REPORT.md)

## Latest diagnostic

The 2023–2025 run uses the five original fold-3 relative-reward checkpoints, model seeds 0–4, their original training scaler, sigma 0.2, and noise seed 123. Independent Gaussian noise affects eleven standardized price features. Execution prices, interest rates, and the trend gate flag stay clean. The five-session holding rule applies.

| Seed | Clean wealth | Noisy wealth |
|---|---:|---:|
| 0 | 1.705454 | 1.656490 |
| 1 | 1.677356 | 1.677356 |
| 2 | 1.716623 | 1.786546 |
| 3 | 1.643509 | 1.643509 |
| 4 | 1.675957 | 1.746653 |

SPY buy-and-hold wealth is 1.835864. The gate forces full exposure on 693 of 751 decision days (92.3%). One model loses wealth, two gain, and two stay unchanged. Both group medians are 1.677356. This diagnostic uses one noise realization and accesses the known 2023–2025 period.

## Reproduction

Use the repository environment with Stable Baselines3, PyTorch, pandas, NumPy, and Gymnasium. Initialize the pinned victimagent submodule. Restore the original checkpoint files at the paths recorded under `experiments/v9_ppo/records/` and supply the original licensed market inputs through 2025. Then run from the repository root, choosing a new output directory.

```powershell
python ppo_v9/attack_lab/rgn_2023_2025.py --output ppo_v9/attack_lab/outputs/rgn_2023_2025_replay
python -m unittest discover -s ppo_v9/attack_lab -p test_prepare.py
```

The evaluator checks checkpoint hashes, scaler fingerprints, model/optimizer preservation, protected input columns, and the reward/NAV identity. Source output directories refuse overwrites. The published evaluator supports both this repository layout and the original sibling-folder layout.

## Artifact coverage

Saved summaries and dataset manifests cover completed runs only. The latest diagnostic includes all ten trade trajectories. The exact five fold-3 gate-relative checkpoints trained through 2018, their five original validation trajectories, and their five known-period diagnostic trajectories are delivered under the repository-root `artifacts/` directory. Every file matches the supplied checklist SHA-256. See [delivery manifest](delivery/MANIFEST.json) and [checksums](delivery/SHA256SUMS.txt), whose paths resolve from the repository root.

The repository is public. The user explicitly authorized public publication of these checkpoints and trajectories. Licensed input datasets, derived observation CSVs, other checkpoint binaries, and large diagnostic arrays remain in the local experiment archive. Historical reports retain original absolute source paths and hashes, so those paths refer to the original workspace. Referenced files outside the delivered pack require that archive. No formal Attack Spec V1 document was verified during these experiments.
