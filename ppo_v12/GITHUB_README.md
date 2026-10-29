# V12 publication

Branch `v12` of `RL-trading-agent-robustness-evaluation/ppo_agent` preserves the
V11 repository at its root and adds this V12 package in `ppo_v12/`.

The completed run is `reports/full_training/full500k_20261008_3/COMPARISON.md`
and its adjacent JSON. All five seeds passed. Earlier attempts and historical
status snapshots are retained; their old PIDs do not indicate current jobs.
Formal adoption is `NOT_EVALUATED`; this is an adaptive known-period training
budget comparison, not an untouched test.

## Contents

- Frozen V12 specs, configs, code, tests and engine-change hashes.
- The isolated engine in `training/` (the Python package retains the inherited
  `victim_ppo.v11` name, but its config is `v12train1_500k`).
- Summary metrics, five seed records, smoke records and provenance.
- Checkpoint hashes and original local paths for identifying trained models.

Licensed CRSP inputs and processed data, per-session ledgers and purchase
records, model checkpoints, optimizer/resume states and runtime artifacts are
not published. Consequently a clone contains the implementation and results,
not the trained weight files. Obtain authorized data and all five checkpoint
sets separately for attack evaluation.

## Local setup

Install the repository's dependencies, including stable-baselines3==2.9.0,
then provide the canonical market, distribution and VIX files under
`ppo_v12/training/data/`, using the paths and hashes in
`ppo_v12/training/configs/ppo_v11.yaml`.

From the repository root:

```powershell
python ppo_v12/training/scripts/run_ppo_v11.py --preflight
```

The training CLI adds its own source paths. Model initialization requests
500,000 transitions (501,760 actual); each causal fine-tune uses 32,768.
The completed seeds each used 1,583,104 transitions. Do not start historical
supervisors just to inspect results: they contain reserved tags and refuse
overwriting existing runs.

Original benchmark/supervisor scripts and audit manifests retain their
historical sibling checkout references (`ppo_v11`, `ppo_v10`) and absolute
paths. Those comparisons require the original local layout and artifacts;
they are not portable one-command reproductions. The root of this branch is
the V11 reference source; use the isolated V12 training entry point above for
V12. Source snapshots and manifests are preserved without rewriting historical
provenance to imply it was generated at a new location.
