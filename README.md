# PPO victim agent — shared environment contracts

## V9 reward-ablation study

The completed V4 five-seed study remains under `reports/PPO_CLEAN_*`. New work
uses the victimagent submodule pinned to
`f4988db5c98d9248533c4ca852cc48ca3f6b7aad` and the frozen configuration in
`configs/ppo_v9_reward_ablation.yaml`.

Raw licensed inputs belong in `victimagent/data/raw/`, not the outer `data/`
directory. Until all six files are present, preflight reports
`WAITING_FOR_DATA` and data building/training is intentionally blocked:

```powershell
.\.venv\Scripts\python.exe scripts\run_ppo_v9.py --preflight
```

After supplying the files, build and verify the canonical inputs, then run both
20,000-step smoke cells:

```powershell
.\.venv\Scripts\python.exe scripts\run_ppo_v9.py --build-data
.\.venv\Scripts\python.exe scripts\run_ppo_v9.py --preflight
.\.venv\Scripts\python.exe scripts\run_ppo_v9.py --smoke --arm ppo_v9_gate_nav
.\.venv\Scripts\python.exe scripts\run_ppo_v9.py --smoke --arm ppo_v9_gate_relative
```

Formal cells use `--run --arm ARM --fold FOLD --seed SEED`. The convenience
PowerShell runner executes all 30 cells serially and aggregates them:

```powershell
.\scripts\run_ppo_v9_matrix.ps1 -Threads 1
```

The requested budget remains 500,000. With `n_steps=2048`, SB3 completes the
last rollout and records 501,760 actual transitions. Both reward arms use the
same PPO configuration and evaluation always uses true NAV reward.

## Historical V4 study

This is the downstream PPO implementation for the RL trading-agent robustness project.
All market mechanics, the 10-dimensional observation, accounting, splits, and V4
close-to-next-close reward timing come from the pinned `victimagent` submodule.
The outer repository owns only PPO configuration, training, frozen deterministic
validation, provenance, and reporting.

## Frozen integration

- Shared repository: `RL-trading-agent-robustness-evaluation/victimagent`
- V4 handoff pin: `5701400076bd3ad4fa80dbc86d5687f480c26021`
- Train: 2010-01-04 through 2018-12-31
- Validation: 2019-01-01 through 2021-12-31
- 2022-01-01 through 2025-12-31 is an already-opened known period and is not loaded
  by the development harness.
- Formal seeds: 0–4; budget: 200,000 steps per seed.
- PPO learning rate decays linearly from `3e-4` to `3e-5` over each run.

The older `../rltrade` tree is historical reference only and is neither imported nor
executed by this project.

## Setup

```powershell
git submodule update --init --recursive
python -m venv .venv
.venv\Scripts\python -m pip install -e ".\victimagent[rl]" -e ".[dev]"
```

Place canonical pipeline outputs at:

```text
data/processed/market_base_with_splits.csv
data/interim/distribution_events.csv
```

Build those outputs from the four frozen files in `data/raw/`, then run the
V4 integration preflight:

```powershell
.venv\Scripts\python scripts\build_shared_data.py --root .
.venv\Scripts\python scripts\run_ppo_preflight.py --root .
```

Run one formal seed only after the PPO configuration is accepted and the canonical
data files are present:

```powershell
.venv\Scripts\python scripts\run_ppo_clean_benchmark.py --seed 0 --root . --data-root .
```

After all five seeds finish, create the strict aggregate:

```powershell
.venv\Scripts\python scripts\run_ppo_clean_benchmark.py --aggregate --root .
```

Aggregation rejects missing seeds, short budgets, technical failures, known-period
access, mutable evaluation state, and mixed code/config/data/package provenance.

Before formal execution, run the separate smoke path:

```powershell
.venv\Scripts\python scripts\run_ppo_clean_benchmark.py --smoke --seed 0 --root . --data-root .
```

`--timesteps` is accepted only with `--smoke`. Formal execution requires a clean
committed worktree and always uses exactly 200,000 steps.

