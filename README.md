# PPO victim agent — shared environment V4

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

Before formal execution, run the separate smoke path:

```powershell
.venv\Scripts\python scripts\run_ppo_clean_benchmark.py --smoke --seed 0 --root . --data-root .
```

`--timesteps` is accepted only with `--smoke`. Formal execution requires a clean
committed worktree and always uses exactly 200,000 steps.

