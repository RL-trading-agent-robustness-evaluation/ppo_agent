# Development log

All timestamps use Asia/Taipei (UTC+08:00) and are recorded to minute precision.
This file is append-only for project file changes and Git/GitHub operations. Each
entry identifies whether a change was made by Codex, the user, or an external tool.

## 2026-09-27

### 14:57 — Codex — Repository scaffold commit

- Recreated `victim_ppo` as a clean downstream PPO repository.
- Added `victimagent` as a Git submodule pinned to the team V4 handoff commit
  `5701400076bd3ad4fa80dbc86d5687f480c26021`.
- Added the PPO V4 configuration, training/evaluation separation, data-loading
  boundary, provenance capture, and JSON/Markdown reporting scaffold.
- Created local Git commit `03a7c4aa0a82c34d0816dd1a166127cd1577937c`
  (`feat: scaffold PPO victim on pinned environment V4`).
- No GitHub push was performed.

### 15:03 — User — Distribution snapshot supplied

- Supplied `data/raw/stkdistributions_spy_tlt_gld_2009_2025.csv` for the pinned
  victimagent ingestion schema.
- File is licensed/local data and remains excluded from Git.

### 15:48 — User — DGS3MO snapshot replaced

- Supplied and renamed `data/raw/DGS3MO_2009_2025.csv` with coverage beginning
  2009-01-02 for the warm-up period.
- File is local data and remains excluded from Git.

### 15:49 — User — DTB3 snapshot replaced

- Supplied and renamed `data/raw/DTB3_2009_2025.csv` with coverage beginning
  2009-01-02 for the warm-up period.
- File is local data and remains excluded from Git.

### 15:56 — Codex — Development logging policy established

- Created `dev/devlog.md` at the user's request.
- Future project-file modifications and Git/GitHub operations must append a
  timestamped entry here in the same change set.
- Verified that all four frozen raw filenames required by the pinned V4 pipeline
  exist under `data/raw/`.

### 16:04 — Codex — PPO readiness implementation started

- Added a linear PPO learning-rate schedule from `3e-4` to `3e-5`.
- Added an outer-repository wrapper for the pinned victimagent data pipeline so
  raw licensed data and generated outputs remain outside the submodule.
- Added a V4 integration preflight that verifies the exact submodule pin,
  successful Gate 1 evidence, actual `SingleAssetTradingEnvV4` construction,
  10-dimensional observations, three shared actions, and known-period exclusion.
- Added PPO contract and learning-rate unit tests and documented the new setup.
- Separated smoke and formal output paths, prohibited formal timestep overrides,
  and required a frozen configuration plus clean worktree for formal runs.
- No formal training or GitHub operation was performed.

### 16:17 — Codex — Isolated runtime installed

- Created the Git-ignored `.venv` using native Windows Python 3.14.4.
- Installed both projects in editable mode with the declared development and RL
  dependencies, including Stable-Baselines3 2.9.0 and CPU PyTorch 2.14.0.
- Replaced an intermediate MSYS Python 3.12 environment because it could not use
  the available native NumPy/PyTorch wheels; no project source or data was lost.

### 16:18 — Codex — Shared Gate 1 data pipeline passed

- Ran the pinned victimagent data pipeline through `scripts/build_shared_data.py`.
- Generated canonical local interim and processed data under `data/interim/` and
  `data/processed/`; these licensed-derived files remain excluded from Git.
- Generated `reports/gate1_report.json`, `reports/GATE1_REPORT.md`,
  `reports/raw_manifest.json`, and `reports/missing_distribution_events.csv`.
- Gate 1 passed with zero failures. The two upstream warnings are the declared
  absence of a bundled external exchange calendar and missing embedded download
  timestamps; source hashes are recorded.
- Corrected the PPO preflight to read the upstream report's canonical
  `gate_1_status` field.

### 16:20 — Codex — Test and preflight corrections

- The first combined test command ran pinned tests from the outer repository and
  produced path-related failures because upstream tests intentionally resolve
  `configs/` from the victimagent root. No implementation failure was found.
- Reran the pinned victimagent suite from its repository root: all runnable tests
  passed, with seven declared skips and upstream warnings only.
- Ran the outer PPO suite: both tests passed.
- Corrected the V4 preflight cutoff comparison to compare pandas timestamps
  rather than a timestamp with a string after its first run raised `TypeError`.

### 16:22 — Codex — PPO V4 smoke passed and configuration frozen

- V4 preflight passed with the exact submodule SHA, actual
  `SingleAssetTradingEnvV4`, observation shape `(10,)`, action weights
  `[0.0, 0.5, 1.0]`, 2,264 train rows, 757 validation rows, and no known-period
  access.
- Completed the separate 4,096-step PPO seed-0 smoke run and deterministic
  2019–2021 validation. Evaluation parameters, optimizer, and scaler were
  unchanged; reward–NAV identity error was approximately `5e-16`.
- Generated local smoke checkpoint/log/trajectory artifacts and the committed
  evidence candidates `reports/PPO_SMOKE_SEED0.{json,md}` and
  `reports/ppo_v4_preflight.json`.
- Froze `configs/ppo_v4.yaml` before formal training, including the linear
  learning-rate schedule from `3e-4` to `3e-5`.
- No formal 200,000-step run or GitHub operation was performed.

### 16:24 — Codex — Frozen-config verification and artifact policy

- Reran PPO tests, V4 preflight, and the 4,096-step PPO smoke against the frozen
  configuration; all passed.
- Kept smoke reports and editable-install metadata local through `.gitignore`.
- Made formal `PPO_CLEAN_SEED<N>` reports trackable for audit retention.
- Scoped the formal clean-worktree check to research source, scripts, configs,
  project metadata, and the victimagent submodule so reports from completed
  seeds do not block later preregistered seeds.

### 16:25 — Codex — Readiness changes prepared for local commit

- Final outer PPO tests passed after all readiness and artifact-policy changes.
- Prepared one local Git commit containing the implementation, frozen config,
  Gate 1 evidence, V4 preflight evidence, tests, documentation, and this log.
- No GitHub push was requested or performed.
- The first local commit attempt was rejected because the managed sandbox exposed
  `.git` as read-only; no index or commit was created by that attempt.

