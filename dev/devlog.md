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

### 16:49 — Codex — Five-seed PPO aggregation added

- Added `--aggregate` to require exactly formal seeds 0–4 and reject missing,
  short-budget, failed, known-period, mutable-evaluation, or mixed-provenance runs.
- Added mean, median, sample standard deviation, minimum, maximum, and IQR across
  every retained seed for the declared validation metrics.
- Added checkpoint, scaler, and trajectory SHA-256 hashes to each future run report.
- Added an aggregation failure test and documented the command.
- No formal training or GitHub operation had been performed at this point.

### 16:51 — Codex — Aggregator verified and prepared for local commit

- All three outer PPO tests passed after adding strict aggregation.
- Prepared the aggregation implementation, test, documentation, and devlog for
  a local Git commit before starting any formal seed.
- No GitHub push was requested or performed.

### 17:04 — Codex — Formal PPO seed 0 completed

- Ran the preregistered formal seed 0 for exactly 200,000 training steps at
  outer commit `2bb1d6676cdd8b87f269d967054a97725c636e48` and victimagent V4 commit
  `5701400076bd3ad4fa80dbc86d5687f480c26021`.
- The linear learning rate descended from `3e-4` toward the declared `3e-5`
  floor; all recorded optimizer diagnostics remained finite.
- Deterministic 2019–2021 validation technically passed with no known-period
  access, unchanged parameters/optimizer/scaler, and reward–NAV error
  `1.44e-15`.
- Validation metrics: final wealth `1.487132×`, CAGR `14.1673%`, annualized
  excess return `14.0949%`, Sharpe versus lagged cash `0.769862`, maximum
  drawdown `-33.6840%`, turnover `21.052604`, and transaction costs
  `$25,074.53`.
- Generated `reports/PPO_CLEAN_SEED0.{json,md}` and local checkpoint, scaler,
  training-log, and validation-trajectory artifacts with SHA-256 hashes.
- Deliberately did not commit between formal seeds so every seed will record the
  identical research-code provenance required by the strict aggregator.
- No GitHub push was performed.

## 2026-09-30

### 11:20 — Codex — V4 formal study preserved as historical reference

- Verified and staged the completed formal reports for PPO seeds 0–4 together
  with the strict five-seed aggregate report (`PPO_CLEAN_BENCHMARK`).
- Preserved the reports at outer research-code commit
  `2bb1d6676cdd8b87f269d967054a97725c636e48` and victimagent V4 commit
  `5701400076bd3ad4fa80dbc86d5687f480c26021`.
- The aggregate retained all five seeds and reported median validation CAGR
  `23.9079%`, Sharpe `1.245645`, maximum drawdown `-27.0204%`, and final wealth
  `1.900426x`; these results remain historical V4 evidence and will not be
  overwritten by the V9 study.
- Prepared a dedicated local commit before changing the victimagent submodule or
  adding any V9 implementation. No GitHub push was performed.

### 11:21 — Codex — V4 historical-reference commit created

- Created local commit `f0961ae` (`docs: preserve PPO V4 formal results`) with
  all five formal seed reports, the aggregate benchmark, and their development
  log history.
- No GitHub push was performed.

### 11:26 — Codex — PPO V9 reward-ablation scaffold implemented

- Advanced the victimagent submodule to the team V9 pin
  `f4988db5c98d9248533c4ca852cc48ca3f6b7aad`.
- Added the frozen two-arm PPO reward-ablation configuration, preserving the
  linear `3e-4` to `3e-5` learning-rate schedule, `n_steps=2048`, and the
  documented 500,000 requested / 501,760 realized SB3 transition budget.
- Added V9 PPO construction, training, true-NAV evaluation, action tracing,
  state-responsiveness checks, append-only cell records, paired aggregation,
  TensorBoard output, a serial 30-cell matrix launcher, and user commands.
- Added explicit six-file raw-data gating. `victimagent/data/raw/` remains empty;
  no V4 outer-repository data was copied or reused.
- Added PPO V9 contract and SPY relative-reward tests. The combined outer PPO
  suite and pinned victimagent V8/V9 suites passed: 27 tests, with 10 upstream
  Gymnasium float32 precision warnings and no failures.
- Ran data-free preflight successfully. It reports `WAITING_FOR_DATA`, lists all
  six missing raw inputs, confirms the exact V9 pin, and confirms that the known
  period was not accessed.
- Prepared a local V9 scaffold commit after the successful verification run.
- No training, data build, known-period access, or GitHub push was performed.

### 12:55 — User/Codex — V9 raw inputs supplied and canonical data built

- The user supplied the six V8/V9 contract inputs under the outer ignored
  `data/raw/` directory, plus supplemental DFF, DTB3, full-history DGS3MO, and
  VIX files.
- Verified that both CRSP market files and both distribution files exactly match
  the victimagent V9 pinned SHA-256 values. The two FRED DGS3MO inputs are
  intentionally unpinned by the team contract.
- Copied only the six contract inputs into the ignored
  `victimagent/data/raw/` directory. Supplemental series remain unused because
  the frozen V8/V9 observation contract declares `external_series: none`.
- Ran the official victimagent V8 data builder. It passed with 8,286 market
  rows, 130 distribution events, and the required processed hashes:
  market `281ed8c0530720b5cedf4abd7e9288375dad4fe073f55b0b814560ebcb54af9b`;
  events `a981523e00b05e804091bf4671835bade363559215403614344bf5a099541668`.
- Updated the V9 CLI so preflight performs full upstream fold/data/hash checks
  automatically once all six raw inputs exist, while retaining the useful
  `WAITING_FOR_DATA` result when inputs are absent.
- Adjusted formal cleanliness validation to allow the official builder's
  untracked `victimagent/reports/v8/DATA_BUILD.json`, while still rejecting any
  tracked submodule modification and enforcing the exact pinned submodule SHA.
- No training or known-period access was performed.

### 12:57 — Codex — First V9 control smoke stopped before training

- Attempted the preregistered 20,000-step `ppo_v9_gate_nav` smoke cell for
  fold 1, seed 0.
- Stable-Baselines3 stopped during logger construction because the TensorBoard
  package was not installed. No model learning step occurred and no checkpoint
  or validation trajectory was produced.
- Retained the failed smoke reservation and record as append-only audit
  evidence; no file was overwritten or deleted.
- Added TensorBoard as an explicit runtime dependency and added smoke retry tags
  so corrected attempts receive new identities such as `_smoke_retry1`.

### 13:01 — Codex — Both corrected PPO V9 smoke cells passed

- Installed the declared TensorBoard 2.21.0 runtime and retained it only in the
  ignored local virtual environment.
- Completed new append-only 20,000-requested-step (`20,480` realized) smoke
  attempts for both `ppo_v9_gate_nav` and `ppo_v9_gate_relative`, fold 1 seed 0.
- Both runs produced finite losses, updated model parameters, preserved the
  train-only scaler, preserved model/optimizer/scaler state during evaluation,
  terminated naturally, passed the true reward–NAV identity at approximately
  `1.54e-15`, and passed the state-responsiveness check.
- Both smoke policies produced validation wealth `1.110581x`; this equality is
  not interpreted as a performance result because smoke runs are technical
  checks and the trend gate dominated most decisions in this window.
- TensorBoard logs, checkpoints, training CSV logs, validation trajectories,
  attempt reservations, and run records were written to their V9 PPO-specific
  ignored artifact namespaces. No known-period data was accessed.
- Reran the combined outer PPO and pinned victimagent V8/V9 suites after the
  data build and smoke corrections: all 27 tests passed with upstream precision
  warnings only. Prepared the verified formal-run code for a local commit.

### 14:09 — Codex — Updated handoff compliance audit paused formal matrix

- Read the user's updated PPO/A2C V9 handoff in full and audited the frozen PPO
  runner and completed records against its literal field and report contract.
- Confirmed that the data, folds, 20-dimensional observations, official V9
  environment factory, gate, rewards, actions, costs, seeds, PPO settings,
  deterministic true-NAV evaluation, responsiveness check, and 501,760 realized
  transitions for a 500,000 requested budget were compliant.
- Found reporting gaps: PPO-local rather than victimagent arm names in `arm`, no
  explicit `algorithm`, `total_timesteps`, or outer-repository commit fields,
  no hard assertion of the frozen DQN buy-and-hold wealth values, and an
  incomplete validation summary generator.
- Stopped hidden supervisor PID 44284 before it could launch further batches.
  Four already-running fold-1 control seeds were subsequently interrupted; all
  reservations and any completed records remain intact and were not deleted or
  overwritten.
- Registered append-only formal record revision `v9spec1`, changed new records
  to use victimagent arm names with a separate PPO-local label, added algorithm,
  requested total timesteps, outer commit provenance, and exact per-fold DQN
  buy-and-hold assertions. Existing records remain historical/noncompliant-schema
  evidence and will not be included in the revised formal aggregation.
- No final training or 2022–2025 known-period access was performed.

### 14:24 — Codex — V9 validation summary contract completed

- Expanded revised aggregation to generate append-only
  `reports/v9_ppo/VALIDATION_SUMMARY.{json,md}` with per-arm/per-fold median
  log wealth versus buy-and-hold, win counts, paired reward deltas, all required
  risk/cost/exposure/gate statistics, five-day-rule blocked share, technical and
  behavioral issue lists, baselines, and the pinned DQN descriptive reference.
- Changed the configured final-arm identifier to the required shared name
  `v9_gate_relative_reward`; it remains preregistered rather than selected.

### 20:12 — User/Codex — Local validation stopped after active batch

- At the user's request, stopped validation supervisor PID 40160 so it cannot
  launch another local batch. The four already-running fold-1 relative-reward
  cells continue to completion and remain append-only.
- Added `notebooks/ppo_v9_colab_resume.ipynb` for a Google Drive/Colab handoff.
  The notebook checks out frozen PPO commit `2bbad36`, verifies victimagent pin
  `f4988db`, restores existing records and artifacts, rebuilds and verifies the
  canonical data, refuses ambiguous prior attempts, trains only missing
  `v9spec1` cells with at most two CPU workers, syncs evidence to Drive after
  each PASS, and aggregates only after all 30 records pass.
- Prepared ignored Git bundle `artifacts/ppo_v9_code_2bbad36.bundle` so Colab
  can reproduce the unpushed frozen outer commit without publishing it first.
- The notebook does not perform final training or read the known period. Raw
  licensed files remain Drive-local and are never committed.
- This notebook and log remain uncommitted while the current formal cells finish
  so every validation run continues to report frozen code commit `2bbad36`.

### 20:18 — Codex — Local batch completed and Colab state frozen

- Confirmed the four active fold-1 relative-reward cells completed at 501,760
  transitions with PASS records. Local corrected validation now contains 20 of
  30 cells, and no local training worker or supervisor remains active.
- Prepared ignored archive `artifacts/ppo_v9_state_after20.zip` containing the
  append-only experiment state and all completed V9 PPO checkpoints and
  trajectories required for the Colab continuation.
- Updated the notebook to restore this archive before discovering the remaining
  cells; therefore Colab will train only the final 10 missing cells.
- Validated the notebook JSON and compiled every Python code cell; verified the
  state archive contains 115 entries. Prepared the notebook and this log for a
  local commit; ignored transfer archives remain outside Git.

### 20:24 — Codex — Colab code source changed to GitHub

- Updated the Colab notebook to clone the PPO repository directly from GitHub's
  `v9` branch and then check out frozen training commit `2bbad36`.
- Removed the Drive code-bundle requirement. The Drive state archive remains
  required because checkpoints and trajectories are prohibited from Git, and
  the six licensed raw inputs remain Drive-only under the team contract.
- Revalidated the notebook JSON and compiled all Python cells after the change;
  prepared the notebook and log for a local commit. No GitHub push was made.

### 21:10 — User/Codex — Remaining validation resumed locally

- At the user's request, audited the append-only matrix and confirmed exactly 20
  compliant PASS records, 10 missing cells, and zero ambiguous attempts for the
  missing cells.
- Resumed only the 10 missing `v9_gate_relative_reward` cells (folds 2 and 3,
  seeds 0–4) with a hidden supervisor limited to two simultaneous processes and
  one Torch thread per process.
- The supervisor will run compliant aggregation only after all 30 records pass;
  final training and known-period confirmation remain out of scope for this run.

### 22:08 — Codex — V4/V9 interim comparison image prepared

- Added `reports/v9_ppo/render_ppo_interim_summary.py` and generated
  `reports/v9_ppo/ppo_v4_v9_interim_summary.png` in the same academic green
  table style as the supplied A2C reference image.
- The image compares SPY buy-and-hold, historical V4 PPO, and the completed V9
  NAV-control arm on the common 2019–2021 window. It also reports the completed
  fold-1 reward ablation without treating it as a final conclusion.
- Marked V9 explicitly `INCOMPLETE`: the snapshot contains 22/30 formal PASS
  records (15/15 control and 7/15 relative-reward), with eight reward cells
  still outstanding. The two active fold-2 workers were not interrupted.

