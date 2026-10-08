# Run V11 from a fresh clone

The published branch includes code, research specifications, prompts, tests,
audit records and all five result summaries. Historical V4/V9/V10 records remain
available. Data and model/runtime artifacts are excluded from Git.

```powershell
git clone --recurse-submodules --branch v11-adaptive-policy https://github.com/RL-trading-agent-robustness-evaluation/ppo_agent.git ppo_v11
cd ppo_v11
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ./victimagent
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Use Python 3.11 or newer and the pinned SB3 2.9.0 from pyproject.toml. Original
runs used Python 3.14, torch 2.14.0, Gymnasium 1.3.0, numpy 2.5.3 and pandas
2.3.3; different numerical/runtime versions need their own recorded provenance.

Provide the following inputs from your authorized data source or the project's
data owner. Their exact hashes are recorded in config and the source manifest:

- `data/processed/v8/spy_market_1993_2025.csv`
- `data/interim/v8/spy_cash_distributions_1993_2025.csv`
- `data/raw/VIXCLS_1990_2026.csv`

The processed SPY/cash-distribution inputs were generated with the V8 pipeline
and include licensed data. A generic downloaded SPY CSV cannot reproduce these
inputs. VIX is the recorded FRED VIXCLS snapshot, rather than a newly revised
download. No dataset is silently fabricated or substituted.

```powershell
.\.venv\Scripts\python.exe -B scripts/prepare_local_clone.py
.\.venv\Scripts\python.exe -B scripts/run_ppo_v11.py --preflight
.\.venv\Scripts\python.exe -B -m pytest tests/test_v11.py tests/test_v11_reward_sequence.py tests/test_local_clone.py
```

The setup helper checks source against the recorded parent Git commits,
checks exact dataset hashes and the submodule commit, then creates ignored
`LOCAL_SOURCE_MANIFEST.json`. It omits generated egg-info and accounts for
Git LF/CRLF conversion. The historical `SOURCE_MANIFEST.json` and all original
result hashes remain intact. Modified source, wrong submodule or missing/wrong
inputs cause setup to stop; it is not a way to bypass source/data validation.

Before new formal or known-period training, run all three smoke arms with your
own unique tag. Follow the main README and implementation Prompt for the full
matrix or explicitly authorized direct adaptive diagnostic. Original models are
not included: local users train new models, rather than loading old checkpoints.
Full chronological runs can take hours and produce substantial local logs.

The existing 2022–2025 report is an adaptive known-period diagnostic, not an
untouched test or a completed 45-cell development selection.
