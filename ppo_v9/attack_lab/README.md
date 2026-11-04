# Gaussian noise exercise for V9

Start by editing `perturb_values()` in `prepare.py`. It intentionally raises
`NotImplementedError`: the noise manipulation is yours to implement. The
surrounding code loads CSVs, copies clean data, validates output, checks replay,
and saves inspectable artifacts. There is no model inference or training here.

The attack specification was not found in the workspace. This is a proposed
exercise for additive noise on standardized observation features, not a claim
of compliance with that specification. Confirm attack location, sigma units,
columns, timing, correlations, clipping and gate treatment against the actual
spec before using outputs in an attack experiment. The manifest records this
limitation. A raw-price attack needs a different input contract and feature
recalculation; do not feed raw prices into this scaffold.

## Usual workflow

1. Freeze the clean model, data window, training-fitted scaler and baseline.
2. Decide where noise enters. For observation noise, keep true execution
   prices, dividends, interest accrual and NAV on the clean stream.
3. Draw epsilon from the specified Gaussian distribution and produce
   `x_adv = x_clean + epsilon`. Sigma is a standard deviation, not a variance.
   In standardized coordinates, sigma=0.05 means 0.05 training standard
   deviations; this is an illustrative number, not the attack-spec setting.
4. Save clean and perturbed feature tables plus noise seed, config and hashes.
5. Later connect the perturbed table to the observed stream only, compare the
   same frozen policies on clean and noisy observations, and repeat over
   independently declared noise seeds. Training seeds and noise seeds are
   separate variables. Use the same noise realization across model seeds when
   making paired comparisons.

V9 `bundle.market_obs` contains date, observed_close, twelve standardized
context columns and rule/gate flags. Its eleven price-derived features are
listed in `victimagent/src/victimagent/v8/features.py` as `V8_MARKET_FEATURES`;
`cash_rate_known` is the twelfth context feature. Whether to perturb the rate
must come from the spec. Account and holding-rule state are composed online;
they cannot be prepared as a policy-independent static dataset.

When ready for real data, export the chosen fold's `bundle.market_obs` using
the existing V9 bundle factory, with the fold's training window as `fit` and
validation window as `select`. Preserve the matching scaler fingerprint and
source hashes alongside that clean CSV. Do not refit on validation or noisy
data. No real-data exporter is run by this exercise. Put local CSVs in
`local_data/` (ignored by this lab's gitignore). Licensed data must not be
committed in any parent repository; this new directory is currently outside
the `victim_ppo` Git repository.

This scaffold preserves observed_close and gate flags. That is a deliberate
exercise boundary, pending the spec: a feature-only attack leaves the gate
unchanged unless the attack protocol explicitly says otherwise. Do not replace
the environment's true market table with this CSV. A future integration must
validate feature names, date alignment and the frozen scaler before replacing
only allowed columns in market_obs.

## Run and debug

Run these from `C:\03_College\Grad_proj\ppo_v9\attack_lab` using any working
Python 3.11+; only the standard library is required.

```powershell
python -m unittest discover -s . -p 'test_prepare.py' -v
python prepare.py --demo --mode identity --output outputs/demo_identity_01
```

Identity mode must produce identical clean/perturbed CSV hashes. Open the two
CSVs and `manifest.json`: preview and per-column delta statistics should all
show zero change. This runs before you implement noise.

Next implement the TODO function, then:

```powershell
python prepare.py --demo --mode attack --output outputs/demo_attack_01
```

The test `test_unfinished_attack_fails_explicitly` documents the starter state;
replace it with tests of your implementation after filling the function.
Useful checks: sigma=0 gives identity; same noise seed gives identical data;
different noise seeds usually give different data for nonzero sigma; protected
columns stay identical. Over many samples, independent zero-mean noise should
have empirical mean near 0 and standard deviation near sigma. Eight demo rows
are too few for a distribution check. Clipping changes the distribution.

For real data, create a local config from `config.example.json`, fill its TODOs,
set attacked column names and sigma, and start with only five rows:

```powershell
python prepare.py --input local_data/clean_fold1.csv --config config.local.json --mode attack --limit 5 --output outputs/real_debug_01
```

Remove `--limit` only after inspecting the preview. Debug outputs are explicitly
marked with their row limit. Every run needs a new output directory. Failed
attempts retain `FAILURE.txt`; use a new suffix after fixing the cause.

For an interactive debugger, insert `breakpoint()` inside `perturb_values()`
or launch:

```powershell
python -m pdb prepare.py --demo --mode attack --output outputs/debug_01
```

Pdb commands: `b perturb_values` to stop at your function, `c` to continue,
`p values`, `p config`, `n` to execute the next line, `s` to step inside a call,
`where` for the stack, and `q` to quit. Inspect your noise before adding it and
the result afterward. Do not reseed per row or draw extra random numbers while
debugging unless you restart the run afterward; doing so changes the noise.
