"""Generate feature-noise datasets and evaluate every original V9 checkpoint."""
from dataclasses import replace
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / 'victim_ppo'
if not ROOT.exists():
    ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'victimagent/src')]
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import prepare
from victim_ppo.v9 import runner
from victim_ppo.v9.config import load_config
from victimagent.v8.features import V8_MARKET_FEATURES
from victimagent.evaluation import scaler_fingerprint, scaler_sha256, optimizer_sha256
from victimagent.v4.evaluation import rollout

OUT = HERE/'outputs'/'v9_gaussian_20261019_02'


def save_csv(path, frame):
    with path.open('x', newline='', encoding='utf-8') as handle:
        frame.to_csv(handle, index=False)


def evaluate(model, bundle, spy, arm, seed, path):
    env = runner.v9._environment(bundle, spy, arm, training=False)
    before = (runner.parameter_sha256(model), optimizer_sha256(model),
              scaler_sha256(bundle.observation_builder))
    def choose(observation, _index):
        action, _ = model.predict(observation, deterministic=True)
        return int(np.asarray(action).item())
    try:
        rows, metrics, error = rollout(env, choose, seed=seed)
        after = (runner.parameter_sha256(model), optimizer_sha256(model),
                 scaler_sha256(bundle.observation_builder))
        if before != after or abs(error) > 1e-10:
            raise AssertionError('Evaluation mutated model/optimizer/scaler or violated reward identity')
        trace = runner.v9._action_trace(rows, bundle, arm)
        save_csv(path, pd.DataFrame(rows))
        return {'metrics': metrics, 'reward_nav_error': error, 'technical_pass': True,
                'trajectory_sha256': prepare.digest(path),
                'trajectory_path': str(path.relative_to(OUT)),
                'action_trace': trace}
    finally:
        env.close()


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    config = load_config(ROOT/'configs/ppo_v9_reward_ablation.yaml')
    provenance = runner.preflight(ROOT, config, require_data=True, require_clean=True)
    noise_config = json.loads((HERE/'config.example.json').read_text(encoding='utf-8'))
    # Demo names have no real-data meaning: explicitly declare actual attack scope.
    noise_config['columns'] = list(V8_MARKET_FEATURES)
    protocol = {'created_at_utc': runner.now(), 'preflight': provenance,
                'noise_config': noise_config, 'original_config_sha256': prepare.digest(HERE/'config.example.json'),
                'script_sha256': prepare.digest(__file__), 'perturbation_script_sha256': prepare.digest(HERE/'prepare.py'),
                'scope': 'all 30 original V9 cells, seeds 0-4, both arms, three validation folds',
                'protected': 'true market, interest rate, account/rule states, observed_close, trend gate',
                'formal_attack_spec_verified': False, 'known_period_accessed': False,
                'noise_realizations': 1, 'no_retraining': True,
                'packages': {p: version(p) for p in ['stable-baselines3','torch','numpy','pandas','gymnasium']}}
    runner.write_new(OUT/'PROTOCOL.json', protocol)
    runner.v8._set_threads(1)
    market, events = runner.v8._v8_inputs(ROOT/'victimagent', ROOT/'victimagent')
    folds = runner.load_rolling_folds(ROOT/'victimagent/configs/v8_folds.yaml').folds
    comparisons, baselines, datasets = [], {}, {}
    for fold in folds:
        folder = OUT/fold.fold_id
        folder.mkdir()
        clean, spy = runner.v9.v9_bundle(market, events, fit=fold.train, select=fold.validation)
        fingerprint = scaler_fingerprint(clean.observation_builder)
        source = clean.market_obs.copy(deep=True)
        save_csv(folder/'clean_observations.csv', source)
        rows = source.astype(str).to_dict('records')
        for index, record in enumerate(rows):
            for column in noise_config['columns']:
                record[column] = repr(float(source.iloc[index][column]))
        noisy_rows, stats = prepare.transform(rows, noise_config, 'attack')
        repeat, _ = prepare.transform(rows, noise_config, 'attack')
        if repeat != noisy_rows:
            raise AssertionError('Noise seed does not reproduce the dataset')
        zero_config = {**noise_config, 'sigma': 0.0}
        zero_rows, _ = prepare.transform(rows, zero_config, 'attack')
        if zero_rows != rows:
            raise AssertionError('Zero-noise dataset is not identical')
        # Restore the exact binary clean values before replacing attacked columns.
        noisy_obs = source.copy(deep=True)
        for column in noise_config['columns']:
            noisy_obs[column] = [float(row[column]) for row in noisy_rows]
        save_csv(folder/'perturbed_observations.csv', noisy_obs)
        # Evaluation reads the saved dataset, rather than only an in-memory copy.
        loaded = pd.read_csv(folder/'perturbed_observations.csv', float_precision='round_trip', parse_dates=['date'])
        loaded['date'] = pd.to_datetime(loaded['date'])
        if not np.array_equal(loaded[noise_config['columns']].to_numpy(), noisy_obs[noise_config['columns']].to_numpy()):
            raise AssertionError('Serialized noisy features changed')
        protected = [c for c in source if c not in noise_config['columns']]
        for column in protected:
            if not np.array_equal(source[column].to_numpy(), loaded[column].to_numpy()):
                raise AssertionError(f'Protected column changed: {column}')
        noisy = replace(clean, market_obs=loaded)
        if noisy.market_true is not clean.market_true:
            raise AssertionError('True execution market replaced')
        baseline = runner.v9._baselines_with_rule(clean, spy)
        bh = baseline['buy_and_hold']['final_wealth']
        if not math.isclose(bh, runner.EXPECTED_BUY_AND_HOLD[fold.fold_id], abs_tol=1e-12, rel_tol=0):
            raise AssertionError('B&H mismatch')
        baselines[fold.fold_id] = baseline
        datasets[fold.fold_id] = {'metadata': clean.metadata, 'scaler': fingerprint,
            'clean_sha256': prepare.digest(folder/'clean_observations.csv'),
            'perturbed_sha256': prepare.digest(folder/'perturbed_observations.csv'),
            'noise_statistics': stats}
        runner.write_new(folder/'DATASET_MANIFEST.json', datasets[fold.fold_id])
        for ppo_arm in runner.ARMS:
            for seed in range(5):
                label = f'{ppo_arm}_{fold.fold_id}_seed{seed}'
                record_path = ROOT/runner.RECORDS/f'{label}_v9spec1.json'
                original = json.loads(record_path.read_text(encoding='utf-8'))
                checkpoint = ROOT/original['checkpoint_path']
                if (original['status'] != 'PASS' or prepare.digest(checkpoint) != original['checkpoint_sha256']
                        or original['scaler'] != fingerprint):
                    raise AssertionError(f'Invalid checkpoint/scaler: {label}')
                model = PPO.load(checkpoint, device='cpu')
                model.policy.set_training_mode(False)
                arm = config.upstream_arm(ppo_arm)
                normal = evaluate(model, clean, spy, arm, seed, folder/f'{label}_clean.csv')
                if (normal['trajectory_sha256'] != original['trajectory_sha256']
                        or normal['metrics']['final_wealth'] != original['final_wealth']):
                    raise AssertionError(f'Clean replay differs from original: {label}')
                attacked = evaluate(model, noisy, spy, arm, seed, folder/f'{label}_noisy.csv')
                if not clean.market_obs.equals(source):
                    raise AssertionError('Clean observations changed')
                entry = {'arm': arm, 'fold': fold.fold_id, 'seed': seed,
                    'checkpoint_sha256': prepare.digest(checkpoint), 'original_record_sha256': prepare.digest(record_path),
                    'clean': normal, 'perturbed': attacked, 'buy_and_hold': bh,
                    'clean_beats_bh': normal['metrics']['final_wealth'] > bh,
                    'perturbed_beats_bh': attacked['metrics']['final_wealth'] > bh,
                    'wealth_change': attacked['metrics']['final_wealth']-normal['metrics']['final_wealth']}
                comparisons.append(entry)
                runner.write_new(folder/f'{label}_comparison.json', entry)
                print(f"{label}: clean={normal['metrics']['final_wealth']:.8f}, noisy={attacked['metrics']['final_wealth']:.8f}, B&H={bh:.8f}", flush=True)
    grouped = []
    for arm in [config.upstream_arm(a) for a in runner.ARMS]:
        for fold in folds:
            cells = [r for r in comparisons if r['arm']==arm and r['fold']==fold.fold_id]
            grouped.append({'arm': arm, 'fold': fold.fold_id,
                'buy_and_hold_wealth': baselines[fold.fold_id]['buy_and_hold']['final_wealth'],
                'clean_median_wealth': float(np.median([r['clean']['metrics']['final_wealth'] for r in cells])),
                'perturbed_median_wealth': float(np.median([r['perturbed']['metrics']['final_wealth'] for r in cells])),
                'clean_wins': sum(r['clean_beats_bh'] for r in cells),
                'perturbed_wins': sum(r['perturbed_beats_bh'] for r in cells),
                'median_paired_return_change_pp': float(np.median([r['wealth_change']*100 for r in cells]))})
    runner.write_new(OUT/'COMPARISON.json', {'status': 'COMPLETE', 'protocol': protocol,
        'grouped': grouped, 'baselines': baselines, 'datasets': datasets, 'cells': comparisons})
    lines = ['# V9 Gaussian feature-noise diagnostic', '',
        f"Noise sigma={noise_config['sigma']}, noise seed={noise_config['noise_seed']}; 11 standardized price features.",
        'Interest rate, trend gate, execution prices, costs and ledger use the clean contract. No retraining.',
        'One noise realization; formal attack-spec alignment has not been verified. Development folds only; known period not accessed.', '',
        'All 30 clean trajectories exactly reproduce their original hashes and wealth. All noisy evaluations pass state and reward checks.', '',
        '| Arm | Fold | B&H return | Clean median return | Noisy median return | Clean wins | Noisy wins | Median paired change (pp) |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for g in grouped:
        lines.append(f"| {g['arm']} | {g['fold']} | {(g['buy_and_hold_wealth']-1)*100:.2f}% | {(g['clean_median_wealth']-1)*100:.2f}% | {(g['perturbed_median_wealth']-1)*100:.2f}% | {g['clean_wins']}/5 | {g['perturbed_wins']}/5 | {g['median_paired_return_change_pp']:+.3f} |")
    lines += ['', '## Every checkpoint', '', '| Arm | Fold | Seed | Clean return | Noisy return | B&H return | Change (pp) |', '|---|---|---:|---:|---:|---:|---:|']
    for r in comparisons:
        lines.append(f"| {r['arm']} | {r['fold']} | {r['seed']} | {(r['clean']['metrics']['final_wealth']-1)*100:.3f}% | {(r['perturbed']['metrics']['final_wealth']-1)*100:.3f}% | {(r['buy_and_hold']-1)*100:.3f}% | {r['wealth_change']*100:+.3f} |")
    lines += ['', 'Full CAGR, Sharpe, drawdown, exposure, turnover, costs and action distributions are in COMPARISON.json.',
              'Noise may improve or worsen returns. Random Gaussian noise is a robustness diagnostic, not a targeted optimization attack.',
              'The trend gate masks agent actions in uptrend sessions, so these results do not test gate or raw-price perturbation.']
    with (OUT/'COMPARISON.md').open('x', encoding='utf-8') as handle:
        handle.write('\n'.join(lines)+'\n')


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        if OUT.exists():
            with (OUT/'FAILURE.txt').open('x', encoding='utf-8') as handle:
                traceback.print_exc(file=handle)
        raise
