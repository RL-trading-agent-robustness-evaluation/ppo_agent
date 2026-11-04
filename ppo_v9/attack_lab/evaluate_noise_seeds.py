"""Extend the frozen V9 diagnostic to ten noise realizations, with paired comparisons."""
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np
import pandas as pd
import evaluate_v9 as base

HERE = Path(__file__).resolve().parent
OUT = HERE/'outputs'/'v9_gaussian_10noise_20261019_01'
PREVIOUS = HERE/'outputs'/'v9_gaussian_20261019_02'
NOISE_SEEDS = tuple(range(123, 133))


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    base.OUT = OUT
    original = json.loads((PREVIOUS/'COMPARISON.json').read_text(encoding='utf-8'))
    config = base.load_config(base.ROOT/'configs/ppo_v9_reward_ablation.yaml')
    provenance = base.runner.preflight(base.ROOT, config, require_data=True, require_clean=True)
    protocol = original['protocol']
    if base.prepare.digest(HERE/'prepare.py') != protocol['perturbation_script_sha256']:
        raise ValueError('Perturbation code changed since seed 123')
    if provenance['ppo_config_sha256'] != protocol['preflight']['ppo_config_sha256']:
        raise ValueError('Model configuration changed')
    noise_config = protocol['noise_config']
    base.runner.write_new(OUT/'PROTOCOL.json', {
        'noise_seeds': NOISE_SEEDS, 'noise_config': noise_config, 'no_retraining': True,
        'preflight': provenance, 'previous_report_sha256': base.prepare.digest(PREVIOUS/'COMPARISON.json'),
        'reuse_seed123': True, 'script_sha256': base.prepare.digest(__file__),
        'known_period_accessed': False, 'formal_attack_spec_verified': False,
        'scope': '30 frozen checkpoints x 10 noise seeds; 123 retained, 124-132 newly evaluated'})
    base.runner.v8._set_threads(1)
    market, events = base.runner.v8._v8_inputs(base.ROOT/'victimagent', base.ROOT/'victimagent')
    folds = base.runner.load_rolling_folds(base.ROOT/'victimagent/configs/v8_folds.yaml').folds
    results = []
    for row in original['cells']:
        results.append({**row, 'noise_seed': 123, 'source': str(PREVIOUS/'COMPARISON.json')})
    for fold in folds:
        folder = OUT/fold.fold_id
        folder.mkdir()
        clean, spy = base.runner.v9.v9_bundle(market, events, fit=fold.train, select=fold.validation)
        source = clean.market_obs.copy(deep=True)
        if base.scaler_fingerprint(clean.observation_builder) != original['datasets'][fold.fold_id]['scaler']:
            raise ValueError('Scaler changed since first evaluation')
        base.save_csv(folder/'clean_observations.csv', source)
        if base.prepare.digest(folder/'clean_observations.csv') != original['datasets'][fold.fold_id]['clean_sha256']:
            raise ValueError('Clean observations changed')
        records = source.astype(str).to_dict('records')
        for i, row in enumerate(records):
            for column in noise_config['columns']:
                row[column] = repr(float(source.iloc[i][column]))
        variants = {}
        for noise_seed in NOISE_SEEDS[1:]:
            target = folder/f'noise{noise_seed}'
            target.mkdir()
            setting = {**noise_config, 'noise_seed': noise_seed}
            noisy_rows, stats = base.prepare.transform(records, setting, 'attack')
            replay, _ = base.prepare.transform(records, setting, 'attack')
            if noisy_rows != replay:
                raise AssertionError('Noise replay mismatch')
            noisy_obs = source.copy(deep=True)
            for column in setting['columns']:
                noisy_obs[column] = [float(row[column]) for row in noisy_rows]
            path = target/'perturbed_observations.csv'
            base.save_csv(path, noisy_obs)
            loaded = pd.read_csv(path, parse_dates=['date'], float_precision='round_trip')
            for column in source:
                expected = noisy_obs[column] if column in setting['columns'] else source[column]
                if not np.array_equal(expected.to_numpy(), loaded[column].to_numpy()):
                    raise AssertionError(f'Dataset serialization or protected-column mismatch: {column}')
            variants[noise_seed] = replace(clean, market_obs=loaded)
            base.runner.write_new(target/'DATASET_MANIFEST.json', {
                'noise_config': setting, 'statistics': stats, 'sha256': base.prepare.digest(path),
                'scaler': base.scaler_fingerprint(clean.observation_builder), 'metadata': clean.metadata})
        for row in original['cells']:
            if row['fold'] != fold.fold_id:
                continue
            ppo_arm = 'ppo_v9_gate_relative' if row['arm']=='v9_gate_relative_reward' else 'ppo_v9_gate_nav'
            label = f"{ppo_arm}_{fold.fold_id}_seed{row['seed']}"
            old = json.loads((base.ROOT/base.runner.RECORDS/f'{label}_v9spec1.json').read_text(encoding='utf-8'))
            checkpoint = base.ROOT/old['checkpoint_path']
            if base.prepare.digest(checkpoint) != row['checkpoint_sha256']:
                raise ValueError('Checkpoint changed')
            model = base.PPO.load(checkpoint, device='cpu')
            model.policy.set_training_mode(False)
            # One current clean replay per model verifies that reused baselines still match.
            normal = base.evaluate(model, clean, spy, row['arm'], row['seed'], folder/f'{label}_clean.csv')
            if normal['trajectory_sha256'] != row['clean']['trajectory_sha256']:
                raise AssertionError('Clean trajectory changed')
            for noise_seed, noisy in variants.items():
                attacked = base.evaluate(model, noisy, spy, row['arm'], row['seed'],
                                         folder/f'noise{noise_seed}'/f'{label}_noisy.csv')
                item = {**row, 'clean': normal, 'perturbed': attacked, 'noise_seed': noise_seed,
                        'perturbed_beats_bh': attacked['metrics']['final_wealth'] > row['buy_and_hold'],
                        'wealth_change': attacked['metrics']['final_wealth'] - normal['metrics']['final_wealth']}
                results.append(item)
                base.runner.write_new(folder/f'noise{noise_seed}'/f'{label}_comparison.json', item)
            if not clean.market_obs.equals(source):
                raise AssertionError('Clean source mutated')
            print(f'{label}: nine additional noise seeds complete; {len(results)}/300 total', flush=True)
    grouped, per_model = [], []
    for arm in ['v9_gate_relative_reward', 'v9_trend_gate']:
        for fold in folds:
            rows = [r for r in results if r['arm']==arm and r['fold']==fold.fold_id]
            if len(rows)!=50:
                raise ValueError('Incomplete arm/fold matrix')
            clean_rows = [r for r in rows if r['noise_seed']==123]
            wealth = [r['perturbed']['metrics']['final_wealth'] for r in rows]
            changes = [r['wealth_change']*100 for r in rows]
            per_noise_medians = [float(np.median([r['perturbed']['metrics']['final_wealth'] for r in rows if r['noise_seed']==s])) for s in NOISE_SEEDS]
            grouped.append({'arm': arm, 'fold': fold.fold_id, 'bh_return_pct': (rows[0]['buy_and_hold']-1)*100,
                'clean_median_return_pct': (float(np.median([r['clean']['metrics']['final_wealth'] for r in clean_rows]))-1)*100,
                'noisy_median_return_pct': (float(np.median(wealth))-1)*100,
                'noisy_wins': sum(r['perturbed_beats_bh'] for r in rows), 'evaluations': 50,
                'clean_wins': sum(r['clean_beats_bh'] for r in clean_rows),
                'median_paired_change_pp': float(np.median(changes)),
                'worst_paired_change_pp': min(changes), 'best_paired_change_pp': max(changes),
                'per_noise_median_return_range_pct': [(min(per_noise_medians)-1)*100,(max(per_noise_medians)-1)*100]})
            for seed in range(5):
                selected = [r for r in rows if r['seed']==seed]
                per_model.append({'arm': arm, 'fold': fold.fold_id, 'model_seed': seed,
                    'wins_across_noise_seeds': sum(r['perturbed_beats_bh'] for r in selected),
                    'clean_return_pct': (selected[0]['clean']['metrics']['final_wealth']-1)*100,
                    'noisy_return_min_pct': min((r['perturbed']['metrics']['final_wealth']-1)*100 for r in selected),
                    'noisy_return_max_pct': max((r['perturbed']['metrics']['final_wealth']-1)*100 for r in selected)})
    base.runner.write_new(OUT/'COMPARISON.json', {'status':'COMPLETE','noise_seeds':NOISE_SEEDS,
        'grouped':grouped,'per_model':per_model,'cells':results,
        'interpretation':'Noise repetitions share models and market periods; they are not independent training runs.'})
    lines = ['# V9 Gaussian noise: ten seeds', '',
        'Noise seeds 123–132; sigma=0.05 on 11 standardized price-derived features. Same frozen 30 checkpoints; no retraining.',
        'Clean gate, rates and execution market. All current clean replays match original trajectory hashes.',
        'Seed 123 is reused from the verified prior diagnostic; seeds 124–132 are new. Formal attack-spec alignment remains unverified.', '',
        '| Group | Fold | B&H return | Clean median return | Noisy median return | Clean wins | Noisy wins | Median paired change (pp) | Worst / best change (pp) |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for g in grouped:
        lines.append(f"| {g['arm']} | {g['fold']} | {g['bh_return_pct']:.2f}% | {g['clean_median_return_pct']:.2f}% | {g['noisy_median_return_pct']:.2f}% | {g['clean_wins']}/5 | {g['noisy_wins']}/50 | {g['median_paired_change_pp']:+.3f} | {g['worst_paired_change_pp']:+.3f} / {g['best_paired_change_pp']:+.3f} |")
    lines += ['', 'Noisy medians pool 5 models x 10 noise realizations. Win counts are repeated evaluations, not independent training-seed evidence.', '',
              '| Group | Fold | Model seed | Clean return | Noisy return range | Wins across noise seeds |',
              '|---|---|---:|---:|---:|---:|']
    for m in per_model:
        lines.append(f"| {m['arm']} | {m['fold']} | {m['model_seed']} | {m['clean_return_pct']:.3f}% | {m['noisy_return_min_pct']:.3f}% to {m['noisy_return_max_pct']:.3f}% | {m['wins_across_noise_seeds']}/10 |")
    with (OUT/'COMPARISON.md').open('x',encoding='utf-8') as handle:
        handle.write('\n'.join(lines)+'\n')


if __name__=='__main__':
    try:
        main()
    except BaseException:
        if OUT.exists():
            with (OUT/f'FAILURE_{os.getpid()}.txt').open('x',encoding='utf-8') as handle:
                traceback.print_exc(file=handle)
        raise
