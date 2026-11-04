"""Small paired V9 RGN sweep: two strengths, one noise seed, 15 frozen models."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import traceback
import numpy as np
import pandas as pd
import evaluate_v9 as base

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE/'outputs/v9_gaussian_20261019_02'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'outputs/v9_small_sigma_20261029_01')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    try:
        base.OUT = out
        old = json.loads((PREVIOUS/'COMPARISON.json').read_text(encoding='utf-8'))
        setting = old['protocol']['noise_config']
        config = base.load_config(base.ROOT/'configs/ppo_v9_reward_ablation.yaml')
        preflight = base.runner.preflight(base.ROOT, config, require_data=True, require_clean=True)
        if base.prepare.digest(HERE/'prepare.py') != old['protocol']['perturbation_script_sha256']:
            raise AssertionError('Perturbation implementation changed since reference')
        base.runner.write_new(out/'PROTOCOL.json', {
            'algorithm':'V9 PPO', 'arm':'v9_gate_relative_reward', 'model_seeds':list(range(5)),
            'noise_seed':123, 'sigmas':[0.1,0.2], 'columns':setting['columns'],
            'representation':setting['representation'], 'folds':['fold_1','fold_2','fold_3'],
            'protected':'interest rate, gate flags, account/rule state, true market',
            'common_random_numbers':True, 'no_retraining':True, 'known_period_accessed':False,
            'formal_attack_spec_verified':False, 'preflight':preflight,
            'source_comparison_sha256':base.prepare.digest(PREVIOUS/'COMPARISON.json'),
            'script_sha256':base.prepare.digest(__file__)})
        base.runner.v8._set_threads(1)
        market, events = base.runner.v8._v8_inputs(base.ROOT/'victimagent',base.ROOT/'victimagent')
        results = []
        folds = base.runner.load_rolling_folds(base.ROOT/'victimagent/configs/v8_folds.yaml').folds
        for fold in folds:
            folder = out/fold.fold_id
            folder.mkdir()
            clean, spy = base.runner.v9.v9_bundle(market,events,fit=fold.train,select=fold.validation)
            source = clean.market_obs.copy(deep=True)
            base.save_csv(folder/'clean_observations.csv',source)
            if base.prepare.digest(folder/'clean_observations.csv')!=old['datasets'][fold.fold_id]['clean_sha256']:
                raise AssertionError('Clean data mismatch')
            records = source.astype(str).to_dict('records')
            for i,row in enumerate(records):
                for column in setting['columns']:
                    row[column]=repr(float(source.iloc[i][column]))
            variants, normalized_noise = {}, []
            for sigma in (0.1,0.2):
                target = folder/f'sigma_{sigma}'
                target.mkdir()
                noise_setting={**setting,'sigma':sigma,'noise_seed':123}
                perturbed, stats = base.prepare.transform(records,noise_setting,'attack')
                replay,_ = base.prepare.transform(records,noise_setting,'attack')
                if perturbed!=replay:
                    raise AssertionError('Noise replay mismatch')
                obs=source.copy(deep=True)
                for column in setting['columns']:
                    obs[column]=[float(row[column]) for row in perturbed]
                delta=(obs[setting['columns']].to_numpy()-source[setting['columns']].to_numpy())/sigma
                normalized_noise.append(delta)
                base.save_csv(target/'perturbed_observations.csv',obs)
                loaded=pd.read_csv(target/'perturbed_observations.csv',parse_dates=['date'],float_precision='round_trip')
                for column in source:
                    expected=obs[column] if column in setting['columns'] else source[column]
                    if not np.array_equal(loaded[column].to_numpy(),expected.to_numpy()):
                        raise AssertionError(f'CSV/protected column mismatch: {column}')
                variants[sigma]=replace(clean,market_obs=loaded)
                base.runner.write_new(target/'DATASET_MANIFEST.json',{
                    'noise_config':noise_setting,'statistics':stats,
                    'sha256':base.prepare.digest(target/'perturbed_observations.csv'),
                    'scaler':base.scaler_fingerprint(clean.observation_builder),'metadata':clean.metadata})
            np.testing.assert_allclose(normalized_noise[0],normalized_noise[1],rtol=0,atol=1e-12)
            for seed in range(5):
                reference=next(r for r in old['cells'] if r['arm']=='v9_gate_relative_reward' and r['fold']==fold.fold_id and r['seed']==seed)
                label=f'ppo_v9_gate_relative_{fold.fold_id}_seed{seed}'
                saved=json.loads((base.ROOT/base.runner.RECORDS/f'{label}_v9spec1.json').read_text(encoding='utf-8'))
                checkpoint=base.ROOT/saved['checkpoint_path']
                if base.prepare.digest(checkpoint)!=saved['checkpoint_sha256'] or saved['scaler']!=base.scaler_fingerprint(clean.observation_builder):
                    raise AssertionError('Checkpoint/scaler mismatch')
                model=base.PPO.load(checkpoint,device='cpu')
                model.policy.set_training_mode(False)
                normal=base.evaluate(model,clean,spy,saved['arm'],seed,folder/f'{label}_clean.csv')
                if normal['trajectory_sha256']!=reference['clean']['trajectory_sha256']:
                    raise AssertionError('Clean replay mismatch')
                for sigma,bundle in variants.items():
                    attacked=base.evaluate(model,bundle,spy,saved['arm'],seed,folder/f'sigma_{sigma}'/f'{label}_noisy.csv')
                    row={'fold':fold.fold_id,'model_seed':seed,'noise_seed':123,'sigma':sigma,
                         'buy_and_hold':reference['buy_and_hold'],'clean':normal,'noisy':attacked,
                         'sigma_005_reference':reference['perturbed']['metrics'],
                         'return_change_pp':100*(attacked['metrics']['final_wealth']-normal['metrics']['final_wealth']),
                         'checkpoint_sha256':saved['checkpoint_sha256']}
                    base.runner.write_new(folder/f'sigma_{sigma}'/f'{label}_comparison.json',row)
                    results.append(row)
                if not clean.market_obs.equals(source):
                    raise AssertionError('Clean observations mutated')
                print(f'{fold.fold_id} model {seed}: sigma 0.1 and 0.2 complete ({len(results)}/30)',flush=True)
        groups=[]
        for fold in folds:
            for sigma in (0.1,0.2):
                rows=[r for r in results if r['fold']==fold.fold_id and r['sigma']==sigma]
                groups.append({'fold':fold.fold_id,'sigma':sigma,'bh_return_pct':100*(rows[0]['buy_and_hold']-1),
                    'clean_median_return_pct':100*(float(np.median([r['clean']['metrics']['final_wealth'] for r in rows]))-1),
                    'noisy_median_return_pct':100*(float(np.median([r['noisy']['metrics']['final_wealth'] for r in rows]))-1),
                    'wins_vs_bh':sum(r['noisy']['metrics']['final_wealth']>r['buy_and_hold'] for r in rows),
                    'median_paired_change_pp':float(np.median([r['return_change_pp'] for r in rows])),
                    'return_change_range_pp':[min(r['return_change_pp'] for r in rows),max(r['return_change_pp'] for r in rows)]})
        base.runner.write_new(out/'COMPARISON.json',{'status':'COMPLETE','grouped':groups,'cells':results})
        lines=['# Small V9 Gaussian sigma sweep','',
            'Relative-reward V9 only; five frozen model seeds per fold, noise seed 123. Same Gaussian directions across strengths.',
            'Noise on eleven standardized market features. Clean gate/rate/ledger; no retraining; no known-period access.',
            'All clean trajectories reproduce original hashes; noisy evaluations preserve model/optimizer/scaler and reward invariants.',
            'This is a one-noise-seed diagnostic, not a robustness conclusion or verified formal attack-spec evaluation.','',
            '| Fold | Sigma | B&H return | Clean median | Noisy median | Wins vs B&H | Median paired change (pp) |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for g in groups:
            lines.append(f"| {g['fold']} | {g['sigma']} | {g['bh_return_pct']:.2f}% | {g['clean_median_return_pct']:.2f}% | {g['noisy_median_return_pct']:.2f}% | {g['wins_vs_bh']}/5 | {g['median_paired_change_pp']:+.3f} |")
        lines+=['','| Fold | Model seed | Sigma | Clean return | Noisy return | Change (pp) |','|---|---:|---:|---:|---:|---:|']
        for r in results:
            lines.append(f"| {r['fold']} | {r['model_seed']} | {r['sigma']} | {100*(r['clean']['metrics']['final_wealth']-1):.3f}% | {100*(r['noisy']['metrics']['final_wealth']-1):.3f}% | {r['return_change_pp']:+.3f} |")
        with (out/'COMPARISON.md').open('x',encoding='utf-8') as handle:
            handle.write('\n'.join(lines)+'\n')
    except BaseException:
        with (out/'FAILURE.txt').open('x',encoding='utf-8') as handle:
            traceback.print_exc(file=handle)
        raise


if __name__=='__main__':
    main()
