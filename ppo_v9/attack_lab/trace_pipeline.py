"""Audit CSV -> policy tensor -> probabilities -> gate/rule -> trades -> NAV."""
from dataclasses import replace
import json
import math
from pathlib import Path
import traceback
import numpy as np
import pandas as pd
import torch
import evaluate_v9 as base
from victimagent.v4.evaluation import metrics

HERE = Path(__file__).resolve().parent
OUT = HERE/'outputs'/'v9_pipeline_trace_20261020_02'
SOURCE = HERE/'outputs'/'v9_gaussian_10noise_20261019_01'
OLD = HERE/'outputs'/'v9_gaussian_20261019_02'


def probabilities(model, obs):
    tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        return model.policy.get_distribution(tensor).distribution.probs.cpu().numpy()[0]


def trace(model, bundle, clean_frame, spy, seed, folder, reference):
    env = base.runner.v9._environment(bundle, spy, 'v9_gate_relative_reward', training=False)
    before = (base.runner.parameter_sha256(model), base.optimizer_sha256(model),
              base.scaler_sha256(bundle.observation_builder))
    observation, reset = env.reset(seed=seed)
    initial_nav = float(reset['initial_close_nav'])
    records, observations, counterfactuals = [], [], []
    reward_sum = 0.
    try:
        while True:
            index = env.unwrapped._decision_index
            context_names = bundle.observation_builder.context_features
            expected = bundle.market_obs.loc[index, list(context_names)].to_numpy(dtype=np.float32)
            if not np.array_equal(observation[:12], expected):
                raise AssertionError('CSV features did not reach the actual policy input')
            clean_context = clean_frame.loc[index, list(context_names)].to_numpy(dtype=np.float32)
            if observation[11] != clean_context[11]:
                raise AssertionError('Interest rate changed')
            # Same-state counterfactual isolates feature noise from account/rule
            # divergence: only the eleven market features are replaced by clean.
            cf = observation.copy()
            cf[:11] = clean_context[:11]
            if not np.array_equal(cf[11:], observation[11:]):
                raise AssertionError('Counterfactual changed account or rule state')
            p, q = probabilities(model, observation), probabilities(model, cf)
            action = int(np.argmax(p))
            predicted, _ = model.predict(observation, deterministic=True)
            if action != int(np.asarray(predicted).item()):
                raise AssertionError('Inspected probabilities disagree with deterministic policy')
            cf_action = int(np.argmax(q))
            gate = bool(bundle.market_obs.loc[index, 'trend_above_sma_200'])
            rule = env.env
            lock = rule.sessions_since_change is not None and rule.sessions_since_change <= 5
            obs_next, reward, terminated, truncated, info = env.step(action)
            gated = 2 if gate else action
            cf_gated = 2 if gate else cf_action
            expected_executed = None if lock else gated
            if info['gated_action'] != gated or info['executed_action'] != expected_executed:
                raise AssertionError('Gate/rule execution differs from traced logic')
            reward_sum += float(reward)
            record = {'decision_date': str(info['decision_date'].date()),
                'execution_date': str(info['execution_date'].date()),
                'action': action, 'counterfactual_clean_feature_action': cf_action,
                'market_input_changed': bool(np.any(observation[:11] != cf[:11])),
                'max_abs_input_delta': float(np.max(np.abs(observation[:11]-cf[:11]))),
                'max_abs_probability_delta': float(np.max(np.abs(p-q))),
                'gate_forced_full': gate, 'gated_action': gated, 'cf_gated_action': cf_gated,
                'five_day_lock_active': bool(lock), 'executed_action': info['executed_action'],
                'order_placed': info['order_placed'], 'order_rejection_reason': info['order_rejection_reason'],
                'trade_notional': float(info['trade']['trade_notional']),
                'transaction_cost': float(info['transaction_cost']),
                'end_nav': float(info['end_close_nav']), 'reward': float(reward),
                'true_execution_open': float(info['true_execution_open']),
                'true_valuation_close': float(info['true_valuation_close'])}
            for i in range(3):
                record[f'prob_{i}'] = float(p[i])
                record[f'cf_prob_{i}'] = float(q[i])
            records.append(record)
            observations.append(observation.copy())
            counterfactuals.append(cf)
            observation = obs_next
            if terminated or truncated:
                if not terminated or truncated:
                    raise AssertionError('Evaluation did not terminate naturally')
                break
        after = (base.runner.parameter_sha256(model), base.optimizer_sha256(model),
                 base.scaler_sha256(bundle.observation_builder))
        nav_error = reward_sum-math.log(records[-1]['end_nav']/initial_nav)
        if before != after or abs(nav_error)>1e-10:
            raise AssertionError('Evaluation state or reward invariant failed')
        frame = pd.DataFrame(records)
        expected_trace = pd.read_csv(reference, float_precision='round_trip')
        if list(frame['decision_date']) != list(expected_trace['decision_date']):
            raise AssertionError('Trace dates differ from original evaluation')
        for column in ['action','gated_action','trade_notional','end_nav','true_execution_open','true_valuation_close']:
            if not np.array_equal(frame[column].to_numpy(), expected_trace[column].to_numpy()):
                raise AssertionError(f'Trace failed to reproduce saved evaluation: {column}')
        if not np.array_equal(frame['executed_action'].to_numpy(), expected_trace['executed_action'].to_numpy(), equal_nan=True):
            raise AssertionError('Executed targets differ from saved evaluation')
        base.save_csv(folder/'daily_trace.csv', frame)
        with (folder/'policy_inputs.npz').open('xb') as handle:
            np.savez_compressed(handle, actual=np.asarray(observations),
                                same_state_clean_features=np.asarray(counterfactuals))
        flip = frame['action'] != frame['counterfactual_clean_feature_action']
        gate_masked = flip & (frame['gated_action']==frame['cf_gated_action'])
        after_gate = frame['gated_action']!=frame['cf_gated_action']
        return frame, {'steps':len(frame), 'input_changed_steps':int(frame['market_input_changed'].sum()),
            'probability_changed_steps':int((frame['max_abs_probability_delta']>0).sum()),
            'probability_changed_over_1e_7_steps':int((frame['max_abs_probability_delta']>1e-7).sum()),
            'same_state_action_flips':int(flip.sum()), 'flips_masked_by_gate':int(gate_masked.sum()),
            'flips_masked_by_five_day_lock':int((after_gate & frame['five_day_lock_active']).sum()),
            'flips_reaching_executed_target':int((after_gate & ~frame['five_day_lock_active']).sum()),
            'reward_nav_error':nav_error, 'policy_input_verified':True,
            'saved_evaluation_reproduced':True, 'model_optimizer_scaler_unchanged':True,
            'trace_sha256':base.prepare.digest(folder/'daily_trace.csv'),
            'policy_inputs_sha256':base.prepare.digest(folder/'policy_inputs.npz')}
    finally:
        env.close()


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    report=json.loads((SOURCE/'COMPARISON.json').read_text(encoding='utf-8'))
    config=base.load_config(base.ROOT/'configs/ppo_v9_reward_ablation.yaml')
    preflight=base.runner.preflight(base.ROOT,config,require_data=True,require_clean=True)
    base.runner.write_new(OUT/'PROTOCOL.json',{'scope':'relative reward; folds 2/3; five model seeds x ten noise seeds',
        'same_state_counterfactual':'replace only first 11 features, preserving actual account and rule state',
        'preflight':preflight,'source_report_sha256':base.prepare.digest(SOURCE/'COMPARISON.json'),
        'script_sha256':base.prepare.digest(__file__),'no_retraining':True,'known_period_accessed':False})
    base.runner.v8._set_threads(1)
    market,events=base.runner.v8._v8_inputs(base.ROOT/'victimagent',base.ROOT/'victimagent')
    summaries=[]
    folds=base.runner.load_rolling_folds(base.ROOT/'victimagent/configs/v8_folds.yaml').folds
    for fold in folds:
        if fold.fold_id not in ('fold_2','fold_3'):
            continue
        clean,spy=base.runner.v9.v9_bundle(market,events,fit=fold.train,select=fold.validation)
        source=clean.market_obs.copy(deep=True)
        for model_seed in range(5):
            label=f'ppo_v9_gate_relative_{fold.fold_id}_seed{model_seed}'
            record=json.loads((base.ROOT/base.runner.RECORDS/f'{label}_v9spec1.json').read_text(encoding='utf-8'))
            checkpoint=base.ROOT/record['checkpoint_path']
            if base.prepare.digest(checkpoint)!=record['checkpoint_sha256']:
                raise AssertionError('Checkpoint changed')
            model=base.PPO.load(checkpoint,device='cpu')
            model.policy.set_training_mode(False)
            for noise_seed in range(123,133):
                cell=next(c for c in report['cells'] if c['arm']=='v9_gate_relative_reward' and c['fold']==fold.fold_id and c['seed']==model_seed and c['noise_seed']==noise_seed)
                data_root=OLD if noise_seed==123 else SOURCE
                data_folder=data_root/fold.fold_id if noise_seed==123 else data_root/fold.fold_id/f'noise{noise_seed}'
                data_path=data_folder/'perturbed_observations.csv'
                dataset_meta=json.loads((data_folder/'DATASET_MANIFEST.json').read_text(encoding='utf-8'))
                expected_hash=dataset_meta.get('sha256',dataset_meta.get('perturbed_sha256'))
                if base.prepare.digest(data_path)!=expected_hash:
                    raise AssertionError('Noisy dataset hash changed')
                obs=pd.read_csv(data_path,float_precision='round_trip',parse_dates=['date'])
                attacked=replace(clean,market_obs=obs)
                target=OUT/fold.fold_id/f'model{model_seed}'/f'noise{noise_seed}'
                target.mkdir(parents=True)
                reference=data_root/cell['perturbed']['trajectory_path']
                frame,summary=trace(model,attacked,source,spy,model_seed,target,reference)
                clean_ref=OLD/cell['clean']['trajectory_path'] if noise_seed==123 else SOURCE/cell['clean']['trajectory_path']
                clean_trace=pd.read_csv(clean_ref,float_precision='round_trip')
                summary.update(fold=fold.fold_id,model_seed=model_seed,noise_seed=noise_seed,
                    actual_action_changes_vs_clean=int((frame['action'].to_numpy()!=clean_trace['action'].to_numpy()).sum()),
                    actual_gated_action_changes_vs_clean=int((frame['gated_action'].to_numpy()!=clean_trace['gated_action'].to_numpy()).sum()),
                    actual_trade_notional_changes_vs_clean=int((frame['trade_notional'].to_numpy()!=clean_trace['trade_notional'].to_numpy()).sum()),
                    actual_nav_changes_vs_clean=int((frame['end_nav'].to_numpy()!=clean_trace['end_nav'].to_numpy()).sum()),
                    final_wealth_change=cell['wealth_change'])
                base.runner.write_new(target/'SUMMARY.json',summary)
                summaries.append(summary)
            print(f'{fold.fold_id} model {model_seed}: all ten noise seeds traced; {len(summaries)}/100',flush=True)
    grouped=[]
    keys=['steps','input_changed_steps','probability_changed_steps','same_state_action_flips',
          'flips_masked_by_gate','flips_masked_by_five_day_lock','flips_reaching_executed_target',
          'actual_action_changes_vs_clean','actual_gated_action_changes_vs_clean','actual_trade_notional_changes_vs_clean']
    for fold in ['fold_2','fold_3']:
        for seed in range(5):
            rows=[s for s in summaries if s['fold']==fold and s['model_seed']==seed]
            grouped.append({'fold':fold,'model_seed':seed,**{key:sum(s[key] for s in rows) for key in keys},
                'unchanged_final_wealth_runs':sum(s['final_wealth_change']==0 for s in rows)})
    base.runner.write_new(OUT/'TRACE_REPORT.json',{'status':'COMPLETE','grouped':grouped,'cells':summaries})
    lines=['# Policy-input and execution pipeline audit','',
        '100 attacked rollouts: relative-reward models, folds 2/3, model seeds 0–4, noise seeds 123–132.',
        'Every policy input is checked against the saved CSV at float32 precision; every probability argmax against model.predict.',
        'All actual traces must reproduce saved action, gate, executed target, true price, trade notional and NAV paths exactly.',
        'Same-state counterfactual replaces only market features, preserving the attacked rollout account and rule states. It is a one-step diagnostic, not a second simulated portfolio.', '',
        '| Fold | Model seed | Steps | Changed inputs | Changed probabilities | Feature-induced action flips | Gate masked | Lock masked | Reach target | Actual trade changes vs clean | Same final wealth /10 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g in grouped:
        lines.append(f"| {g['fold']} | {g['model_seed']} | {g['steps']} | {g['input_changed_steps']} | {g['probability_changed_steps']} | {g['same_state_action_flips']} | {g['flips_masked_by_gate']} | {g['flips_masked_by_five_day_lock']} | {g['flips_reaching_executed_target']} | {g['actual_trade_notional_changes_vs_clean']} | {g['unchanged_final_wealth_runs']}/10 |")
    lines+=['','Counts sum across ten noise seeds; repeated dates are not independent market samples.',
        'Gate/lock columns classify same-state action flips. Actual trade changes compare full attacked and clean rollouts and can include later account feedback.',
        'Daily CSVs include policy probabilities, actions, gate, lock, rejection reason, trades and NAV. NPZs contain actual 20-D inputs and their same-state clean-feature counterfactuals.']
    with (OUT/'TRACE_REPORT.md').open('x',encoding='utf-8') as handle:
        handle.write('\n'.join(lines)+'\n')


if __name__=='__main__':
    try:
        main()
    except BaseException:
        if OUT.exists():
            with (OUT/'FAILURE.txt').open('x',encoding='utf-8') as handle:
                traceback.print_exc(file=handle)
        raise
