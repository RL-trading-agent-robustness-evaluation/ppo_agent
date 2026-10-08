"""Chronological training; updates never replace or reset the scoring ledger."""
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
import time

import numpy as np
import pandas as pd
from victimagent.evaluation import scaler_sha256, optimizer_sha256
from victimagent.v2.registry import sha256_file
from victimagent.v4.evaluation import rollout, metrics
from victimagent.v9.study import _baselines_with_rule, _action_trace
from victim_ppo.v9.runner import build_model, parameter_sha256, responsiveness
from .config import ARMS, FOLDS
from .data import MarketData
from .environment import RecentWindowEnv, trading_environment
from .policy import schedule_at

def write_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, default=str, allow_nan=False)

def preflight(root, config):
    hashes = {key: sha256_file(root/config.raw[key]) for key in ('market','events','vix')}
    for key in ('market','events'):
        if hashes[key] != config.raw[key+'_sha256']:
            raise ValueError(f'{key} parent processed-data hash mismatch')
    if version('stable-baselines3') != '2.9.0':
        raise ValueError('SB3 2.9.0 required')
    data = MarketData(root, config)
    # All development values must satisfy the documented missing-data rule.
    for index in range(251, len(data.sessions)):
        schedule_at(data.vix, index, 251)
    source_manifest = root/'SOURCE_MANIFEST.json'
    if not source_manifest.exists():
        raise FileNotFoundError('SOURCE_MANIFEST.json required')
    source = json.loads(source_manifest.read_text(encoding='utf-8'))
    if source['victimagent_actual_commit'] != config.raw['parent_commit']:
        raise ValueError('copied victimagent commit differs from configured parent')
    for rel, expected in source['copied_files'].items():
        if sha256_file(root/rel) != expected:
            raise ValueError(f'copied parent file changed: {rel}')
    implementation_hashes={p.relative_to(root).as_posix():sha256_file(p)
        for base in (root/'src/victim_ppo/v11',root/'scripts')
        for p in sorted(base.glob('*.py'))}
    estimates={}
    for fold,(_,_,end) in FOLDS.items():
        if fold=='known':
            continue
        final=int(data.sessions.searchsorted(pd.Timestamp(end),side='right'))-1
        estimates[fold]={}
        for fixed in (False,True):
            last=251
            count=0
            for index in range(252,final):
                if schedule_at(data.vix,index,last,fixed=fixed)['due']:
                    last=index
                    count+=1
            estimates[fold]['fixed' if fixed else 'vix']=dict(
                fine_tunes=count,transitions=200704+32768*count)
    total=5*sum(2*entry['vix']['transitions']+entry['fixed']['transitions'] for entry in estimates.values())
    return dict(status='PASS', data_hashes=hashes, implementation_hashes=implementation_hashes,
                config_sha256=sha256_file(config.path),
                spec_sha256=sha256_file(root/'docs/PPO_V11_ADAPTIVE_POLICY_REWARD_SPEC_ZH_HANT.md'),
                prompt_sha256=sha256_file(root/'docs/prompts/PPO_V11_IMPLEMENTATION_PROMPT_ZH_HANT.md'),
                parent_commit=config.raw['parent_commit'], source_manifest_sha256=sha256_file(source_manifest),
                initial_requested=200000, initial_expected=200704,
                fine_tune_transitions=32768, scaler_sha256=data.scaler_hash,
                initialization_date=str(data.fit.end.date()),
                packages={p:version(p) for p in ('stable-baselines3','torch','gymnasium','numpy','pandas')},
                known_period_accessed=False,
                budget_estimate=dict(trajectories=45, initial_transitions=45*200704,
                    per_fold_schedule=estimates,total_expected_transitions=total,
                    note='Calendar-derived exact update counts on current development data; CPU time measured by smoke.'))

class Trainer:
    def __init__(self, data, config, directory, arm, seed, smoke=False):
        self.data, self.config, self.directory = data, config, directory
        self.arm, self.seed, self.smoke = arm, seed, smoke
        self.model = None
        self.updates = []
        self.last_update = 251

    def train(self, index, reason, initial=False):
        import torch
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.logger import configure
        from stable_baselines3.common.utils import get_schedule_fn

        cutoff = self.data.sessions[index]
        bundle, spy = self.data.bundle(cutoff)
        ordinal = len(self.updates)
        # SeedSequence avoids coupling sampler state to action RNG consumption.
        sample_seed = int(np.random.SeedSequence([self.seed, ordinal, 110]).generate_state(1)[0])
        env = RecentWindowEnv(bundle, spy, self.arm.endswith('_mix'), sample_seed)
        stage = self.directory/f'update_{ordinal:04d}'
        stage.mkdir(exist_ok=False)
        training_path = stage/'training_steps.jsonl'
        requested = (20000 if self.smoke else 200000) if initial else 32768
        expected = math.ceil(requested/2048)*2048
        before_params = parameter_sha256(self.model) if self.model else None
        before_optimizer = optimizer_sha256(self.model) if self.model else None
        if self.model is None:
            self.model = build_model(env, self.config, seed=self.seed,
                                     tensorboard_log=stage/'tensorboard')
            before_params = parameter_sha256(self.model)
        else:
            self.model.set_env(env)
            self.model.learning_rate = 3e-5
            self.model.lr_schedule = get_schedule_fn(3e-5)
            if optimizer_sha256(self.model) != before_optimizer:
                raise RuntimeError('set_env discarded optimizer state')
        self.model.set_logger(configure(str(stage), ['csv','tensorboard']))
        total_before = self.model.num_timesteps
        scaler_before = scaler_sha256(bundle.observation_builder)
        diagnostics = Counter()
        components = Counter()
        streaks = Counter()
        observations = []

        class Callback(BaseCallback):
            def _on_step(callback):
                for info in callback.locals['infos']:
                    record = {key: info[key] for key in (
                        'decision_date','valuation_date','nav_log_reward','spy_log_return',
                        'relative_log_reward','trend_component_reward','positive_return_streak',
                        'positive_streak_bonus','weighted_streak_bonus','training_reward',
                        'agent_action','gate_forced_full','gated_action','executed_action')}
                    handle.write(json.dumps(record, default=str, allow_nan=False)+'\n')
                    diagnostics['steps'] += 1
                    diagnostics['bonus_steps'] += int(info['positive_streak_bonus'] > 0)
                    diagnostics['nav_abs'] += abs(info['nav_log_reward'])
                    streaks[str(info['positive_return_streak'])] += 1
                    for key in ('nav_log_reward','relative_log_reward','trend_component_reward',
                                'positive_streak_bonus','weighted_streak_bonus'):
                        components[key] += info[key]
                        components[key+'_absolute'] += abs(info[key])
                if not np.isfinite(callback.locals['new_obs']).all():
                    raise RuntimeError('nonfinite training observation')
                if len(observations) < 128:
                    observations.extend(np.asarray(callback.locals['new_obs']).copy())
                return True

        started = time.perf_counter()
        with training_path.open('x', encoding='utf-8') as handle:
            self.model.learn(total_timesteps=requested, reset_num_timesteps=initial,
                             callback=Callback(), progress_bar=False)
        realized = self.model.num_timesteps-total_before
        if realized != expected or diagnostics['steps'] != realized:
            raise RuntimeError('unexpected rollout transition count')
        if not all(torch.isfinite(t).all().item() for t in self.model.policy.state_dict().values()):
            raise RuntimeError('nonfinite policy parameters')
        logged = self.model.logger.name_to_value
        losses = {key:float(value) for key,value in logged.items()
                  if key.startswith('train/') and 'loss' in key}
        if not losses or not np.isfinite(list(losses.values())).all():
            raise RuntimeError('no finite PPO training loss')
        if scaler_sha256(bundle.observation_builder) != scaler_before:
            raise RuntimeError('scaler changed during training')
        if parameter_sha256(self.model) == before_params:
            raise RuntimeError('policy failed to update')
        checkpoint = stage/'model.zip'
        self.model.save(checkpoint)
        from stable_baselines3 import PPO
        loaded = PPO.load(checkpoint, device='cpu')
        if parameter_sha256(loaded) != parameter_sha256(self.model) or optimizer_sha256(loaded) != optimizer_sha256(self.model):
            raise RuntimeError('checkpoint failed parameter/optimizer roundtrip')
        probabilities = pd.DataFrame(dict(start=env.starts, endpoint_age=env.ages,
                                          probability=env.probabilities))
        probabilities.to_csv(stage/'windows.csv', index=False)
        payload = dict(ordinal=ordinal, trigger_date=str(cutoff.date()), cutoff=str(cutoff.date()),
                       activation_date=str(self.data.sessions[index+1].date()) if index+1<len(self.data.sessions) else None,
                       first_affected_fill=str(self.data.sessions[index+2].date()) if index+2<len(self.data.sessions) else None,
                       previous_update_date=str(self.data.sessions[self.last_update].date()), reason=reason,
                       requested_transitions=requested, realized_transitions=realized,
                       parameter_before=before_params, parameter_after=parameter_sha256(self.model),
                       optimizer_before=before_optimizer, optimizer_after=optimizer_sha256(self.model),
                       scaler_sha256=scaler_before, checkpoint_sha256=sha256_file(checkpoint),
                       checkpoint=str(checkpoint.relative_to(self.directory)),
                       windows_sha256=sha256_file(stage/'windows.csv'),
                       training_steps_sha256=sha256_file(training_path),
                       sample_seed=sample_seed, sampler_state=env.sampler.bit_generator.state,
                       sampled_endpoint_ages=env.sampled_ages,
                       sampled_transition_date_counts=dict(Counter(env.transition_dates)),
                       seconds=time.perf_counter()-started, learning_rate=float(self.model.lr_schedule(0)),
                       losses=losses, diagnostics=dict(diagnostics), components=dict(components),
                       streak_distribution=dict(streaks), responsiveness=responsiveness(self.model,observations))
        write_new(stage/'record.json', payload)
        self.updates.append(payload)
        self.last_update = index
        return payload

def scoring_row(info, reward, terminated, truncated, action):
    row = dict(info)
    for key in ('decision_date','execution_date','valuation_date','reward_available_date','next_observation_date'):
        row[key] = str(pd.Timestamp(row[key]).date())
    row.update(start_nav=info['start_close_nav'], end_nav=info['end_close_nav'],
               trade_notional=info['trade']['trade_notional'], action=int(action),
               reward=float(reward), terminated=bool(terminated), truncated=bool(truncated))
    return row

def decide_then_update(trainer, observation, index, reason, env, *, allow_update=True):
    """Lock the old model's action before learning; do not touch the account."""
    action=int(trainer.model.predict(observation,deterministic=True)[0])
    changed=False
    if allow_update and reason['due']:
        ledger_before=pickle.dumps(env)
        trainer.train(index,reason)
        if pickle.dumps(env)!=ledger_before:
            raise RuntimeError('training mutated scoring environment')
        changed=True
    return action,changed

def save_state(directory, trainer, phase, cursor, env=None, obs=None, rows=None):
    """Trusted local pickle stores ledger/wrappers, RNG, clock and model together.

    Saved only after a completed transition/update, never halfway through learn.
    No pending action exists at these boundaries. Do not load external pickles.
    """
    import torch
    path = directory/f'resume_{phase}_{cursor:05d}'
    path.mkdir(exist_ok=False)
    trainer.model.save(path/'model.zip')
    state = dict(phase=phase, cursor=cursor, last_update=trainer.last_update,
                 updates=trainer.updates, env=env, obs=obs, rows=rows or [],
                 random_state=random.getstate(), numpy_state=np.random.get_state(),
                 torch_state=torch.get_rng_state(), scaler_hash=trainer.data.scaler_hash,
                 parameter_hash=parameter_sha256(trainer.model),
                 optimizer_hash=optimizer_sha256(trainer.model))
    with (path/'state.pkl').open('xb') as handle:
        pickle.dump(state,handle)
    write_new(path/'hashes.json', dict(model=sha256_file(path/'model.zip'),
                                     state=sha256_file(path/'state.pkl')))

def restore_state(path, trainer):
    import torch
    from stable_baselines3 import PPO
    hashes = json.loads((path/'hashes.json').read_text())
    for key,file in (('model','model.zip'),('state','state.pkl')):
        if sha256_file(path/file) != hashes[key]:
            raise RuntimeError('resume file hash mismatch')
    with (path/'state.pkl').open('rb') as handle:
        state=pickle.load(handle)
    trainer.model=PPO.load(path/'model.zip',device='cpu')
    if (state['scaler_hash']!=trainer.data.scaler_hash or
        state['parameter_hash']!=parameter_sha256(trainer.model) or
        state['optimizer_hash']!=optimizer_sha256(trainer.model)):
        raise RuntimeError('resume model/scaler/optimizer mismatch')
    trainer.last_update,trainer.updates=state['last_update'],state['updates']
    random.setstate(state['random_state'])
    np.random.set_state(state['numpy_state'])
    torch.set_rng_state(state['torch_state'])
    return state

def run_cell(root, config, arm, fold, seed, tag, smoke=False, resume=None):
    import torch
    torch.set_num_threads(1)
    if arm not in ARMS or fold not in FOLDS or seed not in range(5):
        raise ValueError('invalid arm/fold/seed')
    if not tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in tag):
        raise ValueError('tag must be a nonempty safe identifier')
    label=f'{arm}_{fold}_seed{seed}_{tag}'
    namespace='smoke' if smoke else ('known' if fold=='known' else 'formal')
    directory=root/'artifacts/ppo_v11'/namespace/label
    record_path=root/'experiments/v11_ppo/records'/f'{namespace}_{label}.json'
    attempt=root/'experiments/v11_ppo/attempts'/f'{namespace}_{label}_{time.time_ns()}.json'
    if record_path.exists():
        raise FileExistsError('completed trajectory exists')
    audit=preflight(root,config)
    if not smoke:
        # Formal runs require evidence from all three smoke arms, not merely
        # existence of a config or an unverified command in the README.
        smoke_records=[]
        for path in (root/'experiments/v11_ppo/records').glob('smoke_*.json'):
            record=json.loads(path.read_text())
            if record.get('technical_pass') and record.get('config_version')==config.raw['version']:
                smoke_records.append(record)
        if set(r['arm'] for r in smoke_records)!=set(ARMS):
            raise RuntimeError('all three V11 smoke arms must pass before formal/known runs')
    if resume is None:
        directory.mkdir(parents=True,exist_ok=False)
        write_new(directory/'identity.json',dict(arm=arm,fold=fold,seed=seed,smoke=smoke,
                                               audit=audit,config_version=config.raw['version']))
    else:
        resume=Path(resume).resolve()
        if resume.parent != directory.resolve():
            raise ValueError('resume must belong to the same run directory')
        identity=json.loads((directory/'identity.json').read_text())
        if any(identity[k]!=v for k,v in dict(arm=arm,fold=fold,seed=seed,smoke=smoke).items()) or identity['audit']!=audit:
            raise ValueError('resume identity/provenance mismatch')
    train_end, val_start, val_end=FOLDS[fold]
    data=MarketData(root,config,end=val_end)
    trainer=Trainer(data,config,directory,arm,seed,smoke)
    started=time.perf_counter()
    try:
        state=restore_state(resume,trainer) if resume else None
        if state is None:
            trainer.train(251,dict(reason='initialization'),initial=True)
            save_state(directory,trainer,'training',252)
        boundary=int(data.sessions.searchsorted(pd.Timestamp(train_end),side='right'))
        if smoke:
            boundary=253  # exactly one forced update, then short score-only smoke
        cursor=state['cursor'] if state and state['phase']=='training' else 252
        if not state or state['phase']=='training':
            for index in range(cursor,boundary):
                reason=schedule_at(data.vix,index,trainer.last_update,
                                   fixed=arm.endswith('fixed_nav'),smoke=smoke)
                if reason['due']:
                    trainer.train(index,reason)
                    save_state(directory,trainer,'training',index+1)
        score_start=data.sessions[253] if smoke else pd.Timestamp(val_start)
        score_end=data.sessions[min(273,len(data.sessions)-1)] if smoke else pd.Timestamp(val_end)
        bundle,spy=data.bundle(score_end,start=score_start)
        if not smoke and fold=='known' and not state:
            trainer.model.save(directory/'pre_2022_model.zip')
            write_new(directory/'pre_2022_freeze.json',dict(
                training_boundary=train_end,scoring_start=val_start,
                parameter_sha256=parameter_sha256(trainer.model),
                optimizer_sha256=optimizer_sha256(trainer.model),
                scaler_sha256=data.scaler_hash,
                last_update_date=str(data.sessions[trainer.last_update].date()),
                checkpoint_sha256=sha256_file(directory/'pre_2022_model.zip')))
        if state and state['phase']=='validation':
            env,obs,rows=state['env'],state['obs'],state['rows']
        else:
            env=trading_environment(bundle)
            obs,_=env.reset(seed=seed)
            rows=[]
        scored_observations=[]
        while True:
            decision=env.unwrapped._date(env.unwrapped._decision_index)
            index=int(data.sessions.get_loc(decision))
            scored_observations.append(np.asarray(obs).copy())
            reason=schedule_at(data.vix,index,trainer.last_update,fixed=arm.endswith('fixed_nav'))
            action,changed=decide_then_update(trainer,obs,index,reason,env,allow_update=not smoke)
            obs,reward,terminated,truncated,info=env.step(action)
            row=scoring_row(info,reward,terminated,truncated,action)
            row['policy_updated_after_decision']=changed
            rows.append(row)
            if terminated or truncated:
                if not terminated or truncated:
                    raise RuntimeError('scoring failed natural termination')
                break
            if changed:
                save_state(directory,trainer,'validation',index+1,env,obs,rows)
        error=sum(r['reward'] for r in rows)-math.log(rows[-1]['end_nav']/rows[0]['start_nav'])
        if abs(error)>1e-10:
            raise RuntimeError('scoring reward/NAV identity failed')
        measures=metrics(rows)
        net=np.array([r['net_return'] for r in rows])
        cash=np.array([r['cash_benchmark_return'] for r in rows])
        measures.update(sharpe_numerator=float((net-cash).mean()),
                        sharpe_denominator=float((net-cash).std(ddof=1)))
        trajectory=directory/'validation.jsonl'
        with trajectory.open('x',encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row,default=str,allow_nan=False)+'\n')
        trainer.model.save(directory/'final_model.zip')
        # Retain the complete final account and learning clock for subsequent
        # deployment. Future input extension must reconnect the environment at
        # this boundary; it must not reset positions or refit the scaler.
        if not smoke:
            save_state(directory,trainer,'deployment',int(data.sessions.get_loc(pd.Timestamp(rows[-1]['valuation_date']))),env,obs,rows)
            write_new(directory/'deployment_contract.json',dict(
                adaptive_updates_allowed=True,learning_rate=3e-5,
                fine_tune_transitions=32768,scaler_frozen=True,
                last_observed_date=rows[-1]['valuation_date'],
                last_completed_update_date=str(data.sessions[trainer.last_update].date()),
                instruction='Extend causal market inputs and reconnect at saved close; preserve ledger, rule state, optimizer and clock. Data-boundary termination is not an account reset.'))
        result=dict(status='PASS',namespace=namespace,label=label,arm=arm,fold=fold,seed=seed,
                    technical_pass=True,config_version=config.raw['version'],audit=audit,
                    known_period_accessed=fold=='known',evaluation_reward='true_nav_log_return',
                    evaluation_learning='causal_adaptive' if not smoke else 'smoke_score_only',
                    metrics=measures,baselines=_baselines_with_rule(bundle,spy),
                    action_trace=_action_trace(rows,bundle,'v9_trend_gate'),
                    responsiveness=responsiveness(trainer.model,scored_observations),
                    updates=trainer.updates,completed_fine_tunes=len(trainer.updates)-1,
                    total_transitions=sum(u['realized_transitions'] for u in trainer.updates),
                    trajectory_sha256=sha256_file(trajectory),
                    final_checkpoint_sha256=sha256_file(directory/'final_model.zip'),
                    reward_nav_error=error,seconds=time.perf_counter()-started)
        write_new(record_path,result)
        write_new(attempt,dict(status='PASS',record=str(record_path),record_sha256=sha256_file(record_path)))
        return result
    except Exception as exc:
        write_new(attempt,dict(status='FAILED',label=label,type=type(exc).__name__,error=str(exc)))
        raise

def contrast(records, treatment, control):
    pairs=[]
    for fold in ('fold_1','fold_2','fold_3'):
        for seed in range(5):
            t=records.get((treatment,fold,seed)); c=records.get((control,fold,seed))
            if not t or not c or not t.get('technical_pass') or not c.get('technical_pass'):
                return dict(status='INCOMPLETE',missing_pair=[fold,seed],treatment=treatment,control=control)
            pairs.append(dict(fold=fold,seed=seed,
                delta=math.log(t['metrics']['final_wealth']/c['metrics']['final_wealth'])))
    medians={fold:float(np.median([p['delta'] for p in pairs if p['fold']==fold])) for fold in ('fold_1','fold_2','fold_3')}
    def aggregate_metric(arm,key,absolute=False):
        return float(np.median([np.median([abs(records[arm,f,s]['metrics'][key]) if absolute else
                   records[arm,f,s]['metrics'][key] for s in range(5)]) for f in ('fold_1','fold_2','fold_3')]))
    mdd_t=aggregate_metric(treatment,'max_drawdown',True)
    mdd_c=aggregate_metric(control,'max_drawdown',True)
    cost_t=aggregate_metric(treatment,'total_transaction_cost')
    cost_c=aggregate_metric(control,'total_transaction_cost')
    gates=dict(positive_folds=sum(v>0 for v in medians.values())>=2,
               positive_pairs=sum(p['delta']>0 for p in pairs)>=10,
               drawdown=mdd_t-mdd_c<=.02+1e-12,
               cost=cost_t<=1.1*cost_c+1e-9 if cost_c>0 else cost_t==0)
    return dict(status='PASS' if all(gates.values()) else 'NOT_ADOPTED',gates=gates,
                treatment=treatment,control=control,pairs=pairs,fold_medians=medians,
                mean_fold_median=float(np.mean(list(medians.values()))),
                fold_win_counts={f:sum(p['delta']>0 for p in pairs if p['fold']==f) for f in medians},
                drawdown_difference_pp=100*(mdd_t-mdd_c),
                control_cost=cost_c,treatment_cost=cost_t)

def aggregate(root, tag):
    records={}
    for arm in ARMS:
        for fold in ('fold_1','fold_2','fold_3'):
            for seed in range(5):
                path=root/'experiments/v11_ppo/records'/f'formal_{arm}_{fold}_seed{seed}_{tag}.json'
                if path.exists():
                    record=json.loads(path.read_text())
                    if record['namespace']!='formal' or record['config_version']!='v11spec2_200k':
                        raise ValueError('incompatible record')
                    records[arm,fold,seed]=record
    reward=contrast(records,ARMS[1],ARMS[0]); schedule=contrast(records,ARMS[0],ARMS[2])
    result=dict(reward=reward,schedule=schedule,
                adopt_complete_vix_mix=reward['status']==schedule['status']=='PASS',
                complete_cells=len(records),expected_cells=45)
    write_new(root/'reports/v11_ppo'/f'{tag}_summary.json',result)
    return result
