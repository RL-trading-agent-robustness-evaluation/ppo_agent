from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from victim_ppo.v11.config import load_config, ARMS
from victim_ppo.v11.data import MarketData
from victim_ppo.v11.policy import window_probabilities, period_for_vix,align_vix,schedule_at
from victim_ppo.v11.environment import RecentWindowEnv,MixedReward,trading_environment
from victim_ppo.v11.runner import contrast,save_state,restore_state,Trainer,decide_then_update
from victimagent.evaluation import scaler_sha256,optimizer_sha256
from victim_ppo.v9.runner import build_model,parameter_sha256

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def config():
    return load_config(ROOT/'configs/ppo_v11.yaml')

@pytest.fixture(scope='module')
def data(config):
    return MarketData(ROOT,config,end='2016-12-31')

def test_frozen_budget(config,tmp_path):
    assert config.raw['initial_requested_transitions']==200000
    assert config.raw['initial_expected_transitions']==200704
    text=config.path.read_text().replace('recent_mass: 0.8','recent_mass: 0.9')
    path=tmp_path/'bad.yaml';path.write_text(text)
    with pytest.raises(ValueError):load_config(path)

@pytest.mark.parametrize('vix,period',[(10,252),(15,224),(20,126),(25,81),(30,56),(40,32),(50,21)])
def test_vix_table(vix,period):
    assert period_for_vix(vix)[1]==period

def test_probabilities():
    starts,ages,p=window_probabilities(252)
    assert len(starts)==189 and ages[-1]==0 and ages[0]==188
    assert abs(p.sum()-1)<1e-12 and np.all(p>0)
    exp=(p-.2/len(p))/.8
    assert exp[-1]/exp[-127]==pytest.approx(2)
    with pytest.raises(ValueError):window_probabilities(63)
    _,_,small=window_probabilities(64)
    assert small.tolist()==[1.]

def test_lag_fill_and_scheduler():
    sessions=pd.bdate_range('2010-01-01',periods=12)
    raw=pd.DataFrame({'observation_date':sessions,'VIXCLS':[20]*8+[np.nan]*4})
    vix=align_vix(raw,sessions)
    assert vix.value.iloc[10]==20 and np.isnan(vix.value.iloc[11])
    assert schedule_at(vix,11,0)['vix_mean']==20
    with pytest.raises(ValueError):schedule_at(vix,12,0)
    vix.loc[sessions[10],'value']=40
    assert schedule_at(vix,10,0)['vix_mean']==20 # current day excluded
    normal=pd.DataFrame({'raw':[20]*140,'value':[20]*140,'filled':[False]*140})
    assert not schedule_at(normal,125,0)['due']
    assert schedule_at(normal,126,0)['due']
    high=normal.copy();high.loc[121:125,'value']=40
    assert schedule_at(high,126,100)['period']==32
    assert not schedule_at(high,126,100,fixed=True)['due']

def test_prefix_and_frozen_scaler(data):
    cutoff=data.sessions[251]
    bundle,spy=data.bundle(cutoff)
    assert len(bundle.market_true)==252
    assert bundle.market_true.date.min()>=pd.Timestamp('2010-01-01')
    original=bundle.market_obs.copy();original_spy=spy.copy()
    future=data.market.date>cutoff
    saved=data.market.copy()
    data.market.loc[future,'close_raw']*=2
    try:
        second,spy_second=data.bundle(cutoff)
        pd.testing.assert_frame_equal(original,second.market_obs)
        pd.testing.assert_series_equal(original_spy,spy_second)
        assert second.observation_builder is data.builder
        assert scaler_sha256(data.builder)==data.scaler_hash
    finally:data.market=saved
    with pytest.raises(ValueError):data.bundle(data.sessions[250])

def test_reward_identity_same_path_and_reset(data):
    bundle,spy=data.bundle(data.sessions[251])
    nav=MixedReward(trading_environment(bundle),spy,False)
    mix=MixedReward(trading_environment(bundle),spy,True)
    o1,_=nav.reset(seed=0);o2,_=mix.reset(seed=0)
    np.testing.assert_array_equal(o1,o2)
    totals=np.zeros(4)
    for index in range(251):
        action=index%3
        o1,r1,t1,x1,i1=nav.step(action)
        o2,r2,t2,x2,i2=mix.step(action)
        np.testing.assert_array_equal(o1,o2)
        assert i1['end_close_nav']==i2['end_close_nav']
        assert i1['transaction_cost']==i2['transaction_cost']
        assert (t1,x1)==(t2,x2)
        assert r2==pytest.approx(r1-.2*i2['spy_log_return']+.5*i2['positive_streak_bonus'])
        assert i2['weighted_streak_bonus']<=.2*max(r1,0)+1e-14
        totals += [r1,r2,i2['spy_log_return'],i2['positive_streak_bonus']]
    wealth=np.log(i1['end_close_nav']/1e6)
    assert abs(totals[0]-wealth)<1e-10
    assert abs(totals[1]-(wealth-.2*totals[2]+.5*totals[3]))<1e-10
    mix.reset();assert mix.streak==0

def test_recent_newest_window_bootstraps(data):
    bundle,spy=data.bundle(data.sessions[251])
    env=RecentWindowEnv(bundle,spy,True,0)
    env.probabilities[:]=0;env.probabilities[-1]=1
    env.reset()
    for i in range(63):obs,reward,terminated,truncated,info=env.step(0)
    assert not terminated and truncated
    assert info['valuation_date']==data.sessions[251]
    env.reset();assert env.env.streak==0

def test_sampler_reproducible_and_same_across_rewards(data):
    bundle,spy=data.bundle(data.sessions[251])
    nav=RecentWindowEnv(bundle,spy,False,42)
    mix=RecentWindowEnv(bundle,spy,True,42)
    for _ in range(10):
        _,left=nav.reset();_,right=mix.reset()
        assert left['decision_index']==right['decision_index']
    assert nav.sampled_ages==mix.sampled_ages

@pytest.mark.parametrize('start',[0,17,188])
def test_cached_training_lookup_matches_parent_exactly(data,start):
    bundle,spy=data.bundle(data.sessions[251])
    original=trading_environment(bundle,training=True)
    cached=trading_environment(bundle,training=True,cached=True)
    left,lreset=original.reset(seed=42,options={'start_index':start,'episode_length':63})
    right,rreset=cached.reset(seed=42,options={'start_index':start,'episode_length':63})
    np.testing.assert_array_equal(left,right)
    assert lreset==rreset
    for i in range(63):
        left,lr,lt,lx,li=original.step(i%3)
        right,rr,rt,rx,ri=cached.step(i%3)
        np.testing.assert_array_equal(left,right)
        assert (lr,lt,lx,li)==(rr,rt,rx,ri)

def test_action_locked_before_update_and_hold_state_preserved(data):
    from types import SimpleNamespace
    bundle,spy=data.bundle(data.sessions[251])
    env=trading_environment(bundle)
    obs,_=env.reset();obs,*_=env.step(0)
    class Model:
        action=0
        def predict(self,observation,deterministic):
            assert deterministic
            return self.action,None
    model=Model()
    def update(index,reason):model.action=1
    trainer=SimpleNamespace(model=model,train=update)
    hold_state=env.env.sessions_since_change
    action,changed=decide_then_update(trainer,obs,252,{'due':True},env)
    assert action==0 and changed and env.env.sessions_since_change==hold_state
    action,changed=decide_then_update(trainer,obs,253,{'due':False},env)
    assert action==1 and not changed

def records_for_metrics():
    records={}
    for arm in ARMS:
        for fold in ('fold_1','fold_2','fold_3'):
            for seed in range(5):
                records[arm,fold,seed]=dict(technical_pass=True,metrics=dict(
                    final_wealth=1.2 if arm==ARMS[1] else 1.,
                    max_drawdown=-.12 if arm==ARMS[1] else -.1,
                    total_transaction_cost=110 if arm==ARMS[1] else 100))
    return records

def test_gates_and_missing_pair():
    records=records_for_metrics()
    assert contrast(records,ARMS[1],ARMS[0])['status']=='PASS'
    records[ARMS[1],'fold_1',0]['technical_pass']=False
    assert contrast(records,ARMS[1],ARMS[0])['status']=='INCOMPLETE'

def test_zero_cost():
    records=records_for_metrics()
    for record in records.values():record['metrics']['total_transaction_cost']=0
    assert contrast(records,ARMS[1],ARMS[0])['gates']['cost']
    for fold in ('fold_1','fold_2','fold_3'):
        for seed in range(5):records[ARMS[1],fold,seed]['metrics']['total_transaction_cost']=1
    assert not contrast(records,ARMS[1],ARMS[0])['gates']['cost']

def test_resume_roundtrip_and_hash_guard(data,config,tmp_path):
    import torch
    torch.set_num_threads(1)
    bundle,spy=data.bundle(data.sessions[251])
    trainer=Trainer(data,config,tmp_path,ARMS[0],0)
    trainer.model=build_model(RecentWindowEnv(bundle,spy,False,0),config,seed=0)
    env=trading_environment(bundle);obs,_=env.reset()
    obs,*_=env.step(0)
    previous=params=parameter_sha256(trainer.model)
    optimizer=optimizer_sha256(trainer.model)
    save_state(tmp_path,trainer,'validation',252,env,obs,[])
    state=restore_state(tmp_path/'resume_validation_00252',trainer)
    assert parameter_sha256(trainer.model)==params
    assert optimizer_sha256(trainer.model)==optimizer
    assert state['env'].env.sessions_since_change==env.env.sessions_since_change
    assert state['env'].unwrapped.ledger.snapshot(100).nav==env.unwrapped.ledger.snapshot(100).nav
    path=tmp_path/'resume_validation_00252/state.pkl'
    path.write_bytes(path.read_bytes()+b'tampered')
    with pytest.raises(RuntimeError):restore_state(path.parent,trainer)
