import gymnasium as gym
import numpy as np
import pandas as pd
from victim_ppo.v11.environment import MixedReward

class ScriptedNAV(gym.Env):
    observation_space=gym.spaces.Box(-1,1,(1,),dtype=np.float32)
    action_space=gym.spaces.Discrete(3)
    def __init__(self):
        self.values=[.01,.01,.01,-.01,.01,0,.01,.01,.01,.01,.01,.01]
        self.dates=pd.bdate_range('2010-01-04',periods=len(self.values))
    def reset(self,**kwargs):
        self.index=0
        return np.zeros(1,dtype=np.float32),{}
    def step(self,action):
        i=self.index;self.index+=1
        return np.zeros(1,dtype=np.float32),self.values[i],self.index==len(self.values),False,{'valuation_date':self.dates[i]}

def test_explicit_streak_zero_and_cap():
    scripted=ScriptedNAV()
    env=MixedReward(scripted,pd.Series(0.,index=scripted.dates),True)
    env.reset()
    streaks=[];bonuses=[]
    for _ in scripted.values:
        _,_,_,_,info=env.step(0)
        streaks.append(info['positive_return_streak'])
        bonuses.append(info['weighted_streak_bonus'])
    assert streaks==[1,2,3,0,1,0,1,2,3,4,5,6]
    np.testing.assert_allclose(bonuses,[0,.0005,.001,0,0,0,0,.0005,.001,.0015,.002,.002])
    env.reset();assert env.streak==0
