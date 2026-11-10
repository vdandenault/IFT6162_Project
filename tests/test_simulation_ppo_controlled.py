from dataclasses import asdict,replace
import torch
from lake_como.simulation.ppo_agent import PPOConfig,gae
from lake_como.simulation.ppo_controlled import train,same_state
from lake_como.simulation.ppo_study import train as original_train,aggregate as original_aggregate


def test_collector_control_reproduces_model_optimizer_and_counters(tmp_path):
    original_train(tmp_path/'old',777,128)
    train(tmp_path/'new',777,128,PPOConfig())
    old=torch.load(tmp_path/'old/final.pt',weights_only=True)
    new=torch.load(tmp_path/'new/final.pt',weights_only=True)
    assert same_state(old,new)


def test_longer_gae_targets_change_one_factor_and_preserve_discount():
    old=PPOConfig();new=replace(old,gae_lambda=.99)
    assert [k for k in asdict(old) if asdict(old)[k]!=asdict(new)[k]]==['gae_lambda']
    # A delayed cost reaches an earlier action more strongly without changing
    # physical rewards or the discount factor in the Bellman objective.
    args=([0.,0.,-1.],[0.,0.,0.],[0.,0.,0.],[False,False,True],[False]*3)
    short,_=gae(*args,old.gamma,old.gae_lambda)
    longer,_=gae(*args,new.gamma,new.gae_lambda)
    assert longer[0]<short[0] and longer[-1]==short[-1]
    assert old.gamma==new.gamma
