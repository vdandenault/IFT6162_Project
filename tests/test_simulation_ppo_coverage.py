from copy import deepcopy
import numpy as np
import pytest
import torch
from lake_como.simulation.ppo_coverage import allocation,first_replicates,eligible,refit,CHECK_PAIRS,digest_actor
from lake_como.simulation.ppo_agent import PPOAgent
from lake_como.simulation.hydraulic_ppo import HydraulicPPOAgent
from lake_como.simulation.ppo_seed_gain import policy_branch
from lake_como.simulation.ppo_diagnosis import make_env
from lake_como.simulation.ppo_protocol import ObservationHistory


def test_year_diversity_trades_repeats_at_identical_sample_budget():
    checks={y for pair in CHECK_PAIRS for y in pair}
    for arm,years in [('concentrated',12),('diverse',24)]:
        cases=allocation(arm)
        assert sum(reps*731 for _,reps in cases)==17544
        selected={y for pair,_ in cases for y in pair}
        assert len(selected)==years and not selected&checks
        assert max(selected)<=1987
    assert max(y for p,_ in allocation('diverse') for y in p)==1987


def test_replicate_slicing_preserves_full_finite_tasks():
    x=np.arange(4*731*15,dtype=np.float32).reshape(4*731,15);y=np.arange(4*731,dtype=np.float32)
    a,b=first_replicates(x,y,2)
    np.testing.assert_array_equal(a,x[:1462]);np.testing.assert_array_equal(b,y[:1462])
    assert not np.shares_memory(a,x)
    with pytest.raises(ValueError):first_replicates(x[:-1],y[:-1],2)
    with pytest.raises(ValueError):first_replicates(x,y,5)


def test_gate_requires_broad_stock_and_seed_agreement():
    mse=dict(original=.3,concentrated=.4,diverse=.2);slopes=dict(original=.4,concentrated=.3,diverse=.2)
    assert eligible(mse,slopes,4)
    assert not eligible(mse,slopes,3)
    assert not eligible(dict(mse,diverse=.3),slopes,5)
    assert not eligible(mse,dict(slopes,diverse=.3),5)


def test_refit_changes_only_critic_without_actor_updates():
    torch.set_num_threads(1);agent=PPOAgent(action_low=[0],action_high=[491.61]);before=digest_actor(agent)
    x=torch.zeros((64,15));x[:,0]=torch.linspace(-.5,1,64);y=x[:,0]
    loss=float((agent.model.value(x)-y).square().mean().detach());refit(agent,x,y,123,updates=100)
    assert float((agent.model.value(x)-y).square().mean().detach())<loss
    assert digest_actor(agent)==before and agent.updates==0 and agent.steps==0


def test_common_policy_branch_preserves_factual_state_and_history():
    torch.set_num_threads(1);env,obs,info=make_env('train',(1960,1961));memory=ObservationHistory();memory.encode(obs,info)
    agent=HydraulicPPOAgent(action_low=env.action_space.low,action_high=env.action_space.high)
    state=env.state;history=deepcopy(memory.history)
    a=policy_branch(env,memory,agent,days=4);b=policy_branch(env,memory,agent,days=4)
    assert a==b and a['complete'] and a['observed_days']==4
    assert env.state==state and env.hour==0 and memory.history==history
    env.close()
