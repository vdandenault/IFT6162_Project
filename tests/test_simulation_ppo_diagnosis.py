from copy import deepcopy
import numpy as np
import torch
from lake_como.simulation.ppo_agent import PPOAgent
from lake_como.simulation.ppo_diagnosis import discounted_returns,make_env,storage_branch,first_action_branch,stochastic_return
from lake_como.simulation.ppo_protocol import ObservationHistory


def setup_state():
    torch.set_num_threads(1)
    env,obs,info=make_env('train',(1960,1961));memory=ObservationHistory()
    agent=PPOAgent(action_low=env.action_space.low,action_high=env.action_space.high)
    for _ in range(8):
        memory.encode(obs,info)
        obs,_,_,_,info=env.step(np.array([200],np.float32))
    memory.encode(obs,info)
    return env,memory,agent


def test_full_return_recursion_without_bootstrap():
    np.testing.assert_array_equal(discounted_returns([1,2,3],.5),[2.75,3.5,3])
    assert discounted_returns([],1).size==0


def test_storage_intervention_preserves_everything_else():
    env,memory,_=setup_state();original=env.state
    branch,changed=storage_branch(env,memory,.15)
    a=memory.encode(env._observation(),env._info());b=changed.encode(branch._observation(),branch._info())
    np.testing.assert_array_equal(a[1:],b[1:])
    assert abs(float(b[0])-float(a[0])-.15)<1e-7
    assert branch.hour==env.hour and branch.state.command==env.state.command
    assert branch.demand_ema==env.demand_ema and branch.last_inflow==env.last_inflow
    assert changed.history==memory.history and env.state==original
    branch.close();env.close()


def test_common_continuation_is_repeatable_and_does_not_mutate_source():
    env,memory,agent=setup_state();state=env.state;hour=env.hour;past=deepcopy(memory.history)
    a=first_action_branch(env,memory,agent,100,days=8)
    b=first_action_branch(env,memory,agent,100,days=8)
    assert a==b and a['complete'] and a['days']==8
    assert env.state==state and env.hour==hour and memory.history==past
    env.close()


def test_common_random_numbers_and_actual_finite_end():
    env,memory,agent=setup_state()
    # A declared two-decision tail from the restored state, with the returned
    # remaining-time observation refreshed, keeps this stochastic check small.
    env.total_hours=env.hour+48
    memory.last_raw=env._observation().copy();memory.last_encoded[:8]=memory.last_raw
    a=stochastic_return(env,memory,agent,0,999)
    b=stochastic_return(env,memory,agent,0,999)
    assert a==b and a['complete'] and a['days']==2
    assert np.isfinite(a['return_995']) and env.hour==8*24
    env.close()
