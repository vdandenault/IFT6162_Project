import math
import numpy as np
import pytest
import torch
from lake_como.simulation.hydraulic_ppo import HydraulicTransform,HydraulicPPOAgent
from lake_como.simulation.ppo_agent import PPOAgent
from lake_como.simulation.ppo_controlled import same_state
from lake_como.simulation.ppo_study import train as original_train
from lake_como.simulation.hydraulic_ppo_study import train


def observation(level,day=1):
    phase=2*math.pi*(day-1)/365
    return np.array([level,math.sin(phase),math.cos(phase),100,100,200,0,.7,100,100,100,100,100,100,1],np.float32)


def test_map_retains_entire_nominal_range_and_constant_endpoints():
    transform=HydraulicTransform()
    for level in [-.5,-.49,-.2,.35,1.24,1.26,2.5]:
        for day in [1,60,180,365]:
            obs=observation(level,day)
            values=np.array([transform(obs,[q])[0] for q in np.linspace(transform.low,transform.high,201)])
            assert values[0]==transform.low and values[-1]==transform.high
            assert (np.diff(values)>=0).all() and values.min()>=transform.low and values.max()<=transform.high
            lo,hi=transform.interval(obs)
            if hi-lo>1e-6:
                assert abs(float(transform(obs,[.5*transform.high])[0])-.5*(lo+hi))<1e-4


def test_transform_does_not_use_inflow_target_or_future_state():
    transform=HydraulicTransform();a=observation(.35,180);b=a.copy();b[3:]=np.arange(12)*1000
    np.testing.assert_array_equal(transform(a,[300]),transform(b,[300]))
    # A collapsed current interval retains nominal choices for changing levels later in the day.
    a=observation(1.26)
    assert transform.interval(a)[0]==transform.interval(a)[1]
    np.testing.assert_array_equal(transform(a,[300]),np.array([300],np.float32))


def test_latent_likelihood_and_initial_networks_are_unchanged(tmp_path):
    torch.set_num_threads(1);raw=PPOAgent(action_low=[0],action_high=[491.61],seed=123)
    mapped=HydraulicPPOAgent(action_low=[0],action_high=[491.61],seed=123)
    assert same_state(raw.model.state_dict(),mapped.model.state_dict())
    obs=observation(-.3)
    torch.manual_seed(999);a=raw.act(obs)
    torch.manual_seed(999);b=mapped.act(obs)
    np.testing.assert_array_equal(a[1],b[1]);assert a[2:]==b[2:]
    np.testing.assert_array_equal(b[0],mapped.transform(obs,a[0]))
    path=tmp_path/'model.pt';mapped.save(path);loaded=HydraulicPPOAgent.load(path)
    np.testing.assert_array_equal(mapped.act(obs,deterministic=True)[0],loaded.act(obs,deterministic=True)[0])
    path.with_suffix('.action.json').unlink()
    with pytest.raises(FileNotFoundError):HydraulicPPOAgent.load(path)


def test_identity_collector_control_is_bitwise_equal(tmp_path):
    original_train(tmp_path/'old',777,128)
    train(tmp_path/'new',777,128,agent_type=PPOAgent)
    assert same_state(torch.load(tmp_path/'old/final.pt',weights_only=True),torch.load(tmp_path/'new/final.pt',weights_only=True))
