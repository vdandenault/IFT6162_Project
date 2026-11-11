import numpy as np
import torch
from reservoir_control.systems.lake_como.simulation.ppo_agent import PPOAgent
from reservoir_control.systems.lake_como.simulation.ppo_protocol import ObservationHistory
from reservoir_control.systems.lake_como.simulation.ppo_diagnosis import make_env
from reservoir_control.systems.lake_como.simulation.ppo_identification import choose,snapshot,restore,daily_mean,digest_actor,sample_saturation


def state():
    env,obs,info=make_env('train',(1960,1961));memory=ObservationHistory()
    for _ in range(8):
        memory.encode(obs,info);obs,_,_,_,info=env.step(np.array([200],np.float32))
    memory.encode(obs,info);return env,memory


def test_snapshot_preserves_unrounded_state_and_history():
    env,memory=state();record=snapshot(env,memory);other,past=restore(env.years,record)
    assert other.state==env.state and other.hour==env.hour and other.demand_ema==env.demand_ema
    np.testing.assert_array_equal(past.encode(other._observation(),other._info()),memory.last_encoded)
    assert past.history==memory.history;other.close();env.close()


def test_daily_release_audit_matches_full_step_without_mutation():
    env,memory=state();before=env.state;hour=env.hour;q=daily_mean(env,200)
    assert env.state==before and env.hour==hour
    _,_,_,_,info=env.step(np.array([200],np.float32))
    assert q==np.mean([r['release_m3_s'] for r in info['hourly']]);env.close()


def test_critic_updates_cannot_change_actor_digest():
    torch.set_num_threads(1);agent=PPOAgent(action_low=[0],action_high=[491.61]);before=digest_actor(agent)
    x=torch.zeros((16,15));x[:,0]=torch.linspace(-.5,1,16);y=2*x[:,0]
    optimizer=torch.optim.Adam(agent.model.critic.parameters(),lr=.003)
    first=float((agent.model.value(x)-y).square().mean().detach())
    for _ in range(100):
        optimizer.zero_grad();loss=(agent.model.value(x)-y).square().mean();loss.backward();optimizer.step()
    assert float(loss.detach())<first/10 and digest_actor(agent)==before
    with torch.no_grad():agent.model.log_std.add_(.01)
    assert digest_actor(agent)!=before


def test_locked_choice_and_completion_rules():
    assert choose(False,.99,1,.1,1,.1)=='incomplete'
    assert choose(True,.50,1,.1,1,.1)=='action_transform'
    assert choose(True,.49,1,.8,1,.8)=='critic_training'
    assert choose(True,.49,1,.7,1,.9)=='none_qualified'
