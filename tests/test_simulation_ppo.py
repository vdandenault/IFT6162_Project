import numpy as np
import pytest
import torch
from reservoir_control.systems.lake_como.simulation.ppo_agent import PPOAgent,PPOConfig,gae,log_probability
from reservoir_control.systems.lake_como.simulation.ppo_protocol import ObservationHistory,MatchedTrainingEpisodes


def test_gae_reset_and_bootstrap():
    a,r=gae([1,2,3],[10,20,30],[20,30,40],[False,True,False],[False,False,True],.9,1)
    np.testing.assert_allclose(a,[-7.2,-18,9],atol=1e-5)
    np.testing.assert_allclose(r,[2.8,2,39],atol=1e-5)


def test_causal_history_and_year_rollover():
    history=ObservationHistory();raw=np.zeros(8,np.float32);raw[4]=99
    first=history.encode(raw,{'elapsed_hours':0});assert first[-1]==0
    for day in range(1,367):
        raw[4]=day;encoded=history.encode(raw,{'elapsed_hours':day*24})
    np.testing.assert_array_equal(encoded[8:14],[365,364,363,362,361,360])
    assert encoded[-1]==1
    np.testing.assert_array_equal(history.encode(raw,{'elapsed_hours':366*24}),encoded)
    with pytest.raises(ValueError):history.encode(raw,{'elapsed_hours':0})


def test_update_and_checkpoint(tmp_path):
    torch.set_num_threads(1)
    agent=PPOAgent(PPOConfig(epochs=2,batch_size=8),action_low=[0],action_high=[491.61])
    obs=np.ones(15,np.float32);rollout={k:[] for k in ['observation','latent','log_probability','reward','value','next_value','terminated','truncated']}
    before=[p.detach().clone() for p in agent.model.parameters()]
    for i in range(32):
        action,z,lp,value=agent.act(obs)
        assert 0<=action[0]<=491.61
        for key,item in dict(observation=obs,latent=z,log_probability=lp,reward=-float(i%5),value=value,next_value=value,terminated=i==31,truncated=False).items():rollout[key].append(item)
    log=agent.update(rollout)
    assert log['actor_gradient_norm']>0 and log['critic_gradient_norm']>0
    assert any(not torch.equal(p,q) for p,q in zip(before,agent.model.parameters()))
    path=tmp_path/'model.pt';agent.save(path);loaded=PPOAgent.load(path)
    np.testing.assert_array_equal(agent.act(obs,deterministic=True)[0],loaded.act(obs,deterministic=True)[0])
    distribution=torch.distributions.Normal(torch.zeros((2,1)),torch.ones((2,1)))
    assert torch.isfinite(log_probability(distribution,torch.tensor([[-100.],[100.]]))).all()


def test_matched_task_no_burnin():
    env=MatchedTrainingEpisodes();obs,info=env.reset(seed=123)
    assert len(env.windows)==41 and env.burn_in_transitions==0
    assert obs[0]==np.float32(.35) and obs[4]==np.float32(env.prior_last_flow)
    assert info['elapsed_hours']==0
    env.close()


def test_failed_required_seed_suppresses_study_mean(tmp_path, monkeypatch):
    import json
    from reservoir_control.systems.lake_como.simulation import ppo_study

    reference = tmp_path / 'reference'
    reference.mkdir()
    (reference / 'annual.csv').write_text('arm,year,seed\nlegacy,1995,123\n')
    monkeypatch.setattr(ppo_study, 'REFERENCE', reference)
    aggregate, SEEDS = ppo_study.aggregate, ppo_study.SEEDS
    for seed in SEEDS:
        folder=tmp_path/f'seed{seed}';folder.mkdir()
        (folder/'training.json').write_text(json.dumps({'complete':False}))
    aggregate(tmp_path,{},[])
    result=json.loads((tmp_path/'summary.json').read_text())
    assert not result['all_required_complete']
    assert result['complete_training_seeds']==0
    assert all(value is None for metrics in result['means'].values() for value in metrics.values())


def test_shared_ppo_uses_explicit_observation_scales(tmp_path):
    from reservoir_control.controllers.ppo.agent import PPOAgent as SharedPPOAgent
    agent = SharedPPOAgent(PPOConfig(observation_dim=3), action_low=[-1], action_high=[1],
                           observation_scales=[1, 10, 100])
    observation = np.array([1, 2, 3], np.float32)
    action = agent.act(observation, deterministic=True)[0]
    path = tmp_path/'shared.pt'
    agent.save(path)
    restored = SharedPPOAgent.load(path)
    np.testing.assert_array_equal(restored.act(observation, deterministic=True)[0], action)
    with pytest.raises(ValueError, match='scale'):
        SharedPPOAgent(PPOConfig(observation_dim=3), action_low=[-1], action_high=[1],
                       observation_scales=[1, 0, 1])
