"""Read-only SAC timing and critic diagnostics; new branches use training years only.

uv run --group reproduction python -m reservoir_control.systems.lake_como.sac_diagnostics
No training, controller changes, validation selection, or test-year evaluation.
"""
import argparse
from copy import deepcopy
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from reservoir_control.systems.lake_como.environment import make_env, settings
from reservoir_control.systems.lake_como.inflow_observation import PastInflowObservation
from reservoir_control.systems.lake_como.mpc_hindsight import digest
from reservoir_control.systems.lake_como.paths import PROJECT_ROOT
from reservoir_control.systems.lake_como.sac import SACAgent
from reservoir_control.systems.lake_como.sac_study import verify_plan
from reservoir_control.systems.lake_como.splits import split_years

STUDY = PROJECT_ROOT / 'results/sac/baseline_200k_v1'


def selected_training_years(inflows):
    """Minimum, lower median, maximum total inflow; ties resolved by year."""
    if set(inflows) != set(split_years('train')):
        raise ValueError('Training years only')
    ranked = sorted(inflows, key=lambda y: (sum(inflows[y][:365]), y))
    return dict(dry=ranked[0], median=ranked[(len(ranked)-1)//2], wet=ranked[-1])


def water_allocations(frame, environmental_flow):
    """An accounting identity, not the simulator's seasonally weighted reward."""
    result = frame.copy()
    release = result.release_m3_s.to_numpy()
    demand = result.demand_m3_s.to_numpy()
    mef = np.asarray(environmental_flow)[result.day.to_numpy(dtype=int)-1]
    available = np.maximum(release-mef, 0.)
    result['useful_m3'] = np.minimum(available, demand)*86400
    result['surplus_after_mef_m3'] = np.maximum(available-demand, 0.)*86400
    result['environmental_allocation_m3'] = np.minimum(release, mef)*86400
    result['released_m3'] = release*86400
    result['absolute_clipping_m3_s'] = np.abs(result.request_m3_s-release)
    result['month'] = pd.date_range('2001-01-01', periods=365).month.to_numpy()[result.day.to_numpy(dtype=int)-1]
    np.testing.assert_allclose(result.useful_m3+result.surplus_after_mef_m3+
                               result.environmental_allocation_m3, result.released_m3, atol=1e-6)
    np.testing.assert_allclose(result.useful_m3+result.deficit_m3, demand*86400, atol=1e-6)
    return result


def timing_analysis(output, params):
    sac = pd.read_csv(STUDY/'trajectories.csv', float_precision='round_trip').query('variant=="plain"').copy()
    ppo = pd.read_csv(PROJECT_ROOT/'results/safety/common_filter_v1/trajectories.csv',
                      float_precision='round_trip').query('family=="ppo" and variant=="unfiltered"').copy()
    sac['family'], ppo['family'] = 'sac', 'ppo'
    keys = ['training_seed', 'year', 'day']
    paired = sac.merge(ppo, on=keys, suffixes=('_sac', '_ppo'), validate='one_to_one')
    assert len(paired)==18250
    for column in ['inflow_m3_s', 'demand_m3_s']:
        np.testing.assert_array_equal(paired[column+'_sac'], paired[column+'_ppo'])
    frame = water_allocations(pd.concat([sac, ppo], ignore_index=True), params['lake_params']['min_env_flow'])
    columns = ['deficit_m3', 'useful_m3', 'surplus_after_mef_m3', 'environmental_allocation_m3',
               'released_m3', 'flood', 'low_water'] + [c for c in frame if c.startswith('weighted_')]
    annual = frame.groupby(['family', 'training_seed', 'year'])[columns].sum().reset_index()
    means = annual.groupby('family')[columns].mean()
    monthly = frame.groupby(['family', 'month']).agg(
        request_m3_s=('request_m3_s', 'mean'), release_m3_s=('release_m3_s','mean'),
        demand_m3_s=('demand_m3_s', 'mean'), level_m=('level_m','mean'),
        clipping_m3_s=('absolute_clipping_m3_s','mean'),
        unmet_million_m3=('deficit_m3', lambda x: x.sum()/50/1e6),
        surplus_million_m3=('surplus_after_mef_m3', lambda x: x.sum()/50/1e6),
        low_water_days=('low_water', lambda x: x.sum()/50)).reset_index()
    terminal = frame.query('day==365').groupby('family')[['level_m','storage_m3']].mean()
    delta = means.loc['sac']-means.loc['ppo']
    # Exact volume decomposition of the extra unmet demand (fixed annual demand).
    decomposition = dict(extra_unmet_million_m3=delta.deficit_m3/1e6,
                         less_total_release_million_m3=-delta.released_m3/1e6,
                         extra_surplus_million_m3=delta.surplus_after_mef_m3/1e6,
                         extra_environmental_allocation_million_m3=delta.environmental_allocation_m3/1e6)
    np.testing.assert_allclose(decomposition['extra_unmet_million_m3'],
                               sum(v for k,v in decomposition.items() if k!='extra_unmet_million_m3'), atol=1e-7)
    np.testing.assert_allclose(-delta.released_m3,
                               terminal.loc['sac','storage_m3']-terminal.loc['ppo','storage_m3'], atol=1e-6)
    penalties = pd.DataFrame({'ppo': -means.loc['ppo'].filter(like='weighted_'),
                              'sac': -means.loc['sac'].filter(like='weighted_')})
    penalties['extra_sac_penalty'] = penalties.sac-penalties.ppo
    for name, table in [('timing_annual',annual), ('timing_monthly',monthly),
                        ('water_allocation',means.reset_index()), ('penalty_decomposition',penalties.reset_index())]:
        table.to_csv(output/f'{name}.csv',index=False)
    result = dict(volume_decomposition=decomposition,
                  mean_absolute_clipping_m3_s=frame.groupby('family').absolute_clipping_m3_s.mean().to_dict(),
                  terminal_level_m=terminal.level_m.to_dict(),
                  extra_penalties=penalties.extra_sac_penalty.to_dict(),
                  interpretation='Saved validation trajectories only; accounting identity, not an intervention or proof of why the agent learned this behavior.')
    (output/'timing_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return monthly, penalties


@torch.no_grad()
def critic_values(agent, observation, requests):
    requests = np.asarray(requests,np.float32).reshape(-1,1)
    normalized = 2*(requests-agent.action_low.numpy())/(agent.action_high.numpy()-agent.action_low.numpy())-1
    observations = torch.from_numpy(np.repeat(np.asarray(observation,np.float32)[None,:],len(requests),axis=0))
    scaled = agent.normalize(observations)
    actions = torch.from_numpy(normalized)
    return agent.q1(scaled,actions).numpy().ravel(), agent.q2(scaled,actions).numpy().ravel()


def rollout_branch(snapshot, agent, first_request, noise_seed):
    """Conditional historical-year soft return; excludes entropy of first action.

    Future actions follow the frozen stochastic actor. Identical noise seeds across
    first requests give common random numbers. This is not a deployable oracle.
    """
    env = deepcopy(snapshot)
    generator = torch.Generator().manual_seed(noise_seed)
    gamma, scale, alpha = agent.config.gamma, agent.config.reward_scale, agent.alpha.item()
    obs, reward, terminated, truncated, _ = env.step(np.array([first_request],np.float32))
    soft, raw, discount, steps = scale*reward, reward, 1., 1
    try:
        while not (terminated or truncated):
            with torch.no_grad():
                action, logp = agent.actor(agent.normalize(torch.from_numpy(obs).unsqueeze(0)),generator=generator)
                request = agent.physical_action(action)[0].numpy()
            obs, reward, terminated, truncated, _ = env.step(request)
            discount *= gamma
            soft += discount*(scale*reward-alpha*logp.item())
            raw += discount*reward
            steps += 1
        return dict(soft_return=soft, discounted_original_return=raw, continuation_steps=steps)
    finally:
        env.close()


def one_step(snapshot, request):
    env = deepcopy(snapshot)
    try:
        obs,reward,terminated,truncated,info=env.step(np.array([request],np.float32))
        return obs,reward,info, bool(terminated or truncated)
    finally:
        env.close()


def same_transition(left, right):
    """Exact physical next-state identity, including the policy-visible observation."""
    return (np.array_equal(left[0],right[0]) and left[2]['storage']==right[2]['storage']
            and left[2]['release']==right[2]['release'] and left[3]==right[3])


def critic_analysis(output, params, plan, years):
    rows, surfaces, branches = [], [], []
    fallback = plan['fallback']
    for seed in plan['config']['learner_seeds']:
        checkpoint=STUDY/f'seed_{seed}/final.pt'
        metadata=json.loads((checkpoint.parent/'run.json').read_text())
        assert digest(checkpoint)==metadata['checkpoint_sha256']
        agent=SACAgent.load(checkpoint)
        before={k:v.clone() for k,v in agent.state_dict().items()}
        rng_before=agent.generator.get_state().clone()
        for label,year in years.items():
            env=PastInflowObservation(make_env('train',year),fallback)
            obs,_=env.reset()
            try:
                for day in range(1,366):
                    action=agent.act(obs,deterministic=True)
                    request=float(agent.physical_action(torch.from_numpy(action))[0])
                    if day in [90,180,270,365]:
                        grid=np.unique(np.r_[np.linspace(0,float(agent.action_high[0]),21),np.float32(request)]).astype(np.float32)
                        q1,q2=critic_values(agent,obs,grid)
                        q=np.minimum(q1,q2)
                        actor_index=int(np.flatnonzero(grid==np.float32(request))[0])
                        actual=one_step(env,request)
                        steps=[one_step(env,float(a)) for a in grid]
                        immediate=np.array([x[1] for x in steps])
                        identical=np.array([same_transition(x,actual) for x in steps])
                        better=identical & (immediate>actual[1]+1e-8)
                        best_identical=int(np.argmax(np.where(identical,immediate,-np.inf)))
                        record=dict(seed=seed,year=year,year_kind=label,day=day,
                                    level_before_m=float(obs[0]),past_inflow_m3_s=float(obs[4]),
                                    actor_request_m3_s=request,actual_release_m3_s=actual[2]['release'],
                                    actor_reward=actual[1],actor_q1=q1[actor_index],actor_q2=q2[actor_index],
                                    actor_min_q=q[actor_index],critic_disagreement=abs(q1[actor_index]-q2[actor_index]),
                                    critic_best_request_m3_s=float(grid[q.argmax()]),
                                    better_same_transition_action_found=bool(better.any()),
                                    better_same_transition_actions=int(better.sum()),
                                    best_same_transition_request=float(grid[best_identical]),
                                    same_transition_reward_gain=float(immediate[best_identical]-actual[1]),
                                    predicted_same_transition_q_gain=float(q[best_identical]-q[actor_index]),
                                    true_same_transition_q_gain=float(agent.config.reward_scale*(immediate[best_identical]-actual[1])),
                                    terminal=day==365)
                        rows.append(record)
                        for index,a in enumerate(grid):
                            surfaces.append(dict(seed=seed,year=year,day=day,request_m3_s=float(a),
                                                 q1=q1[index],q2=q2[index],min_q=q[index],reward=immediate[index],
                                                 actual_release_m3_s=steps[index][2]['release'],
                                                 same_transition_as_actor=bool(identical[index])))
                        candidates=dict(actor=request,lower_50=max(0.,request-50),
                                        higher_50=min(float(agent.action_high[0]),request+50),
                                        critic_grid_best=float(grid[q.argmax()]))
                        if better.any():
                            candidates['same_transition_better']=float(grid[best_identical])
                        for candidate,first in candidates.items():
                            a,b=critic_values(agent,obs,[first])
                            repeats=1 if day==365 else 4
                            for replicate in range(repeats):
                                result=rollout_branch(env,agent,first,10000+seed*100+replicate)
                                branches.append(dict(seed=seed,year=year,day=day,candidate=candidate,
                                                     replicate=replicate,request_m3_s=first,predicted_min_q=float(min(a[0],b[0])),
                                                     **result))
                        print('Critic anchor',seed,label,year,'day',day,flush=True)
                    obs,_,terminated,truncated,_=env.step(np.array([request],np.float32))
                    assert not terminated and truncated==(day==365)
            finally:
                env.close()
        assert all(torch.equal(v,agent.state_dict()[k]) for k,v in before.items())
        assert torch.equal(rng_before,agent.generator.get_state())
    anchors=pd.DataFrame(rows);surface=pd.DataFrame(surfaces);mc=pd.DataFrame(branches)
    means=mc.groupby(['seed','year','day','candidate']).agg(
        request_m3_s=('request_m3_s','first'),predicted_min_q=('predicted_min_q','first'),
        observed_soft_return=('soft_return','mean'),soft_return_sd=('soft_return','std'),
        discounted_original_return=('discounted_original_return','mean'),replicates=('replicate','size')).reset_index()
    means['conditional_q_error']=means.predicted_min_q-means.observed_soft_return
    actor=means.query('candidate=="actor"')[['seed','year','day','predicted_min_q','observed_soft_return']]
    comparisons=means.merge(actor,on=['seed','year','day'],suffixes=('','_actor'),validate='many_to_one')
    comparisons['predicted_gain']=comparisons.predicted_min_q-comparisons.predicted_min_q_actor
    comparisons['observed_conditional_gain']=comparisons.observed_soft_return-comparisons.observed_soft_return_actor
    for name,table in [('critic_anchors',anchors),('critic_surfaces',surface),('critic_branches',mc),
                       ('critic_conditional_returns',means),('critic_action_comparisons',comparisons)]:
        table.to_csv(output/f'{name}.csv',index=False)
    return anchors,surface,means,comparisons


def plots(output, monthly, penalties, anchors, surfaces, means):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.facecolor':'#f5f8fc','text.color':'#183149','font.size':11})
    colors={'ppo':'#267ead','sac':'#27856a'}
    fig,axes=plt.subplots(2,2,figsize=(14,8),layout='constrained')
    for family,frame in monthly.groupby('family'):
        for ax,column,title in [(axes[0,0],'request_m3_s','Requested versus actual release'),
                                 (axes[0,1],'unmet_million_m3','Monthly unmet demand, million m³'),
                                 (axes[1,0],'surplus_million_m3','Monthly surplus after environmental flow, million m³'),
                                 (axes[1,1],'low_water_days','Monthly low-water days')]:
            ax.plot(frame.month,frame[column],color=colors[family],label=family.upper(),marker='o',ms=3)
            ax.set(title=title,xlabel='Canonical month (365-day approximation)',xticks=range(1,13))
            ax.spines[['top','right']].set_visible(False)
        axes[0,0].plot(frame.month,frame.release_m3_s,color=colors[family],ls='--',label=family.upper()+' actual')
    axes[0,0].legend(fontsize=9,ncol=2)
    axes[0,1].legend()
    fig.suptitle('Release timing: saved plain-controller validation trajectories\nFive seed means over 1988–1997; different learned trajectories, not identical states',weight='bold')
    fig.savefig(output/'release_timing.png',dpi=160);plt.close(fig)
    # Choose one explanatory surface by a fixed rule, not largest observed return gain.
    candidates=anchors.query('better_same_transition_action_found').sort_values(['seed','year','day'])
    chosen=(candidates if len(candidates) else anchors.sort_values(['seed','year','day'])).iloc[0]
    curve=surfaces[(surfaces.seed==chosen.seed)&(surfaces.year==chosen.year)&(surfaces.day==chosen.day)]
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    axes[0].plot(curve.request_m3_s,curve.q1,label='Q1',color='#8164a1')
    axes[0].plot(curve.request_m3_s,curve.q2,label='Q2',color='#b78219')
    axes[0].axvline(chosen.actor_request_m3_s,color='#27856a',ls='--',label='Actor request')
    axes[0].axvline(chosen.best_same_transition_request,color='#267ead',ls=':',label='Best identical-state request')
    axes[0].set(xlabel='Requested release, m³/s',ylabel='Scaled soft Q (includes future entropy)',title='What the critics estimate')
    axes[0].legend(fontsize=9)
    axes[1].plot(curve.request_m3_s,curve.actual_release_m3_s,color='#267ead',label='Actual release')
    identical=curve[curve.same_transition_as_actor]
    axes[1].scatter(identical.request_m3_s,identical.actual_release_m3_s,color='#27856a',label='Exactly identical next state')
    axes[1].set(xlabel='Requested release, m³/s',ylabel='Actual mean daily release, m³/s',title='What the simulator actually delivers')
    axes[1].legend(fontsize=9)
    fig.suptitle(f'Training-only critic probe · seed {int(chosen.seed)}, year {int(chosen.year)}, day {int(chosen.day)}\nIdentical next state isolates the immediate clipping penalty; no forecast is given to the actor',weight='bold')
    fig.savefig(output/'critic_action_surface.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    actor=means.query('candidate=="actor"')
    for terminal,label,color in [(False,'Days 90/180/270','#267ead'),(True,'Final day: exact reward target','#b78219')]:
        frame=actor[(actor.day==365)==terminal]
        axes[0].scatter(frame.observed_soft_return,frame.predicted_min_q,label=label,color=color,s=25)
    lo=min(actor.observed_soft_return.min(),actor.predicted_min_q.min());hi=max(actor.observed_soft_return.max(),actor.predicted_min_q.max())
    axes[0].plot([lo,hi],[lo,hi],color='#53687b',ls='--')
    axes[0].set(xlabel='Conditional rollout soft return',ylabel='Predicted min(Q1,Q2)',title='Historical training-year checks, not held-out calibration')
    axes[0].legend(fontsize=8)
    labels=['Flood','Low water','Demand','Surplus','Clipping']
    axes[1].bar(labels,penalties.extra_sac_penalty,color=['#b78219','#8164a1','#267ead','#27856a','#ba554c'])
    axes[1].set(ylabel='Extra annual penalty: SAC minus PPO',title='Where the validation reward gap comes from')
    fig.suptitle('Critic accuracy and observed penalty costs · separate training probes from validation accounting',weight='bold')
    fig.savefig(output/'critic_and_penalties.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'results/sac/diagnostics_200k_v1')
    args=parser.parse_args();output=args.output
    if output.exists() and any(output.iterdir()):
        parser.error('Choose a new empty output to preserve diagnostics')
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    plan=verify_plan(STUDY);params=settings('train');years=selected_training_years(params['inflow'])
    sources=[Path(__file__),STUDY/'plan.json',STUDY/'trajectories.csv',
             PROJECT_ROOT/'results/safety/common_filter_v1/trajectories.csv']
    sources += [STUDY/f'seed_{seed}/final.pt' for seed in plan['config']['learner_seeds']]
    frozen={str(p.relative_to(PROJECT_ROOT)):digest(p) for p in sources}
    declaration=dict(selected_training_years=years,anchor_days=[90,180,270,365],
        learner_seeds=plan['config']['learner_seeds'],grid_points=21,continuation_replicates=4,
        candidates='Actor, actor +/-50 m3/s bounded, critic grid best, and best identical-state alternative if any',
        scope='Post-hoc saved validation timing/accounting; all new interventions on training years. Historical-year soft returns are conditional, not unbiased state-value calibration. No tuning or test evaluation.',
        soft_return='scale*r0 + sum(t>=1) gamma^t*(scale*r_t-alpha*logpi_t); no terminal bootstrap',
        source_sha256=frozen,test_evaluated=False,torch_version=version('torch'))
    (output/'plan.json').write_text(json.dumps(declaration,indent=2)+'\n')
    start=time.perf_counter()
    monthly,penalties=timing_analysis(output,params)
    anchors,surfaces,means,comparisons=critic_analysis(output,params,plan,years)
    plots(output,monthly,penalties,anchors,surfaces,means)
    actor=means.query('candidate=="actor"')
    terminal=actor.query('day==365');nonterminal=actor.query('day!=365')
    dominated=anchors.query('better_same_transition_action_found')
    summary=dict(complete=True,test_evaluated=False,training_years=years,anchors=len(anchors),
        same_transition_dominated_actor_requests=len(dominated),
        dominated_requests_critics_prefer_worse=int((dominated.predicted_same_transition_q_gain<0).sum()),
        mean_dominated_immediate_reward_gain=float(dominated.same_transition_reward_gain.mean()) if len(dominated) else 0.,
        terminal_mean_q_error=float(terminal.conditional_q_error.mean()),
        terminal_mean_absolute_q_error=float(terminal.conditional_q_error.abs().mean()),
        nonterminal_conditional_mean_q_error=float(nonterminal.conditional_q_error.mean()),
        nonterminal_conditional_mean_absolute_q_error=float(nonterminal.conditional_q_error.abs().mean()),
        continuation_transitions=int(pd.read_csv(output/'critic_branches.csv').continuation_steps.sum()),
        total_seconds=time.perf_counter()-start,
        unchanged_sources=all(digest(PROJECT_ROOT/p)==h for p,h in frozen.items()),
        interpretation='Exact same-transition dominance is local evidence; historical-year MC errors also reflect hidden inflow/year information and small action-noise samples. No proof of global critic bias or algorithm inferiority.')
    assert summary['unchanged_sources']
    verify_plan(STUDY)
    (output/'run.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
