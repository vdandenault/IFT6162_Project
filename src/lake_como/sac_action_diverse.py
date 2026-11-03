"""Frozen-actor, action-diverse bootstrap versus conditional rollout supervision."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from lake_como import sac_critic_targets as old
from lake_como.sac import SACAgent, checked_optimizer_step
from lake_como.sac_critic_fit import freeze_noncritics, frozen_state, assert_frozen, training_partitions
from lake_como.sac_critic_sampling import point_predictions
from lake_como.sac_diagnostics import one_step, rollout_branch, same_transition
from lake_como.environment import make_env
from lake_como.inflow_observation import PastInflowObservation
from lake_como.mpc_hindsight import digest
from lake_como.paths import PROJECT_ROOT

TARGETS=PROJECT_ROOT/'results/sac/critic_targets_v1'
ROLES=('actor','lower_50','higher_50')


def diverse_batches(years,config,seed):
    states=old.target_batches(np.asarray(years),config,seed)
    roles=np.random.default_rng(82000+seed).integers(3,size=states.shape)
    return states*3,states*3+roles


def gap_metrics(prediction,returns,requests):
    q=np.asarray(prediction).reshape(-1,3)
    r=np.asarray(returns).reshape(-1,3,8)
    gap=r[:,1:]-r[:,:1];mean=gap.mean(axis=-1)
    error=q[:,1:]-q[:,:1]-mean
    distinct=np.asarray(requests).reshape(-1,3)[:,1:]!=np.asarray(requests).reshape(-1,3)[:,:1]
    informative=distinct&(np.abs(mean)>np.maximum(1e-4,2*gap.std(axis=-1,ddof=1)/np.sqrt(8)))
    return dict(gap_pairs=int(distinct.sum()),gap_mae=float(np.abs(error[distinct]).mean()),
        gap_rmse=float(np.sqrt(np.square(error[distinct]).mean())),informative_pairs=int(informative.sum()),
        sign_correct=int(((q[:,1:]-q[:,:1])*mean>0)[informative].sum()))


@torch.no_grad()
def bootstrap(agent,transition,seed):
    obs,reward,_,ended=transition
    if ended:return agent.config.reward_scale*reward
    scaled=agent.normalize(torch.from_numpy(obs).unsqueeze(0).repeat(16,1))
    a,lp=agent.actor(scaled,generator=torch.Generator().manual_seed(seed))
    value=torch.minimum(agent.target_q1(scaled,a),agent.target_q2(scaled,a))-agent.alpha*lp
    return agent.config.reward_scale*reward+agent.config.gamma*value.mean().item()


def collect(agent,seed,plan,partition,output):
    if partition=='fit':
        records=json.loads((TARGETS/f'seed_{seed}/fitting_anchors.json').read_text())
        replicates=pd.read_csv(TARGETS/f'seed_{seed}/fitting_rollout_replicates.csv',float_precision='round_trip')
    else:
        records=[r for r in json.loads((old.REFERENCES/'reference_anchors.json').read_text()) if r['seed']==seed]
        replicates=pd.read_csv(old.REFERENCES/'rollout_replicates.csv',float_precision='round_trip')
    cached=old.ordered_returns(records,replicates)
    lookup={(r['year'],r['day']):(i,r) for i,r in enumerate(records)}
    rows=[];returns=[];rep_rows=[];cost=0
    for year in sorted({r['year'] for r in records}):
        env=PastInflowObservation(make_env('train',year),plan['fallback']);obs,_=env.reset()
        try:
            for day in range(1,316):
                request=float(agent.physical_action(torch.from_numpy(agent.act(obs,deterministic=True)))[0])
                if (year,day) in lookup:
                    i,record=lookup[(year,day)]
                    np.testing.assert_array_equal(obs,np.asarray(record['observation'],np.float32))
                    assert request==record['request_m3_s']
                    actor_transition=one_step(env,request);cost+=1
                    actor_target=record.get('bootstrap_target',record.get('frozen_target'))
                    requests=[request,float(np.float32(max(0,request-50))),float(np.float32(min(float(agent.action_high[0]),request+50)))]
                    cache={request:cached[i]}
                    for role,action in zip(ROLES,requests):
                        transition=actor_transition if role=='actor' else one_step(env,action)
                        cost+=int(role!='actor')
                        same=same_transition(transition,actor_transition)
                        if action in cache:values=cache[action]
                        elif same:
                            values=cached[i]+agent.config.reward_scale*(transition[1]-actor_transition[1])
                        else:
                            values=[]
                            for rep in range(8):
                                result=rollout_branch(env,agent,action,seed*10000000+year*1000+day*10+rep)
                                cost+=result['continuation_steps'];values.append(result['soft_return'])
                            values=np.asarray(values)
                        cache[action]=values
                        target=(actor_target+agent.config.reward_scale*(transition[1]-actor_transition[1])) if same else bootstrap(agent,transition,seed*10000000+year*1000+day)
                        rows.append(dict(seed=seed,year=year,day=day,role=role,observation=obs.tolist(),request_m3_s=action,bootstrap_target=target,rollout_target=float(np.mean(values))))
                        returns.append(values)
                        for rep,value in enumerate(values):rep_rows.append(dict(seed=seed,year=year,day=day,role=role,replicate=rep,soft_return=float(value)))
                obs,_,term,trunc,_=env.step(np.array([request],np.float32));cost+=1
                assert not (term or trunc)
        finally:env.close()
    (output/f'{partition}_anchors.json').write_text(json.dumps(rows,indent=2)+'\n')
    pd.DataFrame(rep_rows).to_csv(output/f'{partition}_replicates.csv',index=False)
    return old.as_point_data(rows),np.asarray(returns),rows,cost


def score(agent,data,returns,seed,arm,steps,partition):
    with torch.no_grad():
        a,b=point_predictions(agent,data,np.arange(len(data['years'])));q=np.minimum(a.numpy(),b.numpy())
    mean=returns.mean(axis=1);error=q-mean
    return dict(seed=seed,arm=arm,steps=steps,partition=partition,mse=float(np.square(error).mean()),
        actor_mse=float(np.square(error[::3]).mean()),mae=float(np.abs(error).mean()),
        **gap_metrics(q,returns,data['requests'])),pd.DataFrame(dict(seed=seed,arm=arm,steps=steps,partition=partition,
            year=data['years'],day=data['days'],role=np.tile(ROLES,len(q)//3),request=data['requests'],min_q=q,rollout_mean=mean))


def run_seed(seed,root,plan):
    output=root/f'seed_{seed}';output.mkdir();config=old.TargetConfig()
    base=SACAgent.load(old.STUDY/f'seed_{seed}/final.pt');freeze_noncritics(base)
    frozen=frozen_state(base);rng=base.generator.get_state().clone()
    fit,fr,records,cost=collect(base,seed,plan,'fit',output)
    check,cr,_,ccost=collect(base,seed,plan,'check',output);cost+=ccost
    assert not np.isin(check['years'],fit['years']).any()
    old.check_fit_scope(fit['years'])
    cached=np.load(TARGETS/f'seed_{seed}/fit_states_targets.npz')
    for key in ['observations','requests','years','days']:np.testing.assert_array_equal(fit[key][::3],cached[key])
    targets={name:torch.tensor([r[name+'_target'] for r in records],dtype=torch.float32) for name in ['bootstrap','rollout']}
    for name,target in targets.items():np.testing.assert_array_equal(target.numpy()[::3],cached[name+'_targets'])
    np.savez_compressed(output/'fit_data.npz',**fit,**{k+'_targets':v.numpy() for k,v in targets.items()})
    single,multi=diverse_batches(fit['years'][::3],config,seed)
    summaries=[];predictions=[];trace=[]
    def evaluate(agent,arm,steps):
        for partition,data,ret in [('fit',fit,fr),('check',check,cr)]:
            metrics,rows=score(agent,data,ret,seed,arm,steps,partition);summaries.append(metrics);predictions.append(rows)
        assert_frozen(agent,frozen,rng)
    evaluate(base,'baseline',0)
    for name,target in targets.items():
        for budget in config.budgets:
            agent=deepcopy(base)
            saved=torch.load(TARGETS/f'seed_{seed}/{name}_{budget}_critics.pt',weights_only=True)
            agent.q1.load_state_dict(saved['q1']);agent.q2.load_state_dict(saved['q2'])
            evaluate(agent,'single_'+name,budget)
        agent=deepcopy(base);params=list(agent.q1.parameters())+list(agent.q2.parameters())
        optimizer=torch.optim.Adam(params,lr=config.learning_rate)
        for step,indices in enumerate(multi,1):
            q1,q2=point_predictions(agent,fit,indices)
            loss=F.mse_loss(q1,target[indices])+F.mse_loss(q2,target[indices])
            grad=checked_optimizer_step(loss,optimizer,params)
            if step%100==0:trace.append(dict(arm='multi_'+name,step=step,loss=loss.item(),gradient_norm=grad))
            if step in config.budgets:
                evaluate(agent,'multi_'+name,step)
                torch.save(dict(q1=agent.q1.state_dict(),q2=agent.q2.state_dict()),output/f'multi_{name}_{step}_critics.pt')
    pd.DataFrame(summaries).to_csv(output/'metrics.csv',index=False)
    pd.concat(predictions).to_csv(output/'predictions.csv',index=False)
    pd.DataFrame(trace).to_csv(output/'fitting_trace.csv',index=False)
    (output/'run.json').write_text(json.dumps(dict(complete=True,new_simulator_transitions=cost,actor_frozen=True,checking_excluded_from_gradients=True)))
    print('Completed seed',seed,'new transitions',cost,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=PROJECT_ROOT/'results/sac/action_diverse_v2');args=p.parse_args();root=args.output
    if root.exists() and any(root.iterdir()):raise ValueError('New empty output required')
    root.mkdir(parents=True);torch.set_num_threads(1);plan=old.verify_plan(old.STUDY)
    sources=[Path(__file__),TARGETS/'plan.json',old.REFERENCES/'reference_anchors.json',old.REFERENCES/'rollout_replicates.csv']
    sources+=list((PROJECT_ROOT/'src/lake_como').glob('sac*.py'))
    for seed in plan['config']['learner_seeds']:
        sources+=[old.STUDY/f'seed_{seed}/final.pt',*sorted((TARGETS/f'seed_{seed}').glob('*'))]
    hashes={str(f.relative_to(PROJECT_ROOT)):digest(f) for f in sources if f.is_file()}
    declaration=dict(config=asdict(old.TargetConfig()),roles=ROLES,first_request_offsets_m3_s=[0,-50,50],
        fitting_years=training_partitions()[0],checking_years=training_partitions()[1],
        design='Same state minibatches and 128 labels/update. Multi independently samples one of three first requests per state. Single-action controls reused exactly from prior target study. Fixed actor, alpha, target critics and observation transform.',
        targets='Eight common-noise stochastic continuations to annual end; actor-return labels reused. Exact same-next-state alternatives reuse continuation and change only immediate reward. Frozen bootstrap labels use 16 next actions; actor labels reused, so finite sampling noise differs for changed-next-state alternatives.',
        primary_endpoint=1000,secondary_endpoint=100,metrics='All-action and actor-only checking conditional-return MSE; gap MAE/RMSE and signs where paired gap exceeds max(1e-4,2 paired standard errors). Threshold is descriptive, not a significance test.',
        limits='Historical-future conditional values, not population values. Checking years were original SAC training years and used in prior diagnostics. No actor update or controller improvement claim; no validation or test evaluation.',source_sha256=hashes)
    (root/'plan.json').write_text(json.dumps(declaration,indent=2)+'\n')
    for seed in plan['config']['learner_seeds']:run_seed(seed,root,plan)
    metrics=pd.concat([pd.read_csv(root/f'seed_{s}/metrics.csv') for s in plan['config']['learner_seeds']]);metrics.to_csv(root/'metrics.csv',index=False)
    summary=metrics.groupby(['arm','steps','partition']).agg(mse=('mse','mean'),actor_mse=('actor_mse','mean'),gap_mae=('gap_mae','mean'),gap_rmse=('gap_rmse','mean'),informative_pairs=('informative_pairs','sum'),sign_correct=('sign_correct','sum')).reset_index()
    summary['sign_accuracy']=summary.sign_correct/summary.informative_pairs;summary.to_csv(root/'summary.csv',index=False)
    for path,sha in hashes.items():assert digest(PROJECT_ROOT/path)==sha,path
    (root/'run.json').write_text(json.dumps(dict(complete=True,seeds=plan['config']['learner_seeds'],new_critic_updates=10000,reused_single_control_updates=10000,source_hashes_verified=True,actor_frozen=True,test_evaluated=False),indent=2)+'\n')
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':main()
