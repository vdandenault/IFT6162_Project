"""Final frozen-actor TD diagnostic: frozen versus slowly updated target critics."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from lake_como.paths import PROJECT_ROOT
from lake_como.mpc_hindsight import digest
from lake_como.sac import SACAgent,checked_optimizer_step,polyak_update,soft_bellman_target
from lake_como.sac_critic_fit import freeze_noncritics,training_partitions,score_pairs
from lake_como.sac_critic_sampling import point_predictions
from lake_como.sac_critic_targets import as_point_data,ordered_returns,evaluate_points
from lake_como.sac_study import verify_plan

BASE=PROJECT_ROOT/'results/sac/baseline_200k_v1'
POOL=PROJECT_ROOT/'results/sac/critic_sampling_v1'
REF=PROJECT_ROOT/'results/sac/critic_reference_v1'
FIT=PROJECT_ROOT/'results/sac/critic_targets_v1'
PAIR=PROJECT_ROOT/'results/sac/critic_fit_v1'


@torch.no_grad()
def td_target(agent,data,indices,generator):
    obs=agent.normalize(torch.from_numpy(data['next_observations'][indices]))
    action,logp=agent.actor(obs,generator=generator)
    q=torch.minimum(agent.target_q1(obs,action),agent.target_q2(obs,action))
    return soft_bellman_target(torch.tensor(data['rewards'][indices],dtype=torch.float32)[:,None],
        torch.from_numpy(data['ends'][indices])[:,None],q,logp,agent.alpha,
        agent.config.gamma,agent.config.reward_scale).squeeze(1)


def policy_state(agent):
    return {k:v.clone() for k,v in agent.state_dict().items()
            if not k.startswith(('q1.','q2.','target_q1.','target_q2.'))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'results/sac/iterative_td_v1')
    args=parser.parse_args();root=args.output
    if root.exists() and any(root.iterdir()):
        parser.error('Choose a new output directory')
    root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    plan=verify_plan(BASE);fit,check=training_partitions()
    files=[Path(__file__),PROJECT_ROOT/'src/lake_como/sac_critic_targets.py',REF/'reference_anchors.json',REF/'rollout_replicates.csv']
    for seed in plan['config']['learner_seeds']:
        files += [BASE/f'seed_{seed}/final.pt',POOL/f'seed_{seed}/common_pool.npz',
                  FIT/f'seed_{seed}/fitting_anchors.json',FIT/f'seed_{seed}/fitting_rollout_replicates.csv',
                  PAIR/f'seed_{seed}/pairs.npz',PAIR/f'seed_{seed}/pairs.csv',POOL/f'seed_{seed}/fixed_targets.npy']
    hashes={str(p.relative_to(PROJECT_ROOT)):digest(p) for p in files}
    declaration=dict(budgets=[100,1000],primary_endpoint=1000,batch_size=128,learning_rate=.0003,
        fit_years=fit,checking_years=check,arms={'frozen':0.,'moving':.005},
        scope='Uniform full fitting pool, one fresh next-action sample each update in both arms; matching batches/noise; actor/alpha/normalizer fixed. Only target Polyak updates differ. No convergence or controller-improvement claim.',
        source_sha256=hashes,validation_evaluated=False,test_evaluated=False)
    (root/'plan.json').write_text(json.dumps(declaration,indent=2)+'\n')
    checking=json.loads((REF/'reference_anchors.json').read_text())
    checking_returns=pd.read_csv(REF/'rollout_replicates.csv',float_precision='round_trip')
    metrics=[];pairscores=[];traces=[];started=time.perf_counter()
    for seed in plan['config']['learner_seeds']:
        out=root/f'seed_{seed}';out.mkdir()
        data={k:v.copy() for k,v in np.load(POOL/f'seed_{seed}/common_pool.npz').items()}
        eligible=np.flatnonzero(np.isin(data['years'],fit))
        batches=np.random.default_rng(91000+seed).choice(eligible,size=(1000,128),replace=True)
        assert np.isin(data['years'][batches],fit).all()
        records=json.loads((FIT/f'seed_{seed}/fitting_anchors.json').read_text())
        fit_returns=pd.read_csv(FIT/f'seed_{seed}/fitting_rollout_replicates.csv',float_precision='round_trip')
        check_records=[r for r in checking if r['seed']==seed]
        evaluations=[('fit',as_point_data(records),ordered_returns(records,fit_returns),records),
                     ('check',as_point_data(check_records),ordered_returns(check_records,checking_returns),check_records)]
        paired={k:v.copy() for k,v in np.load(PAIR/f'seed_{seed}/pairs.npz').items()}
        rows=pd.read_csv(PAIR/f'seed_{seed}/pairs.csv',float_precision='round_trip')
        targets=torch.from_numpy(np.load(POOL/f'seed_{seed}/fixed_targets.npy')[data['pair_indices']])
        base=SACAgent.load(BASE/f'seed_{seed}/final.pt');freeze_noncritics(base)
        immutable=policy_state(base);rng=base.generator.get_state().clone()
        for partition,points,returns,meta in evaluations:
            record,_=evaluate_points(base,points,returns,meta,'baseline',0,partition);metrics.append(record)
        pairscores.append(score_pairs(base,paired,rows,targets,'baseline',0))
        for arm,tau in declaration['arms'].items():
            model=deepcopy(base);generator=torch.Generator().manual_seed(92000+seed)
            optimizer=torch.optim.Adam(list(model.q1.parameters())+list(model.q2.parameters()),lr=.0003)
            original_targets={k:v.clone() for k,v in model.state_dict().items() if k.startswith('target_q')}
            for step,indices in enumerate(batches,1):
                y=td_target(model,data,indices,generator)
                q1,q2=point_predictions(model,data,indices)
                loss=F.mse_loss(q1,y)+F.mse_loss(q2,y)
                grad=checked_optimizer_step(loss,optimizer,list(model.q1.parameters())+list(model.q2.parameters()))
                polyak_update(model.q1,model.target_q1,tau);polyak_update(model.q2,model.target_q2,tau)
                if step%100==0:
                    traces.append(dict(seed=seed,arm=arm,step=step,loss=loss.item(),gradient_norm=grad,target_mean=y.mean().item()))
                if step in [100,1000]:
                    for partition,points,returns,meta in evaluations:
                        record,_=evaluate_points(model,points,returns,meta,arm,step,partition);metrics.append(record)
                    pairscores.append(score_pairs(model,paired,rows,targets,arm,step))
                    torch.save(dict(q1=model.q1.state_dict(),q2=model.q2.state_dict(),target_q1=model.target_q1.state_dict(),
                        target_q2=model.target_q2.state_dict(),arm=arm,seed=seed,steps=step),out/f'{arm}_{step}.pt')
            assert all(torch.equal(v,model.state_dict()[k]) for k,v in immutable.items())
            assert torch.equal(rng,model.generator.get_state())
            changed=any(not torch.equal(v,model.state_dict()[k]) for k,v in original_targets.items())
            assert changed==(tau>0)
            assert all(p.grad is None for p in model.actor.parameters())
            print('TD complete',seed,arm,flush=True)
    metrics=pd.DataFrame(metrics);pairscores=pd.concat(pairscores,ignore_index=True)
    metrics.to_csv(root/'metrics_by_seed.csv',index=False);pairscores.to_csv(root/'pair_scores.csv',index=False)
    pd.DataFrame(traces).to_csv(root/'trace.csv',index=False)
    summary=metrics.groupby(['arm','steps','partition'])[['mse','mae','bias']].mean().reset_index();summary.to_csv(root/'summary.csv',index=False)
    pair_summary=pairscores.groupby(['arm','fit_step','partition']).min_q_correct_ranking.mean().reset_index()
    pair_summary.to_csv(root/'pair_summary.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for arm,color in [('frozen','#267ead'),('moving','#27856a')]:
        frame=summary.query('arm==@arm and partition=="check"')
        original=summary.query('arm=="baseline" and partition=="check"').iloc[0]
        axes[0].plot([0,*frame.steps],[original.mse,*frame.mse],marker='o',label=arm,color=color)
        p=pair_summary.query('arm==@arm and partition=="check"')
        orig=pair_summary.query('arm=="baseline" and partition=="check"').iloc[0]
        axes[1].plot([0,*p.fit_step],np.r_[orig.min_q_correct_ranking,p.min_q_correct_ranking]*100,marker='o',color=color)
    axes[0].set(title='Checking conditional-return MSE',xlabel='Critic updates');axes[0].legend(title='Target networks')
    axes[1].set(title='Checking clipping-pair ranking (%)',xlabel='Critic updates',ylim=(0,100))
    fig.suptitle('Frozen actor: does iterative TD resolve the fixed-target limitation?')
    fig.savefig(root/'iterative_td.png',dpi=160);plt.close(fig)
    assert all(digest(PROJECT_ROOT/p)==v for p,v in hashes.items());verify_plan(BASE)
    (root/'run.json').write_text(json.dumps(dict(complete=True,actor_alpha_normalizer_unchanged=True,
        target_update_intervention_verified=True,source_hashes_unchanged=True,optimizer_steps=10000,
        new_simulator_steps=0,seconds=time.perf_counter()-started,validation_evaluated=False,test_evaluated=False),indent=2)+'\n')
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':
    main()
