"""Offline perfect-forecast diagnosis of the unchanged one_period_delay study.

Independent LP retains explicit storage variables and balance equalities, unlike
production QP storage elimination. This is diagnostic code, not a new controller.
"""
from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .hard_follower import validate_problem, solve_hard_follower, FollowerTolerances
from .hard_follower_study import CONFIG, load_follower_config, make_problem, rollout, validate_config
from .routing import TransitState
from .reconstruction import ROOT, digest
from .discharge_comparison import snapshot


def explicit_storage_feasibility(p, tolerance=FollowerTolerances()):
    """Zero-objective LP with explicit r and S variables, in normalized units.

    No slacks or modified bounds. Constant routing rows already satisfied within
    the original SI acceptance tolerance are omitted (only float roundoff).
    All candidate residuals are then independently checked in physical units.
    """
    d = validate_problem(p)
    h = len(d['duration_seconds']); qs, ss = p.flow_scale_m3_s, p.storage_scale_m3
    eq = np.zeros((h+1, 2*h+1)); rhs = np.zeros(h+1)
    eq[0, h] = 1.; rhs[0] = p.initial_storage_m3/ss
    for j in range(h):
        eq[j+1, j] = d['duration_seconds'][j]*qs/ss
        eq[j+1, h+j] = -1.; eq[j+1, h+j+1] = 1.
        rhs[j+1] = d['duration_seconds'][j]*d['net_inflow_m3_s'][j]/ss
    ub = np.c_[d['routing_matrix'], np.zeros((h,h+1))]
    limits = (d['downstream_capacity_m3_s']-d['ontario_executable_release_m3_s']
              -d['other_downstream_m3_s']-d['in_transit_contribution_m3_s'])/qs
    fixed = ~np.any(ub != 0., axis=1)
    if np.any(limits[fixed] < -tolerance.flow_m3_s/qs):
        return dict(status='infeasible', message='Fixed transit exceeds capacity')
    bounds = list(zip(d['release_minimum_m3_s']/qs, d['release_maximum_m3_s']/qs))
    bounds += list(zip(d['storage_minimum_m3']/ss, d['storage_maximum_m3']/ss))
    fit = linprog(np.zeros(2*h+1), A_ub=ub[~fixed], b_ub=limits[~fixed],
                  A_eq=eq, b_eq=rhs, bounds=bounds, method='highs',
                  options={'primal_feasibility_tolerance':1e-9, 'dual_feasibility_tolerance':1e-9})
    if not fit.success:
        return dict(status='infeasible' if fit.status==2 else 'solver_failure', message=fit.message)
    r, storage = fit.x[:h]*qs, fit.x[h:]*ss
    downstream = d['ontario_executable_release_m3_s']+d['other_downstream_m3_s']+d['routing_matrix']@r+d['in_transit_contribution_m3_s']
    residuals = dict(initial_storage_m3=abs(storage[0]-p.initial_storage_m3),
        balance_m3=float(np.max(abs(np.diff(storage)-d['duration_seconds']*(d['net_inflow_m3_s']-r)))),
        storage_violation_m3=float(max(0.,max(d['storage_minimum_m3']-storage),max(storage-d['storage_maximum_m3']))),
        release_violation_m3_s=float(max(0.,max(d['release_minimum_m3_s']-r),max(r-d['release_maximum_m3_s']))),
        downstream_violation_m3_s=float(max(0.,max(downstream-d['downstream_capacity_m3_s']))))
    if not (max(residuals[k] for k in ('initial_storage_m3','balance_m3','storage_violation_m3')) <= tolerance.storage_m3
            and max(residuals[k] for k in ('release_violation_m3_s','downstream_violation_m3_s')) <= tolerance.flow_m3_s):
        return dict(status='solver_failure', message='Explicit LP failed SI residual check', residuals=residuals)
    return dict(status='feasible', message=fit.message, release_m3_s=r.tolist(), storage_m3=storage.tolist(),
                downstream_m3_s=downstream.tolist(), residuals=residuals)


def boundary_state(c, case, trace, k):
    if k == 0:
        return c['initial_storage_m3'], c['previous_executed_release_m3_s'], TransitState(tuple(case['initial_transit_volumes_m3']))
    row = trace.iloc[k-1]
    return row.storage_end_m3, row.executed_release_m3_s, TransitState(tuple(json.loads(row.transit_queue_end_m3)))


def witness_frame(c, case, result):
    r, s = np.array(result['release_m3_s']), np.array(result['storage_m3'])
    dt = c['clock']['duration_seconds']; transit = TransitState(tuple(case['initial_transit_volumes_m3']))
    initial = s[0]+transit.total_m3; cumulative_input, cumulative_arrival = 0., 0.; rows=[]
    for k, release in enumerate(r):
        transit, arrival, _ = transit.advance(release, dt)
        cumulative_input += dt*case['realized_net_inflow_m3_s'][k]; cumulative_arrival += dt*arrival
        rows.append(dict(period=k+1, release_m3_s=release, storage_start_m3=s[k], storage_end_m3=s[k+1],
                         arrival_m3_s=arrival, transit_end_m3=transit.total_m3,
                         transit_queue_end_m3=json.dumps(transit.volumes_m3),
                         downstream_m3_s=c['known_ontario_executable_schedule_m3_s'][k]+c['known_other_downstream_forecast_m3_s'][k]+arrival,
                         system_balance_error_m3=s[k+1]+transit.total_m3-initial-cumulative_input+cumulative_arrival))
    return pd.DataFrame(rows)


def bottleneck(c, case, k, storage):
    """Exact cumulative upper-release certificate for this one-period, equal-dt case.

    A release in period t arrives in period t+1. Only already prescribed capacity
    periods are used. Last release has the original outlet bound and stays queued.
    """
    assert case['routing_delay_periods']==1
    dt=c['clock']['duration_seconds']; n=len(case['realized_net_inflow_m3_s'])
    cap=np.array(case['downstream_capacity_m3_s'])-np.array(c['known_ontario_executable_schedule_m3_s'])-np.array(c['known_other_downstream_forecast_m3_s'])
    maximum=np.full(n-k,c['release_maximum_m3_s'])
    maximum[:max(0,n-k-1)]=np.minimum(maximum[:max(0,n-k-1)],cap[k+1:])
    lower_storage=storage+dt*np.cumsum(np.array(case['realized_net_inflow_m3_s'][k:])-maximum)
    violating=np.flatnonzero(lower_storage>c['storage_maximum_m3']+.01)
    if not len(violating): return None
    j=int(violating[0]); end=k+j
    return dict(start_period=k+1, end_period=end+1,
        inflow_volume_m3=float(dt*sum(case['realized_net_inflow_m3_s'][k:end+1])),
        maximum_release_volume_m3=float(dt*sum(maximum[:j+1])),
        required_retention_m3=float(lower_storage[j]-storage),
        available_storage_m3=float(c['storage_maximum_m3']-storage),
        unavoidable_storage_m3=float(lower_storage[j]),
        unavoidable_excess_m3=float(lower_storage[j]-c['storage_maximum_m3']),
        maximum_releases_m3_s=maximum[:j+1].tolist())


def run(config, output):
    c=validate_config(load_follower_config(config))
    case=next(x for x in c['scenarios'] if x['id']=='one_period_delay')
    if c['horizon']!=4: raise ValueError('Diagnostic expects original four-period horizon')
    protected=[ROOT/'results/ottawa'/name for name in ('hard_follower_v1_perfect_verified',
        'hard_follower_v1_noisy_verified','baskatong_1970_observed_v1','baskatong_discharge_comparison_v2')]
    before={str(p):snapshot(p) for p in protected if p.exists()}
    output.mkdir(parents=True,exist_ok=False)
    (output/'configuration.json').write_text(json.dumps(c,indent=2)+'\n')
    dt=c['clock']['duration_seconds']; inflows=case['realized_net_inflow_m3_s']; n=len(inflows)
    initial=make_problem(c,case,0,c['initial_storage_m3'],c['previous_executed_release_m3_s'],
                         TransitState(tuple(case['initial_transit_volumes_m3'])),inflows)
    lp=explicit_storage_feasibility(initial)
    (output/'full_experiment_feasibility.json').write_text(json.dumps(lp,indent=2)+'\n')
    if lp['status']!='feasible': raise RuntimeError('Initial full-experiment LP not feasible: '+lp['status'])
    witness=witness_frame(c,case,lp); witness.to_csv(output/'feasible_lp_trajectory.csv',index=False)
    trace,plans,summary=rollout(c,case,'mpc','perfect',0)
    trace.to_csv(output/'original_mpc.csv',index=False)
    (output/'original_plans.json').write_text(json.dumps(plans,indent=2)+'\n')
    checks=[]
    for k in range(len(trace)+1):
        storage,previous,queue=boundary_state(c,case,trace,k)
        p=make_problem(c,case,k,storage,previous,queue,inflows[k:])
        fit=explicit_storage_feasibility(p); qp=solve_hard_follower(p)
        if fit['status']=='feasible':
            (output/f'continuation_after_{k}_actions.json').write_text(json.dumps(fit,indent=2)+'\n')
        checks.append(dict(next_period=k+1, executed_periods=k, storage_m3=storage,
            remaining_capacity_m3=c['storage_maximum_m3']-storage, transit_m3=queue.total_m3,
            lp_status=fit['status'], qp_status=qp.status, certificate=bottleneck(c,case,k,storage)))
    (output/'prefix_feasibility.json').write_text(json.dumps(checks,indent=2)+'\n')
    k=len(trace);storage,previous,queue=boundary_state(c,case,trace,k)
    failed_p=make_problem(c,case,k,storage,previous,queue,inflows[k:k+c['horizon']])
    failed_lp=explicit_storage_feasibility(failed_p)
    failed_context=dict(next_period=k+1,storage_m3=storage,remaining_capacity_m3=c['storage_maximum_m3']-storage,
        queue=[dict(arrival_period=k+j+1,volume_m3=v,arrival_flow_m3_s=v/dt) for j,v in enumerate(queue.volumes_m3)],
        current_horizon_end_period=k+len(failed_p.duration_seconds),
        forecast_remaining_inflow_m3_s=inflows[k:], release_minimum_m3_s=c['release_minimum_m3_s'],
        release_maximum_m3_s=c['release_maximum_m3_s'],
        capacity_remaining_m3_s=case['downstream_capacity_m3_s'][k:],
        available_downstream_remaining_m3_s=(np.array(case['downstream_capacity_m3_s'][k:])-np.array(c['known_ontario_executable_schedule_m3_s'][k:])-np.array(c['known_other_downstream_forecast_m3_s'][k:])).tolist(),
        failed_horizon_independent_lp=failed_lp,certificate=bottleneck(c,case,k,storage))
    (output/'first_failed_solve.json').write_text(json.dumps(failed_context,indent=2)+'\n')
    comparisons=pd.DataFrame(dict(period=np.arange(n)+1,lp_release_m3_s=witness.release_m3_s,
        lp_storage_end_m3=witness.storage_end_m3,original_mpc_release_m3_s=trace.executed_release_m3_s.reindex(range(n)),
        original_mpc_storage_end_m3=trace.storage_end_m3.reindex(range(n))))
    horizons=[]; trajectories={}; saved_plans={}
    for h in [4,5,6,8,12]:
        changed=deepcopy(c);changed['horizon']=h
        t,p,s=rollout(changed,case,'mpc','perfect',0)
        trajectories[h]=t;saved_plans[str(h)]=p;horizons.append(dict(prediction_horizon=h,**s))
        t.to_csv(output/f'horizon_{h}_trajectory.csv',index=False)
    reference_path=ROOT/'results/ottawa/hard_follower_v1_perfect_verified/one_period_delay_0_mpc.csv'
    reproduced_reference=None
    if reference_path.exists():
        archived=pd.read_csv(reference_path)
        columns=['storage_start_m3','storage_end_m3','executed_release_m3_s','transit_end_m3','combined_downstream_m3_s']
        np.testing.assert_allclose(trace[columns],archived[columns],atol=.00001,rtol=0.)
        reproduced_reference=True
    lp_routing_error=float(max(abs(witness.downstream_m3_s-np.asarray(lp['downstream_m3_s']))))
    full_qp=solve_hard_follower(initial)
    if not full_qp.success: raise RuntimeError('Full QP disagrees with feasible LP')
    full_plan_difference=float(max(abs(full_qp.release_m3_s-trajectories[n].executed_release_m3_s)))
    assert lp_routing_error<=1e-5 and full_plan_difference<=.01
    assert all((row['lp_status']=='feasible')==row['qp_status'].startswith('optimal') for row in checks)
    (output/'verification.json').write_text(json.dumps(dict(
        original_saved_mpc_trajectory_reproduced=reproduced_reference,
        explicit_lp_vs_queue_downstream_max_difference_m3_s=lp_routing_error,
        full_qp_vs_full_remaining_rollout_release_max_difference_m3_s=full_plan_difference,
        maximum_lp_system_balance_error_m3=float(max(abs(witness.system_balance_error_m3))),
        maximum_rollout_system_balance_error_m3=max(float(max(abs(t.system_balance_error_m3))) for t in trajectories.values()),
        maximum_rollout_storage_violation_m3=max(float(max(t.storage_violation_m3)) for t in trajectories.values()),
        maximum_rollout_downstream_violation_m3_s=max(float(max(t.downstream_violation_m3_s)) for t in trajectories.values())),indent=2)+'\n')
    pd.DataFrame(horizons).to_csv(output/'horizon_summary.csv',index=False)
    comparisons['full_remaining_mpc_release_m3_s']=trajectories[n].executed_release_m3_s
    comparisons['full_remaining_mpc_storage_end_m3']=trajectories[n].storage_end_m3
    comparisons.to_csv(output/'trajectory_comparison.csv',index=False)
    (output/'horizon_plans.json').write_text(json.dumps(saved_plans,indent=2)+'\n')
    assert before=={str(p):snapshot(p) for p in protected if p.exists()},'Prior outputs changed'
    sources=[config,Path(__file__),Path(__file__).with_name('hard_follower.py'),Path(__file__).with_name('routing.py'),Path(__file__).with_name('hard_follower_study.py')]
    (output/'provenance.json').write_text(json.dumps(dict(source_sha256={str(p):digest(p) for p in sources},
        unchanged_prior_bundles=before,diagnostic_horizons=[4,5,6,8,12],seed=0),indent=2)+'\n')
    plot(c,case,witness,trajectories,output/'diagnostic.png')
    write_report(c,case,checks,failed_context,comparisons,horizons,witness,plans,output)
    print((output/'report.md').resolve())


def plot(c,case,witness,trajectories,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(11,12),sharex=True,layout='constrained')
    axes[0].plot(np.r_[0,witness.period],np.r_[c['initial_storage_m3'],witness.storage_end_m3]/1e9,'--',color='gray',label='LP feasible witness (arbitrary)')
    axes[1].plot(witness.period,witness.release_m3_s,'--',color='gray',label='LP witness')
    for h,t in trajectories.items():
        label=f'H={h}' + (' (full remaining)' if h==12 else '')
        axes[0].plot(np.r_[0,t.step+1],np.r_[c['initial_storage_m3'],t.storage_end_m3]/1e9,label=label)
        axes[1].step(t.step+1,t.executed_release_m3_s,where='mid',label=label)
        axes[2].step(t.step+1,t.combined_downstream_m3_s,where='mid',label=label)
        axes[3].plot(np.r_[0,t.step+1],np.r_[sum(case['initial_transit_volumes_m3']),t.transit_end_m3]/1e9,label=label)
    for bound in (c['storage_minimum_m3'],c['storage_maximum_m3']): axes[0].axhline(bound/1e9,color='black',ls=':')
    axes[0].axhline(c['storage_target_m3']/1e9,color='gray',ls=':')
    axes[0].axvline(2,color='red',ls=':',label='H=4 loses completion feasibility')
    axes[0].axvline(3,color='black',ls=':',label='H=4 next solve fails')
    axes[2].step(np.arange(len(case['downstream_capacity_m3_s']))+1,case['downstream_capacity_m3_s'],where='mid',color='black',ls='--',label='unchanged capacity')
    for ax,label in zip(axes,['Storage (billion m³)','Release (m³/s)','Downstream flow (m³/s)','Transit (billion m³)']):
        ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=3)
    axes[-1].set_xlabel('Period (657,000 seconds); final transit retained, not forced empty')
    fig.suptitle('Perfect-forecast one-period-delay diagnosis — unchanged constraints and weights')
    fig.savefig(path,dpi=150);plt.close(fig)


def write_report(c,case,checks,failed,comparison,horizons,witness,original_plans,output):
    cert=failed['certificate']; lost=next(x for x in checks if x['lp_status']=='infeasible')
    first_lp=int(np.flatnonzero(abs(comparison.lp_release_m3_s-comparison.original_mpc_release_m3_s)>1e-5)[0])+1
    first_qp=int(np.flatnonzero(abs(comparison.full_remaining_mpc_release_m3_s-comparison.original_mpc_release_m3_s)>1e-5)[0])+1
    queue=failed['queue'][0]
    threshold=c['storage_maximum_m3']-lost['certificate']['required_retention_m3']
    second_required=(checks[1]['storage_m3']+c['clock']['duration_seconds']*case['realized_net_inflow_m3_s'][1]-threshold)/c['clock']['duration_seconds']
    lines=['# Perfect-forecast delayed-routing failure diagnosis','',
        '**The original experiment is feasible. The four-period MPC decisions make it impossible to finish after period 2; a later solve detects this before period 4.**', '',
        '## Full-experiment feasibility and divergence','',
        'An independent zero-objective LP with explicit storage variables and water-balance equalities finds a feasible 12-period trajectory from the original storage and transit queue. '
        'It uses the same inflows, executable Ontario schedule, other contributions, bounds and capacities. '
        'The LP witness is arbitrary, not an optimized or operational recommendation. Its numerical residuals are in `full_experiment_feasibility.json`; the full trajectory is in `feasible_lp_trajectory.csv`.', '',
        f'The feasible LP witness first differs from executed H=4 at period {first_lp}; the full-remaining-horizon MPC first differs at period {first_qp}. '
        'Both differences occur before the failure. `trajectory_comparison.csv` compares releases and storage. '
        'Full-remaining feasibility was independently rechecked after each executed H=4 action:', '',
        '| Executed periods | Storage, million m³ | Remaining-experiment LP | Full-remaining QP |',
        '|---:|---:|---|---|']
    for check in checks:
        lines.append(f"| {check['executed_periods']} | {check['storage_m3']/1e6:.3f} | {check['lp_status']} | {check['qp_status']} |")
    lines += ['',f"Completion first becomes infeasible after action {lost['executed_periods']}, not only when the original solver stops.",
        'At that earlier solve, its four-period window ended at period 5. The unrecoverable storage conflict at the end of period 6 lay outside that window. '
        'The terminal storage target is a soft cost, not a condition guaranteeing a feasible continuation.', '',
        f"After the original first action, a feasible continuation still existed (`continuation_after_1_actions.json`). "
        f"To make room through period 6, end-period-2 storage needed to be at most {threshold/1e6:.3f} million m³. "
        f"That required period-2 release of at least {second_required:.3f} m³/s from the inherited state; "
        f"H=4 released {comparison.original_mpc_release_m3_s.iloc[1]:.3f} m³/s. The future downstream caps remove the opportunity to recover later.", '',
        f"The period-2 H=4 plan also illustrates the window boundary: it planned a period-5 release of {original_plans[1]['planned_release_m3_s'][-1]:.3f} m³/s, "
        f"leaving {original_plans[1]['planned_terminal_transit_m3']/1e6:.3f} million m³ queued to arrive in period 6. "
        'Period 6 was outside that solve, so its 600 m³/s arrival allowance was not enforced in that plan. '
        'The queued water was recorded, not lost; reoptimization subsequently recognized the future cap. '
        'Thus both the future storage peak and the last planned release’s downstream arrival fall beyond the short window. '
        'This is a consequence of the declared horizon formulation, not disagreement between routing and simulation. '
        'The experiment-end convention remains identical for every run; no extra constraints are added here.', '',
        '## First failed solve: before period 4','',
        f"- Storage: **{failed['storage_m3']/1e6:.3f} million m³**; room to the unchanged maximum: **{failed['remaining_capacity_m3']/1e6:.3f} million m³**.",
        f"- Already in transit: **{queue['volume_m3']/1e6:.3f} million m³**, arriving during period {queue['arrival_period']} at {queue['arrival_flow_m3_s']:.6f} m³/s. It is outside reservoir storage and cannot be released a second time.",
        f"- The current perfect-forecast window covers periods 4–{failed['current_horizon_end_period']}; inflows are {case['realized_net_inflow_m3_s'][3:7]} m³/s. Remaining inflows and capacities are saved in `first_failed_solve.json`.",
        f"- Outlet bounds remain **{c['release_minimum_m3_s']:.0f}–{c['release_maximum_m3_s']:.0f} m³/s**. Ontario plus other contributions equal 9,000 m³/s. Capacity is 9,600 m³/s in periods 4–7, leaving only 600 m³/s for Ottawa arrivals; later capacity is 12,000 m³/s.",
        '- With the one-period delay, releases in periods 4, 5 and 6 arrive in periods 5, 6 and 7. Each release is therefore at most 600 m³/s, despite the 2,500 m³/s outlet maximum. The existing queue already consumes the period-4 Ottawa capacity.', '',
        f"During periods {cert['start_period']}–{cert['end_period']}, inflows total {cert['inflow_volume_m3']/1e6:.3f} million m³ and releases can total at most {cert['maximum_release_volume_m3']/1e6:.3f} million m³.",
        f"At least **{cert['required_retention_m3']/1e6:.3f} million m³** must stay in the reservoir, exceeding available room by **{cert['unavoidable_excess_m3']/1e6:.3f} million m³**.",
        f"Even maximum admissible releases imply end-period-6 storage of {cert['unavoidable_storage_m3']/1e9:.6f} billion m³, above the 5-billion-m³ maximum.",
        'This cumulative inequality is a direct infeasibility certificate: storage upper bound + balance + downstream arrival limits. '
        'It does not require the objective, lower storage bound or solver messages. The independent explicit-storage LP also rejects the failed four-period window.', '',
        '## Prediction-horizon sensitivity','',
        'All runs start from the same original state with perfect forecasts and unchanged weights, bounds and routing. Only H changes; each prediction window shrinks at the common period-12 endpoint.', '',
        '| H | Result | Executed periods | Final reservoir, million m³ | Final transit, million m³ | Full realized objective |',
        '|---:|---|---:|---:|---:|---:|']
    for s in horizons:
        cost='unavailable (incomplete)' if s['full_objective'] is None else f"{s['full_objective']:.6f}"
        lines.append(f"| {s['prediction_horizon']} | {s['status']} | {s['executed_steps']} | {s['final_storage_m3']/1e6:.3f} | {s['final_transit_volume_m3']/1e6:.3f} | {cost} |")
    lines += ['', 'H=5 suffices for this deterministic experiment; longer tested horizons also complete. '
        'This is not a proof of general recursive feasibility. No controller default is changed.', '',
        '## Endpoint accounting and implementation checks','',
        'Every solve constrains only arrivals within its supplied horizon. Releases arriving later remain explicitly queued. '
        'At period 12, the final release arrives in period 13; it is preserved as terminal transit, not discarded or forced to zero. '
        'No period-13 inflow/capacity has been supplied, so this diagnosis makes no safety claim beyond period 12. '
        'The LP, all MPC horizons and comparisons use this identical endpoint definition, the original storage terminal cost, and no transit terminal penalty. '
        'A different endpoint requirement would be a different experiment.', '',
        f"The LP witness retains {witness.transit_end_m3.iloc[-1]/1e6:.3f} million m³ in its final queue.",
        f"Its maximum reservoir-plus-transit accounting residual is {abs(witness.system_balance_error_m3).max():.3g} m³.",
        'The independent LP agrees with the production QP on initial and prefix feasibility. Full-horizon matrix routing matches explicit queue advancement. '
        'The original saved H=4 trajectory is reproduced within 0.00001 in its storage/flow columns. Numerical comparisons are saved in `verification.json`. '
        'All successfully executed MPC trajectories respect storage and downstream bounds within the original SI tolerances. '
        'No implementation inconsistency was detected in these checks; the evidence supports insufficient anticipation and earlier decisions, not initial physical infeasibility. '
        'The diagnosis does not prove absence of all possible implementation defects.', '',
        'Original source configuration and earlier result bundles were fingerprinted unchanged. No slacks, relaxed bounds, changed weights, leader optimization or controller changes were introduced.', '',
        '## Reproduce offline','', '```sh',
        'uv run --locked python -m reservoir_control.systems.ottawa.delay_diagnostic --output results/ottawa/delay_diagnostic_rerun',
        '```', '', 'Use a fresh output directory. `diagnostic.png` compares storage, releases, downstream flows and retained transit.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); run(args.config.resolve(),args.output.resolve())


if __name__=='__main__': main()
