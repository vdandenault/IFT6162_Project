from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy

from .hard_follower_study import rollout, make_problem
from .hard_follower import solve_hard_follower, FollowerTolerances
from .delay_diagnostic import explicit_storage_feasibility, boundary_state
from .routing import TransitState
from .reconstruction import ROOT, digest
from .discharge_comparison import snapshot

ARCHIVE=ROOT/'results/ottawa/horizon_sensitivity_v2'


def plain(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [plain(v) for v in value]
    return value


def volume_certificate(p,start_period):
    """Necessary upper-storage inequality for the study's zero/one-delay routing.

    Routing rows have at most one decision release. This permits exact individual
    release upper bounds, without assuming any new arrivals after the endpoint.
    It is a certificate of incompatibility, not an alternative operating policy.
    """
    a=np.asarray(p.routing_matrix);dt=np.asarray(p.duration_seconds)
    if np.any(np.count_nonzero(a,axis=1)>1): return None
    room=np.asarray(p.downstream_capacity_m3_s)-np.asarray(p.ontario_executable_release_m3_s)-np.asarray(p.other_downstream_m3_s)-np.asarray(p.in_transit_contribution_m3_s)
    maximum=np.asarray(p.release_maximum_m3_s,float).copy()
    for j in range(len(dt)):
        rows=np.flatnonzero(a[:,j]>0)
        if len(rows): maximum[j]=min(maximum[j],min(room[rows]/a[rows,j]))
    lower=p.initial_storage_m3+np.cumsum(dt*(np.asarray(p.net_inflow_m3_s)-maximum))
    conflict=np.flatnonzero(lower>np.asarray(p.storage_maximum_m3)[1:]+.01)
    if not len(conflict): return None
    j=int(conflict[0]);upper=np.asarray(p.storage_maximum_m3)[j+1]
    return dict(start_period=start_period,end_period=start_period+j,
        initial_storage_m3=p.initial_storage_m3,available_storage_m3=float(upper-p.initial_storage_m3),
        inflow_volume_m3=float(sum(dt[:j+1]*np.asarray(p.net_inflow_m3_s)[:j+1])),
        maximum_release_volume_m3=float(sum(dt[:j+1]*maximum[:j+1])),
        required_retention_m3=float(lower[j]-p.initial_storage_m3),
        unavoidable_storage_m3=float(lower[j]),unavoidable_excess_m3=float(lower[j]-upper),
        maximum_release_m3_s=maximum[:j+1].tolist())


def check_feasibility(p,tolerance,start_period):
    lp=explicit_storage_feasibility(p,tolerance);qp=solve_hard_follower(p,tolerance)
    if lp['status']=='solver_failure' or qp.status=='solver_failure':
        raise RuntimeError(f'Numerical failure requires investigation: LP={lp}, QP={qp.status}: {qp.message}')
    if (lp['status']=='feasible')!=qp.success:
        raise RuntimeError(f'Independent LP/QP disagreement: {lp["status"]} vs {qp.status}')
    return dict(problem=plain(asdict(p)),independent_lp=lp,qp_status=qp.status,
        qp_residuals=qp.residuals,volume_certificate=volume_certificate(p,start_period))


def inherited_states(c,case,trace):
    """Audit every archived executed balance/queue, never substitute a witness."""
    states=[];storage=c['initial_storage_m3'];previous=c['previous_executed_release_m3_s']
    transit=TransitState(tuple(case['initial_transit_volumes_m3']));dt=c['clock']['duration_seconds']
    tolerance=c['acceptance'];initial_system=storage+transit.total_m3;inputs=arrivals=0.
    for k in range(len(trace)+1):
        actual_s,actual_prev,actual_q=boundary_state(c,case,trace,k)
        np.testing.assert_allclose(actual_s,storage,atol=tolerance['storage_m3'],rtol=0)
        np.testing.assert_allclose(actual_prev,previous,atol=tolerance['flow_m3_s'],rtol=0)
        np.testing.assert_allclose(actual_q.volumes_m3,transit.volumes_m3,atol=tolerance['storage_m3'],rtol=0)
        states.append(dict(executed_actions=k,next_decision_period=k+1,storage_m3=actual_s,
            previous_release_m3_s=actual_prev,transit_volumes_m3=list(actual_q.volumes_m3),
            system_balance_residual_m3=storage+transit.total_m3-initial_system-inputs+arrivals))
        if k==len(trace): break
        row=trace.iloc[k];release=row.executed_release_m3_s
        np.testing.assert_allclose(row.storage_start_m3,storage,atol=tolerance['storage_m3'],rtol=0)
        assert row.step==k and row.duration_seconds==dt
        assert row.realized_net_inflow_m3_s==case['realized_net_inflow_m3_s'][k]
        storage+=dt*(case['realized_net_inflow_m3_s'][k]-release)
        transit,arrival,residual=transit.advance(release,dt)
        assert abs(residual)<=tolerance['storage_m3']
        np.testing.assert_allclose(row.controlled_arrival_m3_s,arrival,atol=tolerance['flow_m3_s'],rtol=0)
        inputs+=dt*case['realized_net_inflow_m3_s'][k];arrivals+=dt*arrival;previous=release
    return states


def offline_witness(c,case,k,storage,transit,result):
    """Witness follows true inputs from the actual inherited boundary, offline only."""
    r=np.array(result['release_m3_s']);s=np.array(result['storage_m3'])
    dt=c['clock']['duration_seconds'];initial=storage+transit.total_m3;inputs=arrivals=0.;rows=[]
    for j,release in enumerate(r):
        period=k+j;transit,arrival,_=transit.advance(release,dt)
        inputs+=dt*case['realized_net_inflow_m3_s'][period];arrivals+=dt*arrival
        combined=c['known_ontario_executable_schedule_m3_s'][period]+c['known_other_downstream_forecast_m3_s'][period]+arrival
        np.testing.assert_allclose(combined,result['downstream_m3_s'][j],atol=c['acceptance']['flow_m3_s'],rtol=0)
        error=s[j+1]+transit.total_m3-initial-inputs+arrivals
        assert abs(error)<=c['acceptance']['storage_m3']
        rows.append(dict(period=period+1,release_m3_s=release,storage_start_m3=s[j],storage_end_m3=s[j+1],
            downstream_m3_s=combined,transit_end_m3=transit.total_m3,transit_queue_m3=json.dumps(transit.volumes_m3),
            system_balance_residual_m3=error,information='offline true-inflow feasibility witness; not a causal recommendation'))
    return pd.DataFrame(rows)


def classify(a,b,c,initial):
    if a!='infeasible': raise RuntimeError('The archived failed noisy forecast must be infeasible')
    if c=='feasible':
        if b!='feasible': raise RuntimeError('Full feasibility cannot coexist with infeasible true prefix')
        return 'forecast_induced_infeasibility'
    if initial!='feasible': return 'initial_physical_infeasibility'
    if b=='feasible': return 'loss_of_recoverability_and_local_forecast_rejection'
    return 'loss_of_recoverability'


def analyze_case(c,case,seed,archive,bank,folder):
    folder.mkdir();key=f"{case['id']}_noisy_{seed}_h5"
    trace=pd.read_csv(archive/'runs'/f'{key}.csv')
    saved_plans=json.loads((archive/'runs'/f'{key}_plans.json').read_text())
    replay,plans,summary=rollout(c,case,'mpc','noisy',seed,forecast_bank=bank)
    if len(replay)!=len(trace): raise RuntimeError('Original failure time does not reproduce')
    # Every physical field is compared in its own original SI tolerance.
    for col in trace.columns:
        if col=='solver_seconds': continue
        if pd.api.types.is_numeric_dtype(trace[col]) and not pd.api.types.is_bool_dtype(trace[col]):
            tol=c['acceptance']['flow_m3_s'] if col.endswith('m3_s') else c['acceptance']['storage_m3']
            np.testing.assert_allclose(replay[col],trace[col],atol=tol,rtol=0,equal_nan=True)
        else:
            if col=='transit_queue_end_m3':
                for x,y in zip(replay[col],trace[col]): np.testing.assert_allclose(json.loads(x),json.loads(y),atol=c['acceptance']['storage_m3'],rtol=0)
            else: assert replay[col].tolist()==trace[col].tolist(),col
    assert summary['status']=='infeasible' and saved_plans[-1]['status']==plans[-1]['status']=='infeasible'
    assert len(plans)==len(saved_plans) and plans[-1]['step']==saved_plans[-1]['step']==len(trace)
    for original,reproduced in zip(saved_plans,plans):
        assert original['status']==reproduced['status']
        k=original['step'];forecast=original['supplied_problem']['net_inflow_m3_s']
        np.testing.assert_array_equal(forecast,bank[k,:len(forecast)])
        np.testing.assert_array_equal(reproduced['supplied_problem']['net_inflow_m3_s'],forecast)
    trace.to_csv(folder/'original_executed_trajectory.csv',index=False)
    states=inherited_states(c,case,trace);tolerance=FollowerTolerances(**c['acceptance']);checks=[]
    for state in states:
        k=state['executed_actions'];queue=TransitState(tuple(state['transit_volumes_m3']))
        p=make_problem(c,case,k,state['storage_m3'],state['previous_release_m3_s'],queue,case['realized_net_inflow_m3_s'][k:])
        result=check_feasibility(p,tolerance,k+1);checks.append(dict(state=state,**result))
        if result['independent_lp']['status']=='feasible':
            offline_witness(c,case,k,state['storage_m3'],queue,result['independent_lp']).to_csv(folder/f'offline_true_witness_after_{k}_actions.csv',index=False)
    k=len(trace);state=states[-1];queue=TransitState(tuple(state['transit_volumes_m3']));window=min(5,12-k)
    first_failed={}
    for name,inputs in [('A_noisy_same_window',bank[k,:window]),
                        ('B_true_same_window',case['realized_net_inflow_m3_s'][k:k+window]),
                        ('C_true_full_remaining',case['realized_net_inflow_m3_s'][k:])]:
        p=make_problem(c,case,k,state['storage_m3'],state['previous_release_m3_s'],queue,inputs)
        first_failed[name]=check_feasibility(p,tolerance,k+1)
    # A must preserve every supplied field, including previous release and c.
    original_problem=saved_plans[-1]['supplied_problem']
    current_problem=first_failed['A_noisy_same_window']['problem']
    for field,value in original_problem.items():
        if field=='weights':
            assert current_problem[field]==value
        else:
            tol=c['acceptance']['storage_m3'] if ('storage' in field or field=='storage_scale_m3') else c['acceptance']['flow_m3_s']
            np.testing.assert_allclose(current_problem[field],value,atol=tol,rtol=0)
    initial=checks[0]['independent_lp']['status'];lost=next((i for i,r in enumerate(checks) if r['independent_lp']['status']=='infeasible'),None)
    if lost is not None and any(r['independent_lp']['status']=='feasible' for r in checks[lost:]):
        raise RuntimeError('True full-experiment feasibility unexpectedly reappeared; investigate')
    a,b,d=[r['independent_lp']['status'] for r in first_failed.values()]
    classification=classify(a,b,d,initial)
    comparison=None
    if lost is not None and lost>0:
        witness=checks[lost-1]['independent_lp']
        comparison=dict(action_period=lost,actual_executed_release_m3_s=trace.executed_release_m3_s.iloc[lost-1],
            one_feasible_continuation_first_release_m3_s=witness['release_m3_s'][0],
            caveat='LP witness is one possible feasible sequence, not the unique required action; all true future inputs are offline only.',
            witness_file=f'offline_true_witness_after_{lost-1}_actions.csv')
    (folder/'prefix_feasibility.json').write_text(json.dumps(plain(checks),indent=2)+'\n')
    (folder/'first_failed_solve_ABC.json').write_text(json.dumps(plain(first_failed),indent=2)+'\n')
    (folder/'loss_action_comparison.json').write_text(json.dumps(plain(comparison),indent=2)+'\n')
    row=dict(scenario=case['id'],seed=seed,first_failed_decision=k+1,executed_periods=k,
        A_noisy_same_window=a,B_true_same_window=b,C_true_full_remaining=d,initial_full_feasibility=initial,
        first_loss_action=lost if lost and initial=='feasible' else None,classification=classification,
        both_mechanisms_demonstrated=(b=='feasible' and d=='infeasible'),
        original_failure_status=summary['status'],replayed_original_trajectory=True,
        failed_storage_m3=state['storage_m3'],failed_transit_m3=sum(state['transit_volumes_m3']),
        max_actual_balance_residual_m3=max(abs(x['system_balance_residual_m3']) for x in states))
    return row,checks,first_failed,comparison


def run(archive,output):
    before=snapshot(archive);settings=json.loads((archive/'configuration.json').read_text());c=deepcopy(settings['original_configuration']);c['horizon']=5
    provenance=json.loads((archive/'forecast_provenance.json').read_text())
    for name,expected in provenance['forecast_bank_sha256'].items():
        if digest(archive/name)!=expected: raise RuntimeError('Saved forecast bank hash mismatch')
    summary=pd.read_csv(archive/'summary.csv')
    failures=summary[(summary.horizon==5)&(summary.forecast_mode=='noisy')&(~summary.successful_completion)&(summary.scenario!='deliberately_infeasible')]
    if len(failures)!=7 or set(failures.status)!={'infeasible'}: raise ValueError('Expected seven archived infeasible H5 cases')
    expected={(s,seed) for s in ('tight_downstream','one_period_delay') for seed in (11,22,33)}
    extras=set(zip(failures.scenario,failures.seed))-expected
    assert expected.issubset(set(zip(failures.scenario,failures.seed))) and len(extras)==1 and next(iter(extras))[0]=='one_period_delay_feasible'
    output.mkdir(parents=True,exist_ok=False)
    (output/'configuration.json').write_text(json.dumps(dict(study_horizon=5,source_archive=str(archive),
        frozen_configuration=c,selected_cases=failures[['scenario','seed']].to_dict('records'),
        scope='Offline feasibility diagnostic only; true future inflows never supplied to replayed noisy policy.'),indent=2)+'\n')
    all_rows=[];all_checks={};details={}
    with np.load(archive/'forecast_bank.npz',allow_pickle=False) as saved:
        for _,r in failures.iterrows():
            case=next(x for x in c['scenarios'] if x['id']==r.scenario);key=f'{r.scenario}_noisy_{int(r.seed)}';bank=saved[key].copy();bank.flags.writeable=False
            row,checks,abc,comparison=analyze_case(c,case,int(r.seed),archive,bank,output/key)
            assert row['first_failed_decision']==r.first_failed_decision_period
            all_rows.append(row);all_checks[key]=checks;details[key]=(abc,comparison)
    frame=pd.DataFrame(all_rows);frame.to_csv(output/'summary.csv',index=False)
    witness_residual=0.;witness_count=0
    for path in output.glob('*/offline_true_witness_after_*_actions.csv'):
        witness=pd.read_csv(path);witness_count+=1
        witness_residual=max(witness_residual,float(abs(witness.system_balance_residual_m3).max()))
    (output/'verification.json').write_text(json.dumps(dict(
        replayed_failed_cases=len(frame),exact_saved_forecasts_reused=True,
        original_failure_status_and_time_reproduced=True,
        inherited_storage_previous_release_and_transit_checked=True,
        independent_LP_and_QP_agree=True,feasible_offline_witnesses=witness_count,
        maximum_witness_system_balance_residual_m3=witness_residual,
        maximum_actual_system_balance_residual_m3=float(frame.max_actual_balance_residual_m3.max()),
        forecast_regeneration=False),indent=2)+'\n')
    plot_recoverability(frame,all_checks,output/'recoverability.png')
    write_report(frame,all_checks,details,output)
    if before!=snapshot(archive): raise RuntimeError('Archived study changed')
    sources=[Path(__file__),Path(__file__).with_name('hard_follower_study.py'),Path(__file__).with_name('hard_follower.py'),
             Path(__file__).with_name('delay_diagnostic.py'),Path(__file__).with_name('routing.py'),ROOT/'src/reservoir_control/controllers/mpc/convex_response.py']
    (output/'provenance.json').write_text(json.dumps(dict(source_sha256={str(p):digest(p) for p in sources},
        source_archive_sha256=before,forecast_provenance=provenance,archive_verified_unchanged=True,
        scipy_version=scipy.__version__,numpy_version=np.__version__,forecast_regeneration=False),indent=2)+'\n')
    print(frame[['scenario','seed','first_failed_decision','A_noisy_same_window','B_true_same_window','C_true_full_remaining','first_loss_action','classification']].to_string(index=False))
    print(f'Report: {output/"report.md"}')


def plot_recoverability(frame,checks,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    for y,row in enumerate(frame.itertuples()):
        key=f'{row.scenario}_noisy_{row.seed}';records=checks[key]
        x=[r['state']['executed_actions'] for r in records]
        ax.plot(x,[y]*len(x),color='lightgray',lw=2,zorder=1)
        for xx,r in zip(x,records):
            feasible=r['independent_lp']['status']=='feasible'
            ax.scatter(xx,y,color='#26836a' if feasible else '#c74747',s=75,zorder=3)
        # The failed decision follows this many executed actions; label the 1-based time.
        ax.scatter(row.executed_periods,y,s=220,facecolors='none',edgecolors='black',linewidths=1.5,zorder=4)
        ax.annotate(f'failure before period {row.first_failed_decision}',(row.executed_periods+.12,y),fontsize=8,va='center')
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([],[],marker='o',color='#26836a',ls='',label='True full completion feasible'),
        Line2D([],[],marker='o',color='#c74747',ls='',label='True full completion infeasible'),
        Line2D([],[],marker='o',markerfacecolor='none',color='black',ls='',label='Original noisy failed solve')],loc='lower left',bbox_to_anchor=(0,1.02),ncol=3,fontsize=8)
    ax.set_yticks(range(len(frame)),[f'{r.scenario} / seed {r.seed}' for r in frame.itertuples()]);ax.invert_yaxis()
    ax.set_xticks(range(7));ax.set_xlim(-.3,7.5);ax.grid(axis='x',alpha=.2)
    ax.set_xlabel('Actually executed actions (0 = original state); true inflows used only for offline checks')
    fig.savefig(path,dpi=150);plt.close(fig)


def write_report(frame,checks,details,output):
    lines=['# Seven noisy H5 failures: forecast rejection or lost recoverability?','',
        '**Six runs lose recoverability after earlier executed actions; one remains recoverable and is rejected by its current noisy forecast. All seven experiments are feasible initially.**','',
        'The original trajectories, saved forecast values and first failed decision times were replayed within existing SI tolerances. '
        'No noise was regenerated. The replayed controller receives only its original saved noisy H5 forecasts. '
        'Independent explicit-storage LPs and the existing pure QP agree on every offline feasibility check; numerical failure/disagreement would stop this diagnostic.', '',
        'A = saved noisy H5 window; B = true inputs over that same window; C = true inputs over the full remaining experiment. '
        'All three inherit the identical actual storage, previous executed release and transit queue at failure. '
        'Only inflow information and, for C, horizon length change in the offline checks.', '',
        '| Scenario | Seed | Failed decision | A | B | C | First loss after action | Classification |',
        '|---|---:|---:|---|---|---|---:|---|']
    for r in frame.itertuples():
        loss='none' if pd.isna(r.first_loss_action) else str(int(r.first_loss_action))
        lines.append(f'| {r.scenario} | {r.seed} | {r.first_failed_decision} | {r.A_noisy_same_window} | {r.B_true_same_window} | {r.C_true_full_remaining} | {loss} | {r.classification} |')
    lines+=['','## What happened','',
        'The six tight-capacity runs are already impossible to complete when the noisy solver stops. Replacing its forecast with true inflows cannot rescue those inherited states. '
        'Their actual storage is still within bounds: loss of recoverability means a future unavoidable conflict, not necessarily an overflow already observed.', '',
        'The less restrictive delayed case, seed 22, remains recoverable at every executed prefix. Its failed noisy forecast overstates the volume that must be retained; '
        'the same-window true inputs and full remaining true inputs both admit a feasible continuation. This is forecast-induced rejection, not lost physical recoverability.', '',
        'No run demonstrates both mechanisms at the failed solve: none has B feasible with C infeasible. '
        'No experiment is initially infeasible and no independent numerical solver inconsistency was found. '
        'Earlier noisy forecasts influenced the decisions in the six lost-recoverability cases, but this diagnostic does not isolate a unique earlier forecast error as their cause.', '',
        '## Loss timing and volume conflicts','']
    for r in frame.itertuples():
        key=f'{r.scenario}_noisy_{r.seed}';abc,comparison=details[key]
        lines += [f'### {r.scenario}, seed {r.seed}','']
        if comparison:
            lost=int(r.first_loss_action);certificate=checks[key][lost]['volume_certificate']
            lines.append(f"Completion is feasible immediately before action {lost}, and infeasible after it. That action executed {comparison['actual_executed_release_m3_s']:.3f} m³/s. "
                f"One offline feasible continuation from the prior actual state starts at {comparison['one_feasible_continuation_first_release_m3_s']:.3f} m³/s. "
                'That LP sequence is an example, not a uniquely required first action or an available causal recommendation.')
        else:
            lines.append('Recoverability remains intact through the failed solve. A complete offline witness from that actual state is saved.')
            certificate=abc['A_noisy_same_window']['volume_certificate']
        if certificate:
            label='True-input' if comparison else 'Noisy-forecast'
            lines.append(f"{label} volume check immediately after the loss action (or at forecast rejection), periods {certificate['start_period']}–{certificate['end_period']}: "
                f"room {certificate['available_storage_m3']/1e6:.3f} million m³, incoming volume {certificate['inflow_volume_m3']/1e6:.3f} million m³, "
                f"maximum release volume {certificate['maximum_release_volume_m3']/1e6:.3f} million m³. "
                f"Required retention is {certificate['required_retention_m3']/1e6:.3f} million m³, exceeding room by {certificate['unavoidable_excess_m3']/1e6:.3f} million m³. "
                'Release bounds include the original outlet and downstream routing restrictions. The inequality does not use objective weights or slack.')
        lines.append('')
    lines+=['## Files, accounting and limits','',
        '`summary.csv` is the seven-row classification table. `recoverability.png` shows feasibility after each actual executed prefix and marks the failed decisions. '
        'Each case directory saves exact inherited states, A/B/C problems and independent results, every full-remaining prefix check, volume certificates, '
        'the archived executed trajectory, and offline feasible witness CSVs wherever such a witness exists. '
        'The continuation immediately before each loss is paired with the actual action in `loss_action_comparison.json`.', '',
        'Witnesses are offline accounting/feasibility constructions, not policies, independently validated hydrology, or recommendations based on information then available. '
        'They never replace actual prefix storage or queues. Water in transit is explicitly routed and retained at the common period-12 endpoint. '
        'No terminal queue-empty constraint or post-experiment capacities are added, and no safety claim is made beyond that endpoint.', '',
        'No controller defaults, constraints, weights, forecasts, operating policy, or archived results were modified. '
        'The study horizon is explicitly H5 while the controller default remains H4. '
        'These seven selected synthetic failures do not support general robustness or causality claims.', '',
        '```sh','uv run --locked python -m reservoir_control.systems.ottawa.noisy_failure_diagnostic --output results/ottawa/noisy_h5_diagnostic_rerun','```']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=ArgumentParser(
        description=(
            "Offline recoverability diagnosis of the seven archived noisy H5 failures.\n"
            "\n"
            "Never generates forecasts or supplies true future inflows to the replayed policy.\n"
        )
    )
    parser.add_argument('--archive',type=Path,default=ARCHIVE)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    run(args.archive.resolve(),args.output.resolve())


if __name__=='__main__':main()
