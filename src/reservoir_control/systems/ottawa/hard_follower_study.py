"""Standalone synthetic receding-horizon hard-follower demonstrations, no leader."""
from argparse import ArgumentParser
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .hard_follower import (HardFollowerProblem, FollowerWeights, FollowerTolerances,
                            solve_hard_follower, InvalidFollowerInput)
from .routing import TransitState
from .reconstruction import ROOT, digest
from .discharge_comparison import snapshot

CONFIG = ROOT/'configs/studies/ottawa/hard_follower_v1.json'


def inflow_forecast(mode, realized_past, initial_estimate, horizon, rng, sigma, *, perfect_values=None):
    if mode == 'perfect':
        if perfect_values is None or len(perfect_values) != horizon: raise ValueError('Explicit perfect-forecast values required')
        return np.asarray(perfect_values, float).copy()
    if mode != 'noisy': raise ValueError('Unknown forecast mode')
    # This branch never receives or indexes future realized inflows.
    last = float(realized_past[-1]) if len(realized_past) else initial_estimate
    return np.full(horizon, last)+rng.normal(0., sigma, horizon)


def make_problem(c, scenario, k, storage, previous, transit, forecast):
    h = len(forecast); dt = np.full(h, c['clock']['duration_seconds'])
    a, carry = transit.forecast(dt)
    return HardFollowerProblem(storage, previous, forecast,
        c['known_ontario_executable_schedule_m3_s'][k:k+h],
        c['known_other_downstream_forecast_m3_s'][k:k+h], a, carry,
        scenario['downstream_capacity_m3_s'][k:k+h],
        np.full(h+1, c['storage_minimum_m3']), np.full(h+1, c['storage_maximum_m3']),
        np.full(h, c['release_minimum_m3_s']), np.full(h, c['release_maximum_m3_s']),
        np.full(h+1, c['storage_target_m3']), np.full(h, c['release_target_m3_s']), dt,
        c['storage_scale_m3'], c['flow_scale_m3_s'], FollowerWeights(**c['weights']))


def rollout(c, scenario, policy, mode, seed, *, solver=solve_hard_follower):
    if policy not in ('mpc', 'reference'): raise ValueError('Unknown policy')
    realized = np.asarray(scenario['realized_net_inflow_m3_s'], float)
    storage, previous = c['initial_storage_m3'], c['previous_executed_release_m3_s']
    transit = TransitState(tuple(scenario['initial_transit_volumes_m3']))
    if len(transit.volumes_m3) != scenario['routing_delay_periods']: raise ValueError('Routing queue/delay mismatch')
    rng = np.random.default_rng(seed); traces, plans = [], []
    tolerances = FollowerTolerances(**c['acceptance'])
    stop = 'completed'; stop_message = ''; n = len(realized)
    initial_system_storage = storage+transit.total_m3
    input_volume, arrived_volume = 0., 0.
    for k in range(n):
        h = min(c['horizon'], n-k)
        forecast = inflow_forecast(mode, realized[:k], c['forecast_noise']['initial_observed_inflow_estimate_m3_s'],
            h, rng, c['forecast_noise']['standard_deviation_m3_s'],
            perfect_values=realized[k:k+h] if mode == 'perfect' else None)
        problem = make_problem(c, scenario, k, storage, previous, transit, forecast)
        status, seconds, residuals, prediction = 'fixed_reference', 0., {}, None
        planned_next, predicted_downstream = np.nan, np.nan
        if policy == 'mpc':
            try: result = solver(problem, tolerances)
            except InvalidFollowerInput as error:
                stop, stop_message = 'invalid_input', str(error)
                plans.append(dict(step=k, status=stop, message=stop_message, action=None)); break
            status, seconds, residuals = result.status, result.elapsed_seconds, result.residuals
            entry = dict(step=k, status=status, message=result.message, solver_seconds=seconds,
                         objective=result.objective, residuals=residuals,
                         supplied_problem={name: (value.tolist() if isinstance(value, np.ndarray) else value)
                            for name, value in asdict(problem).items()}, action=result.first_release_m3_s)
            if not result.success:
                if result.first_release_m3_s is not None: raise RuntimeError('Unsuccessful solve supplied an action')
                plans.append(entry); stop, stop_message = status, result.message; break
            release = result.first_release_m3_s
            if release is None or release != result.release_m3_s[0]: raise RuntimeError('Missing/inconsistent first action')
            prediction = result.release_m3_s.copy()
            terminal_transit = transit.after_plan(result.release_m3_s, problem.duration_seconds)
            entry.update(planned_release_m3_s=result.release_m3_s.tolist(), planned_storage_m3=result.storage_m3.tolist(),
                         planned_downstream_m3_s=result.downstream_flow_m3_s.tolist(),
                         pending_new_release_volume_m3=result.pending_planned_release_volume_m3,
                         planned_terminal_transit_m3=terminal_transit.total_m3)
            plans.append(entry)
            planned_next, predicted_downstream = result.storage_m3[1], result.downstream_flow_m3_s[0]
        else:
            release = c['reference_release_m3_s']
            planned_next = storage+problem.duration_seconds[0]*(forecast[0]-release)
            predicted_downstream = problem.ontario_executable_release_m3_s[0]+problem.other_downstream_m3_s[0]+(
                problem.routing_matrix@np.full(h, release)+problem.in_transit_contribution_m3_s)[0]
        dt = float(problem.duration_seconds[0]); old_transit = transit.total_m3
        # Execute exactly one action using realized forcing, without a storage projection.
        following = storage+dt*(realized[k]-release)
        next_transit, controlled_arrival, route_error = transit.advance(release, dt)
        combined = c['known_ontario_executable_schedule_m3_s'][k]+c['known_other_downstream_forecast_m3_s'][k]+controlled_arrival
        storage_violation = max(0., c['storage_minimum_m3']-following, following-c['storage_maximum_m3'])
        down_violation = max(0., combined-scenario['downstream_capacity_m3_s'][k])
        release_violation = max(0., c['release_minimum_m3_s']-release, release-c['release_maximum_m3_s'])
        w = c['weights']
        stage = (w['storage']*((following-c['storage_target_m3'])/c['storage_scale_m3'])**2
                 +w['release']*((release-c['release_target_m3_s'])/c['flow_scale_m3_s'])**2
                 +w['change']*((release-previous)/c['flow_scale_m3_s'])**2)
        input_volume += realized[k]*dt; arrived_volume += controlled_arrival*dt
        system_error = following+next_transit.total_m3-initial_system_storage-input_volume+arrived_volume
        predicted_storage_violation = max(0., c['storage_minimum_m3']-planned_next, planned_next-c['storage_maximum_m3'])
        traces.append(dict(step=k, storage_start_m3=storage, storage_end_m3=following,
            storage_target_m3=c['storage_target_m3'], storage_minimum_m3=c['storage_minimum_m3'], storage_maximum_m3=c['storage_maximum_m3'],
            executed_release_m3_s=release, previous_executed_release_m3_s=previous,
            planned_first_release_m3_s=release if policy=='mpc' else np.nan,
            release_minimum_m3_s=c['release_minimum_m3_s'], release_maximum_m3_s=c['release_maximum_m3_s'],
            realized_net_inflow_m3_s=float(realized[k]), forecast_net_inflow_m3_s=float(forecast[0]),
            forecast_error_m3_s=float(forecast[0]-realized[k]), planned_next_storage_m3=planned_next,
            predicted_downstream_flow_m3_s=predicted_downstream,
            predicted_first_step_feasible=predicted_storage_violation <= tolerances.storage_m3 and predicted_downstream <= scenario['downstream_capacity_m3_s'][k]+tolerances.flow_m3_s,
            ontario_executable_m3_s=c['known_ontario_executable_schedule_m3_s'][k], other_downstream_m3_s=c['known_other_downstream_forecast_m3_s'][k],
            controlled_arrival_m3_s=controlled_arrival, combined_downstream_m3_s=combined,
            downstream_capacity_m3_s=scenario['downstream_capacity_m3_s'][k],
            transit_start_m3=old_transit, transit_end_m3=next_transit.total_m3,
            transit_queue_end_m3=json.dumps(next_transit.volumes_m3), duration_seconds=dt,
            storage_violation_m3=storage_violation, downstream_violation_m3_s=down_violation,
            release_violation_m3_s=release_violation, stage_cost=stage, solver_status=status,
            solver_seconds=seconds, routing_balance_error_m3=route_error,
            reservoir_balance_error_m3=following-storage-dt*(realized[k]-release), system_balance_error_m3=system_error))
        storage, previous, transit = following, release, next_transit
        if storage_violation > tolerances.storage_m3:
            stop, stop_message = 'physical_storage_violation', 'Realized storage left declared physical bounds; no clipping/emergency policy'; break
        if release_violation > tolerances.flow_m3_s:
            stop, stop_message = 'release_bound_violation', 'Reference/policy release violates physical outlet bounds'; break
    trace = pd.DataFrame(traces)
    complete = len(traces) == n and stop == 'completed'
    terminal = c['weights']['terminal']*((storage-c['storage_target_m3'])/c['storage_scale_m3'])**2 if complete else None
    summary = dict(scenario=scenario['id'], policy=policy, forecast_mode=mode, seed=int(seed),
        status=stop, reason=stop_message, complete=complete, executed_steps=len(trace), requested_steps=n,
        stage_cost=float(trace.stage_cost.sum()) if len(trace) else 0., terminal_cost=terminal,
        full_objective=float(trace.stage_cost.sum()+terminal) if complete else None,
        final_storage_m3=storage, final_transit_volume_m3=transit.total_m3, final_transit_queue_m3=list(transit.volumes_m3),
        initial_transit_volume_m3=initial_system_storage-c['initial_storage_m3'],
        cumulative_system_balance_error_m3=storage+transit.total_m3-initial_system_storage-input_volume+arrived_volume,
        max_storage_violation_m3=float(trace.storage_violation_m3.max()) if len(trace) else 0.,
        max_downstream_violation_m3_s=float(trace.downstream_violation_m3_s.max()) if len(trace) else 0.,
        max_release_violation_m3_s=float(trace.release_violation_m3_s.max()) if len(trace) else 0.,
        predicted_feasible_realized_violation_steps=int(((trace.storage_violation_m3 > tolerances.storage_m3)
            |(trace.downstream_violation_m3_s > tolerances.flow_m3_s))[trace.predicted_first_step_feasible].sum()) if len(trace) else 0,
        release_standard_deviation_m3_s=float(trace.executed_release_m3_s.std(ddof=0)) if len(trace) else None,
        mean_abs_release_change_m3_s=float((trace.executed_release_m3_s-trace.previous_executed_release_m3_s).abs().mean()) if len(trace) else None,
        inflow_forecast_mae_m3_s=float(trace.forecast_error_m3_s.abs().mean()) if len(trace) else None,
        total_solver_seconds=float(sum(x.get('solver_seconds',0.) for x in plans)),
        max_solver_storage_residual_m3=max([x.get('residuals',{}).get('storage_violation_m3',0.) for x in plans] or [0.]),
        solver_failures=int(stop in ('infeasible','solver_failure','invalid_input')))
    return trace, plans, summary


def validate_config(c):
    if c['schema_version'] != 1 or c['clock']['duration_seconds'] != 365*86400/48:
        raise ValueError('This study explicitly uses 48 equal periods per 365-day year (657000 s)')
    if c['horizon'] < 1 or c['reference_release_m3_s'] < 0: raise ValueError('Positive horizon and nonnegative reference required')
    n = len(c['known_ontario_executable_schedule_m3_s'])
    if n == 0 or len(c['known_other_downstream_forecast_m3_s']) != n: raise ValueError('Aligned supplied downstream conditions required')
    for case in c['scenarios']:
        if len(case['realized_net_inflow_m3_s']) != n or len(case['downstream_capacity_m3_s']) != n:
            raise ValueError('Complete scenario arrays required')
        if case['routing_delay_periods'] != len(case['initial_transit_volumes_m3']): raise ValueError('Explicit transit queue required')
    if not np.isfinite(c['forecast_noise']['standard_deviation_m3_s']) or c['forecast_noise']['standard_deviation_m3_s'] < 0:
        raise ValueError('Nonnegative finite noise amplitude required')
    return c


def plot_case(c, case, traces, path, plans=()):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(5, 1, figsize=(11, 14), sharex=True, layout='constrained')
    n = len(case['realized_net_inflow_m3_s'])
    for policy, frame in traces.items():
        if frame.empty:
            axes[0].plot([], [], label=f'{policy}: stopped before execution')
            continue
        x = frame.step.to_numpy()+1
        axes[0].plot(np.r_[0, x], np.r_[c['initial_storage_m3'], frame.storage_end_m3]/1e9, label=policy)
        axes[1].step(x, frame.executed_release_m3_s, where='mid', label=policy)
        if policy == 'mpc':
            axes[1].plot(x, frame.planned_first_release_m3_s, '.', label='planned first action')
        axes[2].step(x, frame.combined_downstream_m3_s, where='mid', label=policy)
        axes[3].plot(x, frame.forecast_error_m3_s, label=policy)
        axes[4].plot(np.r_[0, x], np.r_[sum(case['initial_transit_volumes_m3']), frame.transit_end_m3]/1e9, label=policy)
    for index, plan in enumerate(plans):
        values = plan.get('planned_release_m3_s')
        if values:
            axes[1].plot(np.arange(len(values))+plan['step']+1, values, color='gray', alpha=.25,
                         linewidth=.8, label='replanned horizons' if index == 0 else None)
    for value, label in [(c['storage_minimum_m3'], 'minimum'), (c['storage_maximum_m3'], 'maximum'),
                          (c['storage_target_m3'], 'target')]:
        axes[0].axhline(value/1e9, linestyle='--', color='gray', label=label)
    for value in (c['release_minimum_m3_s'], c['release_maximum_m3_s']):
        axes[1].axhline(value, linestyle='--', color='gray')
    axes[2].step(np.arange(n)+1, case['downstream_capacity_m3_s'], where='mid', color='black', linestyle='--', label='capacity')
    axes[3].axhline(0., color='gray')
    for ax, label in zip(axes, ['Storage (billion m³)', 'Release (m³/s)', 'Combined flow (m³/s)',
                               'Forecast − actual (m³/s)', 'Water in transit (billion m³)']):
        ax.set_ylabel(label); ax.grid(alpha=.2); ax.legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel('Executed period (657,000 seconds each)')
    fig.suptitle(f"Synthetic aggregate — {case['id']}\nTrajectories stop on failure; storage is never clipped")
    fig.savefig(path, dpi=150); plt.close(fig)


def run_study(config, output, mode):
    c = validate_config(json.loads(config.read_text()))
    output.mkdir(parents=True, exist_ok=False)
    # Fingerprint prior small observed/diagnostic bundles, never rewrite them.
    protected = [ROOT/'results/ottawa'/name for name in
        ('baskatong_1970_observed_v1', 'baskatong_discharge_comparison_v2', 'baskatong_linear_scenario_final_v2')]
    before = {str(p): snapshot(p) for p in protected if p.exists()}
    (output/'configuration.json').write_text(json.dumps(c, indent=2)+'\n')
    seeds = [0] if mode == 'perfect' else c['forecast_noise']['seeds']
    summaries, pairs, all_plans = [], [], []
    for case in c['scenarios']:
        for seed in seeds:
            traces, local, mpc_plans = {}, {}, []
            for policy in ('mpc', 'reference'):
                trace, plans, summary = rollout(c, case, policy, mode, seed)
                traces[policy], local[policy] = trace, summary
                if policy == 'mpc': mpc_plans = plans
                stem = f"{case['id']}_{seed}_{policy}"
                trace.to_csv(output/f'{stem}.csv', index=False)
                all_plans.append(dict(scenario=case['id'], seed=seed, policy=policy, plans=plans))
                summaries.append(summary)
            common = min(len(t) for t in traces.values())
            pairs.append(dict(scenario=case['id'], seed=seed, common_executed_steps=common,
                mpc_common_stage_cost=float(traces['mpc'].stage_cost.iloc[:common].sum()) if common else None,
                reference_common_stage_cost=float(traces['reference'].stage_cost.iloc[:common].sum()) if common else None,
                full_cost_comparable=all(s['complete'] for s in local.values()),
                mpc_full_objective=local['mpc']['full_objective'], reference_full_objective=local['reference']['full_objective']))
            plot_case(c, case, traces, output/f"{case['id']}_{seed}.png", mpc_plans)
    pd.DataFrame(summaries).to_csv(output/'summary.csv', index=False)
    pd.DataFrame(pairs).to_csv(output/'paired_comparison.csv', index=False)
    (output/'plans.json').write_text(json.dumps(all_plans, indent=2, allow_nan=False)+'\n')
    (output/'summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False)+'\n')
    assert before == {str(p): snapshot(p) for p in protected if p.exists()}, 'Protected observations changed'
    sources = [Path(__file__), Path(__file__).with_name('hard_follower.py'), Path(__file__).with_name('routing.py'),
               ROOT/'src/reservoir_control/controllers/mpc/convex_response.py', config, ROOT/'uv.lock']
    import importlib.metadata
    provenance = dict(forecast_mode=mode, source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
                      protected_before_after=before, scipy_version=importlib.metadata.version('scipy'),
                      numpy_version=np.__version__, seeds=seeds)
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    lines = ['# Standalone hard-constrained Ottawa follower', '',
        f'Forecast mode: **{mode}**. All physical parameters are synthetic aggregate assumptions, not calibrated Ottawa or Baskatong values.', '',
        'The pure optimizer eliminates storage through the exact SI water balance and solves a strictly convex release QP. '
        'A feasibility LP distinguishes infeasibility; independent SI and KKT residual checks gate every executable action. '
        'Storage, outlet and supplied downstream capacities are hard constraints, with no slack or clipping. '
        'Ontario releases are supplied executable schedules; there is no Ontario level model or leader optimization.', '',
        'Only the first planned release is executed. Forecasts are either explicitly perfect, or previous-observed-inflow persistence '
        'plus seeded Gaussian errors. The noisy branch receives no future realized inflows. Ontario, other contributions and capacities '
        'are known assumptions in both modes. The prediction horizon shrinks at the study endpoint.', '',
        'Routing is an assumed zero or one-period plug-flow delay. Queue entries are volumes; water still travelling at the horizon '
        'or rollout endpoint remains in the queue. The clock is 48 equal periods per 365 days (657,000 s), not the daily Baskatong clock.', '',
        '## Outcomes', '']
    for s in summaries:
        lines.append(f"- {s['scenario']}, seed {s['seed']}, {s['policy']}: **{s['status']}**, {s['executed_steps']}/{s['requested_steps']} periods; "
            f"maximum storage departure {s['max_storage_violation_m3']:.3g} m³; downstream excess {s['max_downstream_violation_m3_s']:.3g} m³/s; "
            f"final transit {s['final_transit_volume_m3']:.3g} m³; solver time {s['total_solver_seconds']:.4f} s.")
    lines += ['', '## Interpretation and limits', '',
        'The pulse and tight-capacity cases test anticipation and retaining water when downstream release space is restricted. '
        'The deliberately infeasible case tests refusal to return an action. These are controller-behavior experiments, '
        'not evidence of formal stability, recursive feasibility or guaranteed flood protection. No emergency policy is introduced.', '',
        'In this run, '+str(sum(s['complete'] for s in summaries if s['policy']=='mpc'))+' of '
        +str(sum(s['policy']=='mpc' for s in summaries))+' MPC rollouts complete. The tight one-period-delay case '
        'demonstrates that a feasible finite-horizon plan need not leave the next planning problem feasible. '
        'The less restrictive delayed case is included to demonstrate successful transit accounting.', '',
        'Noisy predicted feasibility does not imply realized storage feasibility. An actual storage departure is retained in the trace '
        'and terminates execution. Solver infeasibility also stops execution without an action. Reference violations are reported equally.', '',
        'Compare full objective only where both trajectories complete. `paired_comparison.csv` also reports stage costs over the common '
        'executed prefix, excluding premature terminal costs; these truncated comparisons do not establish whole-task superiority. '
        'The summary includes release variability, forecast error, numerical conservation, and predicted-feasible/realized-violation counts. '
        '`plans.json` contains every horizon, acceptance residual and failure reason; CSV traces and PNGs contain executed trajectories.', '',
        'Old observation/diagnostic bundle fingerprints were checked unchanged. The strict reconstruction and legacy soft follower are untouched.', '',
        '## Reproduce offline', '', '```sh',
        f'uv run --group reproduction python -m reservoir_control.systems.ottawa.hard_follower_study --forecast {mode} --output results/ottawa/hard_follower_v1_{mode}_rerun',
        '```', '', 'Use a new output directory; existing results are never overwritten.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print(pd.DataFrame(summaries)[['scenario','policy','seed','status','executed_steps']].to_string(index=False))
    print(f'Report: {output / "report.md"}')


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--forecast', choices=('perfect','noisy'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run_study(args.config.resolve(), args.output.resolve(), args.forecast)


if __name__ == '__main__': main()
