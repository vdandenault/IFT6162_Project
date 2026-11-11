"""Controlled Ontario MPC forecast, feasibility and objective diagnostics."""
from argparse import ArgumentParser
from copy import copy
from dataclasses import asdict, replace
import json
from math import isfinite
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from reservoir_control.controllers.mpc.receding_horizon import RecedingHorizonSearch
from .causal import CausalOntarioSession
from .data import BALANCE_CONVERSION, forcing_from_row, load_inputs
from .model import OntarioModel, Plan2014, ReleaseRequest
from .mpc import summarize
from .native import NativeReference, sha256
from .planning import MPCSettings, OntarioPlanningAdapter

YEARS = (1995, 2017, 2019)
OBJECTIVES = {'no_tracking': (0., 1.), 'no_terminal': (1., 0.), 'neither': (0., 0.)}


def oracle_decision(frame, index, decision):
    """Replace four inflow forecasts, retaining the current causal annual estimate."""
    future = frame.iloc[index:index + 4]
    if len(future) != 4:
        raise ValueError('Four realized future inflows are required for the oracle diagnostic')
    forecast = replace(
        decision.forecast, nts_m3_s=tuple(float(x) for x in future.ontNTS),
        slon_m3_s=float(future.stlouisontOut.iloc[0]),
        source='Oracle four-period Ontario supply and SLON diagnostic',
    )
    return replace(decision, forecast=forecast)


class OracleInflowAdapter(OntarioPlanningAdapter):
    """Privileged inflows enter both planning and the forecast-based M/F rules."""
    def __init__(self, model, settings, frame):
        super().__init__(model, settings)
        self.frame = frame

    def forecast_stages(self, decision, start):
        for lead, (estimate, time) in enumerate(super().forecast_stages(decision, start)):
            slon = float(self.frame.iloc[start.index + 48 + lead].stlouisontOut)
            yield replace(estimate, forecast=replace(estimate.forecast, slon_m3_s=slon)), time


def year_starts(model, frame):
    baseline = CausalOntarioSession(model, frame)
    rule = Plan2014(model.reference)
    starts = {}
    for year in YEARS:
        while baseline.information()[1].year < year:
            decision, time = baseline.information()
            baseline.step(rule.action(baseline.state, decision.forecast, time))
        starts[year] = copy(baseline)
        for _ in range(48):
            decision, time = baseline.information()
            baseline.step(rule.action(baseline.state, decision.forecast, time))
    return starts


def rollout(starts, frame, settings, arm):
    rows, snapshots = [], []
    oracle = arm == 'oracle_inflows'
    for year, start in starts.items():
        session = copy(start)
        adapter = (OracleInflowAdapter(session.model, settings, frame) if oracle
                   else OntarioPlanningAdapter(session.model, settings))
        search = RecedingHorizonSearch(adapter, beam_width=settings.beam_width)
        for _ in range(48):
            causal, time = session.information()
            decision = oracle_decision(frame, session.index, causal) if oracle else causal
            previous = session.state.previous_release_m3_s
            started = perf_counter()
            plan = search.plan(session.state, decision, time)
            if arm == 'persistence':
                snapshots.append((session.state, decision, time, plan))
            realized = frame.iloc[session.index]
            if oracle:
                result = session.model.transition(session.state, plan.actions[0], forcing_from_row(realized),
                                                  time, decision=decision)
                session.state = result.state
                session.index += 1
            else:
                result = session.step(plan.actions[0])
            release = result.metrics['executed_release_m3_s']
            if release != plan.steps[0].release_m3_s:
                raise RuntimeError('First predicted release differs from executed release')
            row = dict(
                arm=arm, policy='mpc', year=year, quarter_month=time.period,
                ontario_end_m=result.state.level_m,
                pointe_claire_m=result.metrics['pointe_claire_level_m'],
                release_m3_s=release, request_m3_s=plan.actions[0].m3_s,
                release_change_m3_s=abs(release - previous),
                plan_feasible=plan.feasible, plan_seconds=perf_counter() - started,
                predicted_max_excess_m=plan.max_excess_m,
                predicted_pointe_claire_m=plan.steps[0].pointe_claire_m,
                tracking_cost=plan.tracking_cost, terminal_cost=plan.terminal_cost,
                forecast_nts_m3_s=decision.forecast.nts_m3_s[0],
                forecast_slon_m3_s=decision.forecast.slon_m3_s,
                realized_nts_m3_s=float(realized.ontNTS),
                realized_slon_m3_s=float(realized.stlouisontOut),
            )
            for lead, step in enumerate(plan.steps, start=1):
                row[f'ontario_margin_qm{lead}_m'] = settings.ontario_max_m - step.state.level_m
                row[f'pointe_claire_margin_qm{lead}_m'] = settings.pointe_claire_max_m - step.pointe_claire_m
            rows.append(row)
    return pd.DataFrame(rows), snapshots


def release_scan(adapter, state, decision, time, grid_step=10.):
    """Scan requests and compute a necessary relaxed one-step cap intersection."""
    settings = adapter.settings
    rule = adapter.rule.action(state, decision.forecast, time)
    requests = sorted(set(np.arange(0., 11500. + grid_step / 2, grid_step)) | {rule.m3_s})
    rows = []
    for request in requests:
        predicted = adapter.forecast_step(state, ReleaseRequest(float(request)), decision, time)
        rows.append(dict(
            year=time.year, quarter_month=time.period, request_m3_s=request,
            release_m3_s=predicted.release_m3_s,
            ontario_end_m=predicted.state.level_m, pointe_claire_m=predicted.pointe_claire_m,
            feasible=(predicted.state.level_m <= settings.ontario_max_m + 1e-9
                      and predicted.pointe_claire_m <= settings.pointe_claire_max_m + 1e-9),
        ))
    grid = pd.DataFrame(rows)
    feasible = grid.loc[grid.feasible, 'release_m3_s']
    # Outward rounding allowances match native 6-decimal balance and 2-decimal stage.
    lower = decision.forecast.nts_m3_s[0] + (
        state.level_m - settings.ontario_max_m - 0.50001e-6
    ) * 10 * BALANCE_CONVERSION
    upper = 604. * (settings.pointe_claire_max_m + 0.0050001 - 16.57) ** (1 / .58)
    upper = upper / decision.pointe_claire_roughness - decision.forecast.slon_m3_s
    diagnostic = dict(
        year=time.year, quarter_month=time.period,
        necessary_release_min_m3_s=lower, necessary_release_max_m3_s=upper,
        relaxed_caps_conflict=bool(lower > upper),
        feasible_grid_requests=len(feasible),
        distinct_executed_releases=int(grid.release_m3_s.nunique()),
        minimum_executed_release_m3_s=float(grid.release_m3_s.min()),
        maximum_executed_release_m3_s=float(grid.release_m3_s.max()),
        feasible_release_min_m3_s=None if feasible.empty else float(feasible.min()),
        feasible_release_max_m3_s=None if feasible.empty else float(feasible.max()),
    )
    return diagnostic, grid


def common_candidate_audit(adapter, state, decision, time):
    """Enumerate the finite request tree and rank its leaves under four objectives."""
    problem = adapter.prepare(state, decision, time)
    weights = {'reference': (1., 1.), **OBJECTIVES}
    best, nodes, leaves = {}, 0, 0

    def rank(path, tracking, terminal):
        return (not path.feasible, 0. if path.feasible else path.max_excess_m,
                0. if path.feasible else path.total_excess_m,
                tracking * path.tracking_cost + terminal * path.terminal_cost)

    def visit(path, depth):
        nonlocal nodes, leaves
        if depth == len(problem.stages):
            leaves += 1
            for name, (tracking, terminal) in weights.items():
                if name not in best or rank(path, tracking, terminal) < rank(best[name], tracking, terminal):
                    best[name] = path
            return
        for child in adapter.expand(path, problem.stages[depth],
                                    final=depth + 1 == len(problem.stages), problem=problem):
            nodes += 1
            visit(child, depth + 1)

    visit(problem.initial_plan, 0)
    records = []
    for name, path in best.items():
        records.append(dict(
            year=time.year, quarter_month=time.period, objective=name,
            feasible=path.feasible, max_excess_m=path.max_excess_m,
            total_excess_m=path.total_excess_m,
            tracking_cost=path.tracking_cost, terminal_cost=path.terminal_cost,
            first_release_m3_s=path.steps[0].release_m3_s,
            expanded_nodes=nodes, complete_schedules=leaves,
        ))
    return records


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        return None
    return value


def save_json(path, value):
    path.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + '\n')


def run(output, checkout=None):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh diagnostic output directory')
    reference = NativeReference(checkout)
    settings = MPCSettings()
    frame = load_inputs(reference)
    sources = [Path(__file__), *[Path(__file__).with_name(name) for name in
               ('planning.py', 'model.py', 'causal.py', 'mpc.py', 'data.py', 'adapter.py', 'native.py')],
               Path(__file__).parents[2] / 'controllers/mpc/receding_horizon.py']
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / 'plan.json', dict(
        years=YEARS, settings=asdict(settings), horizon=4, request_grid_step_m3_s=10,
        wider_beam=96, objective_arms=OBJECTIVES,
        oracle='Four realized Ontario supplies and SLON values; causal annual estimate, ice and roughness retained',
        enumeration='All candidate schedules on every persistence-MPC fallback state; finite action tree only',
        source_sha256={str(path.resolve()): sha256(path) for path in sources},
        upstream=reference.provenance(),
    ))
    starts = year_starts(OntarioModel(reference), frame)
    arms, snapshots = {}, None
    for name, weights in [('persistence', (1., 1.)), ('oracle_inflows', (1., 1.)), *OBJECTIVES.items()]:
        config = replace(settings, tracking_weight=weights[0], terminal_weight=weights[1])
        rows, captured = rollout(starts, frame, config, name)
        rows.to_csv(output / f'{name}.csv', index=False)
        arms[name] = summarize(rows, settings)
        if name == 'persistence':
            snapshots = captured
        print(f'Completed annual arm {name}', flush=True)

    adapter = OntarioPlanningAdapter(starts[YEARS[0]].model, settings)
    wide = RecedingHorizonSearch(adapter, beam_width=96)
    scans, grids, search_rows, objective_rows, saved_states = [], [], [], [], []
    for state, decision, time, original in snapshots:
        scan, grid = release_scan(adapter, state, decision, time)
        scan['baseline_fallback'] = not original.feasible
        scan['original_first_release_m3_s'] = original.steps[0].release_m3_s
        scans.append(scan)
        if original.feasible:
            continue
        grids.append(grid)
        saved_states.append(dict(state=asdict(state), decision=asdict(decision), time=asdict(time)))
        larger = wide.plan(state, decision, time)
        search_rows.append(dict(
            year=time.year, quarter_month=time.period,
            reference_max_excess_m=original.max_excess_m,
            reference_total_excess_m=original.total_excess_m,
            wider_feasible=larger.feasible, wider_max_excess_m=larger.max_excess_m,
            wider_total_excess_m=larger.total_excess_m,
        ))
        objective_rows.extend(common_candidate_audit(adapter, state, decision, time))
        print(f'Completed fallback audit {time.year} QM{time.period}', flush=True)
    pd.DataFrame(scans).to_csv(output / 'action_bounds.csv', index=False)
    pd.concat(grids, ignore_index=True).to_csv(output / 'fallback_action_grid.csv', index=False)
    pd.DataFrame(search_rows).to_csv(output / 'search_comparison.csv', index=False)
    pd.DataFrame(objective_rows).to_csv(output / 'common_candidates.csv', index=False)
    save_json(output / 'fallback_states.json', saved_states)
    persistence = pd.read_csv(output / 'persistence.csv')
    changed_releases = {}
    for arm in OBJECTIVES:
        alternative = pd.read_csv(output / f'{arm}.csv')
        changed_releases[arm] = int((persistence.release_m3_s != alternative.release_m3_s).sum())
    baseline = pd.DataFrame(scans).loc[lambda rows: rows.baseline_fallback]
    best = pd.DataFrame(objective_rows).loc[lambda rows: rows.objective.eq('reference')]
    comparison = best.merge(pd.DataFrame(search_rows), on=['year', 'quarter_month'], validate='one_to_one')
    summary = dict(complete=True, annual_arms=arms, fallback_states=len(search_rows),
                   objective_release_changes=changed_releases,
                   fallback_categories_by_year={
                       str(year): dict(
                           fallback_states=len(group),
                           relaxed_first_step_conflicts=int(group.relaxed_caps_conflict.sum()),
                           first_step_grid_feasible=int((group.feasible_grid_requests > 0).sum()),
                           first_step_grid_infeasible_without_relaxed_conflict=int(
                               ((group.feasible_grid_requests == 0) & ~group.relaxed_caps_conflict).sum()),
                       ) for year, group in baseline.groupby('year')
                   },
                   enumerated_peak_improvements=int((comparison.max_excess_m < comparison.reference_max_excess_m - 1e-9).sum()),
                   first_step_cap_conflicts=sum(row['relaxed_caps_conflict'] for row in scans if row['baseline_fallback']),
                   fallbacks_with_feasible_first_grid_action=sum(row['feasible_grid_requests'] > 0 for row in scans if row['baseline_fallback']),
                   wider_search_feasible=sum(row['wider_feasible'] for row in search_rows),
                   full_candidate_search_feasible=sum(row['feasible'] for row in objective_rows if row['objective'] == 'reference'),
                   total_enumerated_schedules=sum(row['complete_schedules'] for row in objective_rows if row['objective'] == 'reference'))
    save_json(output / 'summary.json', summary)
    figures = plot_results(output, settings)
    save_json(output / 'manifest.json', dict(
        files_sha256={path.name: sha256(path) for path in output.iterdir() if path.is_file()},
        figures_sha256={str(path): sha256(path) for path in figures},
    ))
    return summary


def plot_results(output, settings):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    graphics = Path(__file__).resolve().parents[4] / 'background/images'
    graphics.mkdir(parents=True, exist_ok=True)
    bounds = pd.read_csv(output / 'action_bounds.csv')
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for ax, year in zip(axes, (2017, 2019)):
        selected = bounds.loc[bounds.year.eq(year)]
        ax.plot(selected.quarter_month, selected.necessary_release_min_m3_s, label='Minimum for Ontario study cap')
        ax.plot(selected.quarter_month, selected.necessary_release_max_m3_s, label='Maximum for Pointe-Claire study cap')
        ax.plot(selected.quarter_month, selected.original_first_release_m3_s, label='MPC selected release', color='black', alpha=.6)
        bad = selected.loc[selected.baseline_fallback]
        ax.scatter(bad.quarter_month, bad.original_first_release_m3_s, color='red', marker='x', label='Four-step fallback')
        ax.fill_between(selected.quarter_month, selected.necessary_release_min_m3_s,
                        selected.necessary_release_max_m3_s,
                        where=selected.relaxed_caps_conflict, color='red', alpha=.08)
        ax.set_title(str(year))
        ax.set_ylabel('Release (m³/s)')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=9)
    axes[-1].set_xlabel('Quarter-month')
    fig.suptitle('Forecast cap compatibility at the original MPC states')
    fig.tight_layout()
    fig.savefig(graphics / 'ontario_mpc_fallback_diagnosis.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for column, year in enumerate((2017, 2019)):
        for arm, label in [('persistence', 'Persistence MPC'), ('oracle_inflows', 'Oracle-inflow diagnostic')]:
            rows = pd.read_csv(output / f'{arm}.csv')
            rows = rows.loc[rows.year.eq(year)]
            axes[0, column].plot(rows.quarter_month, rows.ontario_end_m, label=label)
            axes[1, column].plot(rows.quarter_month, rows.pointe_claire_m, label=label)
        axes[0, column].axhline(settings.ontario_max_m, color='black', linestyle='--', label='Study cap')
        axes[1, column].axhline(settings.pointe_claire_max_m, color='black', linestyle='--')
        axes[0, column].set_title(str(year))
        axes[1, column].set_xlabel('Quarter-month')
        for ax in axes[:, column]:
            ax.grid(alpha=.2)
    axes[0, 0].set_ylabel('Ontario end level (m)')
    axes[1, 0].set_ylabel('Pointe-Claire level (m)')
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('Four-quarter-month inflow oracle with lagged ice and roughness')
    fig.tight_layout()
    fig.savefig(graphics / 'ontario_mpc_oracle_inflows.png', dpi=160)
    plt.close(fig)

    summary = json.loads((output / 'summary.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in zip(axes, ('ontario_end_m', 'pointe_claire_m'), ('Ontario', 'Pointe-Claire')):
        names = ['persistence', *OBJECTIVES]
        positions = np.arange(len(YEARS))
        for offset, arm in enumerate(names):
            values = [summary['annual_arms'][arm][str(year)]['mpc'][metric]['excess_m_periods'] for year in YEARS]
            ax.bar(positions + (offset - 1.5) * .19, values, .19, label=arm.replace('_', ' '))
        ax.set_xticks(positions, YEARS)
        ax.set_title(title)
        ax.set_ylabel('Exceedance (m·periods)')
    axes[0].legend(fontsize=9)
    fig.suptitle('Objective ablations with persistence forecasts')
    fig.tight_layout()
    fig.savefig(graphics / 'ontario_mpc_objective_diagnosis.png', dpi=160)
    plt.close(fig)
    return [graphics / name for name in ('ontario_mpc_fallback_diagnosis.png',
            'ontario_mpc_oracle_inflows.png', 'ontario_mpc_objective_diagnosis.png')]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkout', type=Path)
    args = parser.parse_args()
    summary = run(args.output, args.checkout)
    print(json.dumps({key: value for key, value in summary.items() if key != 'annual_arms'}, indent=2))


if __name__ == '__main__':
    main()
