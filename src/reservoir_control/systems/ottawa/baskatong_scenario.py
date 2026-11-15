"""Uncalibrated Baskatong daily-mean scenario model, separate from reconstruction.

Offline only. Effective input is reconstructed once from observations for each
storage mapping and discharge proxy, then held fixed across release experiments.
"""
from argparse import ArgumentParser
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .historical_pilot import extract
from .discharge_comparison import snapshot
from .reconstruction import ROOT, digest

CONFIG = ROOT/'configs/studies/ottawa/baskatong_1970_scenario.json'


@dataclass(frozen=True)
class LinearStorage:
    effective_area_m2: float
    h_ref_m: float

    def __post_init__(self):
        if not np.isfinite([self.effective_area_m2, self.h_ref_m]).all() or self.effective_area_m2 <= 0:
            raise ValueError('Positive finite effective area and finite storage origin required')

    def storage(self, level_m): return self.effective_area_m2*(np.asarray(level_m)-self.h_ref_m)
    def level(self, storage_m3): return self.h_ref_m+np.asarray(storage_m3)/self.effective_area_m2

    def advance(self, storage_m3, effective_input_m3_s, release_m3_s, dt):
        if not np.isfinite([storage_m3, effective_input_m3_s, release_m3_s, dt]).all() or dt <= 0 or release_m3_s < 0:
            raise ValueError('Finite state/input, nonnegative release and positive duration required')
        # Negative effective input is allowed; storage and level are never clipped.
        return storage_m3+dt*(effective_input_m3_s-release_m3_s)


def validate(c):
    if c.get('schema_version') != 1: raise ValueError('Unsupported scenario configuration')
    t, b, p, s = (c[k] for k in ('timing_assumption', 'experimental_bounds_assumed',
                                 'prescribed_release_experiment', 'linear_storage_assumption'))
    if t['duration_seconds'] != 86400: raise ValueError('This scenario explicitly uses nominal 86400-second days')
    if (not np.isfinite(list(b[k] for k in ('level_minimum_m', 'level_maximum_m', 'release_minimum_m3_s', 'release_maximum_m3_s'))).all()
            or b['level_minimum_m'] >= b['level_maximum_m']
            or not 0 <= b['release_minimum_m3_s'] < b['release_maximum_m3_s']):
        raise ValueError('Ordered finite experimental level/outlet bounds required')
    if (not np.isfinite([p['reduction_fraction'], p['maximum_additional_release_m3_s'],
                        s['h_ref_m'], s['inventory_anchor_m3'], c['volume_accounting_tolerance_m3']]).all()
            or not 0 <= p['reduction_fraction'] <= 1 or p['maximum_additional_release_m3_s'] <= 0
            or s['inventory_anchor_m3'] <= 0 or c['volume_accounting_tolerance_m3'] <= 0):
        raise ValueError('Invalid finite scenario parameters')
    dates = [pd.Timestamp(p[k]) for k in ('retention_start_inclusive', 'retention_end_exclusive',
                                         'repayment_start_inclusive', 'repayment_end_exclusive')]
    if not dates[0] < dates[1] <= dates[2] < dates[3]: raise ValueError('Retention must precede a disjoint repayment window')
    if not s['scales'] or len({a['id'] for a in s['scales']}) != len(s['scales']):
        raise ValueError('Distinct storage scenario IDs required')
    for scale in s['scales']:
        a = scale.get('effective_area_m2')
        if a is None: a = scale['anchor_multiplier']
        if not np.isfinite(a) or a <= 0: raise ValueError('Positive area or volume-anchor multiplier required')
    return c


def has_flag(value): return not pd.isna(value) and str(value).strip() != ''


def align_centers(level_records, discharge_records, c):
    """No timezone assertion: date labels become assumed model-day centers."""
    o = c['observed_inputs']; dt = c['timing_assumption']['duration_seconds']
    dates = pd.date_range(o['first_date'], o['last_interval_start_date'], freq='D')
    if o['following_context_date'] != (dates[-1]+pd.Timedelta(days=1)).strftime('%Y-%m-%d'):
        raise ValueError('Following-date context must immediately follow the final start date')
    if level_records.DATE.duplicated().any() or discharge_records.DATE.duplicated().any():
        raise ValueError('Duplicate daily labels cannot be silently resolved')
    levels, flows = level_records.set_index('DATE'), discharge_records.set_index('DATE')
    def field(frame, date, name): return frame.at[date, name] if date in frame.index else np.nan
    records = []
    for day in dates:
        left, right = day.strftime('%Y-%m-%d'), (day+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        h0, h1 = [field(levels, date, 'LEVEL') for date in (left, right)]
        q0, q1 = [field(flows, date, 'DISCHARGE') for date in (left, right)]
        flags = {f'{kind}_symbol_{side}': field(frame, date, column)
            for kind, frame, column in [('level', levels, 'LEVEL_SYMBOL_EN'), ('discharge', flows, 'DISCHARGE_SYMBOL_EN')]
            for side, date in [('left', left), ('right', right)]}
        source_flagged = any(has_flag(v) for v in flags.values())
        valid = np.isfinite([h0, h1, q0, q1]).all() and min(q0, q1) >= 0
        records.append(dict(date=left, next_date=right,
            model_interval_start=left+'T12:00:00', model_interval_end=right+'T12:00:00',
            duration_seconds=dt, observed_level_start_m=h0, observed_level_end_m=h1,
            proxy_discharge_left_m3_s=q0, proxy_discharge_right_m3_s=q1,
            proxy_interval_release_m3_s=(q0+q1)*.5 if np.isfinite([q0, q1]).all() else np.nan,
            input_available=bool(valid), source_flagged=source_flagged,
            partial_day_affected=any(str(v) == 'Partial Day' for v in flags.values()),
            uncertainty_flags=';'.join(k+':'+str(v) for k, v in flags.items() if has_flag(v)), **flags))
    return pd.DataFrame(records)


def effective_inputs(intervals, mapping):
    out = intervals.copy()
    out['observed_scenario_storage_start_m3'] = mapping.storage(out.observed_level_start_m)
    out['observed_scenario_storage_end_m3'] = mapping.storage(out.observed_level_end_m)
    eff = ((out.observed_scenario_storage_end_m3-out.observed_scenario_storage_start_m3)
           /out.duration_seconds+out.proxy_interval_release_m3_s)
    out['effective_net_input_m3_s'] = eff.where(out.input_available)
    out['negative_effective_input'] = out.effective_net_input_m3_s < 0
    return out


def prescribe(intervals, c):
    """Bounded prescribed requests; no optimization or storage safety correction."""
    p, b = c['prescribed_release_experiment'], c['experimental_bounds_assumed']
    debt, retained, paid = 0., 0., 0.
    rows = []
    for row in intervals.itertuples():
        q, dt = row.proxy_interval_release_m3_s, row.duration_seconds
        finite = np.isfinite(q)
        proxy_feasible = finite and b['release_minimum_m3_s'] <= q <= b['release_maximum_m3_s']
        release, reduce_volume, pay_volume = np.nan, 0., 0.
        if proxy_feasible:
            release = q
            if p['retention_start_inclusive'] <= row.date < p['retention_end_exclusive']:
                reduce = min(p['reduction_fraction']*q, q-b['release_minimum_m3_s'])
                release = q-reduce; reduce_volume = reduce*dt
                retained += reduce_volume; debt += reduce_volume
            elif p['repayment_start_inclusive'] <= row.date < p['repayment_end_exclusive']:
                extra = min(p['maximum_additional_release_m3_s'], b['release_maximum_m3_s']-q, debt/dt)
                release = q+extra; pay_volume = extra*dt
                paid += pay_volume; debt -= pay_volume
                if abs(debt) <= c['volume_accounting_tolerance_m3']: debt = 0.
        rows.append(dict(experiment_release_m3_s=release, reference_release_m3_s=q,
            reference_outlet_feasible=proxy_feasible, retained_interval_m3=reduce_volume,
            repaid_interval_m3=pay_volume, cumulative_retained_m3=retained,
            cumulative_repaid_m3=paid, outstanding_retained_m3=debt))
    return pd.DataFrame(rows, index=intervals.index)


def simulate(inputs, mapping, release, c):
    b = c['experimental_bounds_assumed']
    state = float(mapping.storage(inputs.observed_level_start_m.iloc[0]))
    records = []; flagged_influence = False
    for (_, row), q in zip(inputs.iterrows(), release, strict=True):
        start = state
        outlet_bad = np.isfinite(q) and not b['release_minimum_m3_s'] <= q <= b['release_maximum_m3_s']
        valid = bool(row.input_available) and np.isfinite([state, row.effective_net_input_m3_s, q]).all() and not outlet_bad
        following = mapping.advance(state, row.effective_net_input_m3_s, q, row.duration_seconds) if valid else np.nan
        level0, level1 = float(mapping.level(start)), float(mapping.level(following))
        residual = following-start-row.duration_seconds*(row.effective_net_input_m3_s-q) if valid else np.nan
        flagged_influence = flagged_influence or bool(row.source_flagged)
        high = max(0., max(level0, level1)-b['level_maximum_m']) if np.isfinite([level0, level1]).all() else np.nan
        low = max(0., b['level_minimum_m']-min(level0, level1)) if np.isfinite([level0, level1]).all() else np.nan
        records.append(dict(date=row.date, model_interval_start=row.model_interval_start,
            model_interval_end=row.model_interval_end, duration_seconds=row.duration_seconds,
            observed_level_start_m=row.observed_level_start_m, observed_level_end_m=row.observed_level_end_m,
            effective_net_input_m3_s=row.effective_net_input_m3_s, release_m3_s=q,
            storage_start_m3=start, storage_end_m3=following, level_start_m=level0, level_end_m=level1,
            mass_balance_error_m3=residual, continuous_state_available=bool(valid),
            release_bound_violation=bool(outlet_bad), level_upper_excess_m=high, level_lower_excess_m=low,
            storage_minimum_assumed_m3=float(mapping.storage(b['level_minimum_m'])),
            storage_maximum_assumed_m3=float(mapping.storage(b['level_maximum_m'])),
            storage_upper_excess_m3=high*mapping.effective_area_m2,
            storage_lower_excess_m3=low*mapping.effective_area_m2,
            source_flagged=bool(row.source_flagged), uncertainty_flags=row.uncertainty_flags,
            state_depends_on_flagged_input=flagged_influence))
        state = following  # Never reset to an observed level after a missing interval.
    return pd.DataFrame(records)


def negative_diagnostics(inputs, clean_only):
    rows = inputs[(~inputs.source_flagged) if clean_only else np.ones(len(inputs), bool)]
    finite = rows.effective_net_input_m3_s.dropna()
    negative = rows[rows.effective_net_input_m3_s < 0]
    return dict(subset='unflagged_intervals_only' if clean_only else 'all_reported_including_uncertain',
        intervals=len(rows), available_intervals=len(finite), negative_input_intervals=len(negative),
        negative_input_fraction=float(len(negative)/len(finite)) if len(finite) else None,
        minimum_effective_net_input_m3_s=float(finite.min()) if len(finite) else None,
        negative_input_volume_magnitude_m3=float((-negative.effective_net_input_m3_s*negative.duration_seconds).sum()))


def evaluate_case(inputs, mapping, ledger, c):
    replay = simulate(inputs, mapping, inputs.proxy_interval_release_m3_s, c)
    alternative = simulate(inputs, mapping, ledger.experiment_release_m3_s, c)
    if not np.array_equal(replay.effective_net_input_m3_s, alternative.effective_net_input_m3_s, equal_nan=True):
        raise RuntimeError('Effective input changed between experiments')
    error = (replay.level_end_m-replay.observed_level_end_m).abs()
    delta_storage = alternative.storage_end_m3-replay.storage_end_m3
    accounting_error = (delta_storage-ledger.outstanding_retained_m3).abs()
    valid = alternative.continuous_state_available
    complete = bool(valid.all())
    final = ledger.iloc[-1]; tolerance = c['volume_accounting_tolerance_m3']
    high = float(alternative.level_upper_excess_m.max()) if alternative.level_upper_excess_m.notna().any() else None
    low = float(alternative.level_lower_excess_m.max()) if alternative.level_lower_excess_m.notna().any() else None
    feasible = complete and bool(ledger.reference_outlet_feasible.all()) and high <= 1e-9 and low <= 1e-9
    summary = dict(available_input_intervals=int(inputs.input_available.sum()),
        flagged_input_intervals=int(inputs.source_flagged.sum()), partial_day_intervals=int(inputs.partial_day_affected.sum()),
        replay_max_level_error_m=float(error.max()) if error.notna().any() else None,
        replay_unflagged_max_level_error_m=float(error[~inputs.source_flagged].max()) if error[~inputs.source_flagged].notna().any() else None,
        replay_complete=bool(replay.continuous_state_available.all()),
        replay_level_violation_intervals=int(((replay.level_upper_excess_m > 1e-9) | (replay.level_lower_excess_m > 1e-9)).sum()),
        replay_outlet_violation_intervals=int(replay.release_bound_violation.sum()),
        alternative_complete=complete, alternative_feasible_under_assumptions=feasible,
        alternative_level_violation_intervals=int(((alternative.level_upper_excess_m > 1e-9) | (alternative.level_lower_excess_m > 1e-9)).sum()),
        alternative_upper_excess_peak_m=high, alternative_lower_excess_peak_m=low,
        alternative_peak_level_m=float(alternative.level_end_m.max()) if valid.any() else None,
        max_level_increase_m=float((alternative.level_end_m-replay.level_end_m).max()) if valid.any() else None,
        retained_volume_m3=float(final.cumulative_retained_m3), repaid_volume_m3=float(final.cumulative_repaid_m3),
        remaining_unreleased_volume_m3=float(final.outstanding_retained_m3),
        release_totals_match=complete and abs(final.outstanding_retained_m3) <= tolerance,
        reference_release_volume_m3=float((inputs.proxy_interval_release_m3_s*inputs.duration_seconds).sum()) if inputs.input_available.all() else None,
        alternative_release_volume_m3=float((ledger.experiment_release_m3_s*inputs.duration_seconds).sum()) if complete else None,
        retained_storage_ledger_max_error_m3=float(accounting_error.max()) if accounting_error.notna().any() else None,
        max_mass_balance_error_m3=float(pd.concat([replay.mass_balance_error_m3, alternative.mass_balance_error_m3]).abs().max()) if valid.any() else None,
        final_level_difference_m=float(alternative.level_end_m.iloc[-1]-replay.level_end_m.iloc[-1]) if complete else None)
    if complete:
        volume_difference = summary['reference_release_volume_m3']-summary['alternative_release_volume_m3']
        summary['release_volume_ledger_error_m3'] = volume_difference-summary['remaining_unreleased_volume_m3']
        if abs(summary['release_volume_ledger_error_m3']) > tolerance or summary['retained_storage_ledger_max_error_m3'] > tolerance:
            raise RuntimeError('Retained-volume accounting failed')
    return replay, alternative, summary


def figures(all_inputs, trajectories, ledgers, summaries, c, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    palette = ['#cd722b', '#287ea5', '#6b4fa2', '#319269']
    bounds = c['experimental_bounds_assumed']
    for source in all_inputs.source.unique():
        subset = summaries[summaries.source == source]
        fig, axes = plt.subplots(4, 1, figsize=(13, 13), sharex=True, layout='constrained')
        first_case = subset.case_id.iloc[0]
        first = all_inputs[all_inputs.case_id == first_case]
        dates = pd.to_datetime(first.model_interval_end)
        axes[0].plot(dates, first.observed_level_end_m, color='black', linewidth=2., label='Observed mean, placed at model center')
        for color, row in zip(palette, subset.itertuples()):
            candidate = trajectories[(trajectories.case_id == row.case_id) & (trajectories.experiment == 'retain_then_release')]
            status = 'feasible' if row.alternative_feasible_under_assumptions else 'INFEASIBLE'
            axes[0].plot(dates, candidate.level_end_m, color=color, label=f'{row.scale}: assumed K={row.implied_K_m3/1e9:.5g} billion m³ ({status})')
            data = all_inputs[all_inputs.case_id == row.case_id]
            axes[2].plot(pd.to_datetime(data.model_interval_start), data.effective_net_input_m3_s,
                         color=color, linewidth=1., label=row.scale)
        replay = trajectories[(trajectories.case_id == first_case) & (trajectories.experiment == 'historical_proxy_replay')]
        axes[0].plot(dates, replay.level_end_m, '--', color='#20936b', linewidth=1.3, label='Proxy replay (accounting check)')
        axes[0].axhline(bounds['level_maximum_m'], color='#a34e49', linestyle=':', label='Assumed upper experiment bound')
        axes[0].axhline(bounds['level_minimum_m'], color='#a34e49', linestyle=':')
        axes[0].set_ylabel('Scenario / observed level (m)\nhistorical station datum')
        axes[0].legend(fontsize=8, loc='upper left')
        ledger = ledgers[ledgers.case_id == first_case]
        start_dates = pd.to_datetime(first.model_interval_start)
        axes[1].step(start_dates, ledger.reference_release_m3_s, where='post', label='Historical proxy, adjacent-day average', color='black')
        axes[1].step(start_dates, ledger.experiment_release_m3_s, where='post', label='Prescribed retain then release', color='#287ea5')
        axes[1].set_ylabel('Modeled interval release (m³/s)'); axes[1].legend(fontsize=9)
        axes[2].axhline(0, color='gray', linewidth=.8); axes[2].set_ylabel('Effective net input (m³/s)\nnot measured/natural inflow')
        axes[2].legend(fontsize=9)
        flagged = first.source_flagged.to_numpy()
        # Mark affected intervals, not inferred gaps or altered observations.
        if flagged.any():
            axes[2].scatter(start_dates[flagged], first.effective_net_input_m3_s.to_numpy()[flagged],
                            s=15, facecolors='none', edgecolors='black', label='Source-flag-affected interval', zorder=4)
            axes[2].legend(fontsize=8)
        for key, label, color in [('cumulative_retained_m3', 'Cumulative retained', '#287ea5'),
                                  ('cumulative_repaid_m3', 'Cumulative repaid', '#329269'),
                                  ('outstanding_retained_m3', 'Still to release', '#cd722b')]:
            axes[3].plot(start_dates, ledger[key]/1e6, label=label, color=color)
        axes[3].set_ylabel('Scenario volume (million m³)'); axes[3].legend(fontsize=9)
        for ax in axes:
            ax.grid(alpha=.2)
            locator = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(locator); ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        fig.suptitle(f'Baskatong 1970 — {source} discharge interpretation\nUncalibrated daily-center scenario; flagged values retained as uncertain')
        fig.supxlabel('Abstract model-day centers; no historical timezone assertion')
        fig.savefig(output/f'{source}_release_scenarios.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
    for source in summaries.source.unique():
        rows = summaries[summaries.source == source].sort_values('effective_area_m2')
        for ax, column, ylabel in zip(axes, ['max_level_increase_m', 'negative_input_intervals', 'alternative_upper_excess_peak_m'],
            ['Maximum level increase (m)', 'Negative effective-input intervals', 'Assumed upper-bound exceedance (m)']):
            ax.plot(rows.effective_area_m2/1e6, rows[column], 'o-', label=source)
            ax.set(xlabel='Assumed effective area (million m²)', ylabel=ylabel); ax.grid(alpha=.2)
    axes[0].legend(fontsize=9)
    fig.suptitle('Sensitivity to assumed storage scale and discharge interpretation\nAlternative effective inputs are reconstructed separately for each case')
    fig.savefig(output/'scenario_sensitivity.png', dpi=150); plt.close(fig)


def write_report(rows, diagnostics, resolved, c):
    b, p, storage = c['experimental_bounds_assumed'], c['prescribed_release_experiment'], c['linear_storage_assumption']
    text = f"""# Baskatong 1970: approximate release-response scenarios

**Uncalibrated research scenarios; no Ontario controller or network changes.**

## Observed facts

Saved ECCC daily mean levels come from 02LG007; 02LG006 is the primary Mercier
flow proxy and 02LG007 discharge is a separate sensitivity. Neither is verified
as the complete Baskatong outlet. No blending, transfer interpretation or
independent validation is claimed. The January 1, 1971 context mean is available.
Source files and earlier reports remain unchanged.

## Assumed model

- `S=A_eff*(h-h_ref)`; inverse `h=h_ref+S/A_eff`. Origin: **{storage['h_ref_m']:g} m**
  in the historical arbitrary datum, not total geological volume.
- Daily means are placed at **noon** on an abstract model calendar. Use nominal
  **86400-second** intervals and the average of adjacent daily mean flow values.
  The window is January 1, 1970 noon to January 1, 1971 noon, not verified clocks
  or calendar-year midnight boundaries. Missing inputs are not interpolated.
- Default area scales use assumed volume changes of 0.5×, 1× and 2× the **{storage['inventory_anchor_m3']/1e9:g} billion m³**
  order-of-magnitude inventory anchor. **The anchor is not verified usable storage over
  the observed range.** Resolved areas/volumes are in `resolved_scenarios.json`.
  Explicit area overrides are supported; no case is calibrated.
- Experimental levels: **{b['level_minimum_m']:g}–{b['level_maximum_m']:g} m**;
  outlet: **{b['release_minimum_m3_s']:g}–{b['release_maximum_m3_s']:g} m³/s**.
  These are separate assumptions, not observed extrema or modern operating limits.
  Storage/level is never clipped; violations mark the experiment infeasible.

Each case reconstructs **effective net input**, `i_eff=ΔS_obs/dt+q_proxy`, once,
then holds it fixed across release experiments. It can absorb upstream regulation,
proxy/timing/storage errors and omitted outlets; it is not measured/natural inflow.
Negative values are retained. Either adjacent source flag propagates to an interval.
Flagged reported values are used as uncertain inputs, not corrected or replaced.

## Accounting checks

Historical-proxy replay recovers the placed observed levels where executable.
This is an **accounting identity**, not physical validation. Maximum replay level
error is **{rows.replay_max_level_error_m.max():.3g} m**; per-step conservation residual
**{rows.max_mass_balance_error_m3.max():.3g} m³**; retained-volume versus storage-difference
residual **{rows.retained_storage_ledger_max_error_m3.max():.3g} m³**. Missing intervals never
reset simulated state to observations.

## Prescribed counterfactual

Reduce reference release by **{100*p['reduction_fraction']:g}%** on
**[{p['retention_start_inclusive']}, {p['retention_end_exclusive']})**. Repay only
retained water on **[{p['repayment_start_inclusive']}, {p['repayment_end_exclusive']})**,
with at most **{p['maximum_additional_release_m3_s']:g} m³/s** extra and within outlet bounds.
Windows select start-day centers. This historical schedule is an **offline reference**,
not a causal forecast or optimized controller.

| Proxy / scale | A_eff (million m²) | Max level increase (m) | Upper-bound excess (m) | Feasible? |
|---|---:|---:|---:|---|
"""
    def shown(value): return 'unavailable' if pd.isna(value) else f'{value:.3f}'
    for r in rows.itertuples():
        text += f'| {r.source} / {r.scale} | {r.effective_area_m2/1e6:.3f} | {shown(r.max_level_increase_m)} | {shown(r.alternative_upper_excess_peak_m)} | {r.alternative_feasible_under_assumptions} |\n'
    text += '\nModeled volume accounting (same across areas for each proxy):\n\n'
    for source, group in rows.groupby('source'):
        r = group.iloc[0]
        text += (f'- **{source}:** retained {r.retained_volume_m3/1e6:.3f}, repaid {r.repaid_volume_m3/1e6:.3f} '
                 f'million m³; outstanding {r.remaining_unreleased_volume_m3:.6f} m³. '
                 f'Totals matched within tolerance: **{bool(r.release_totals_match)}**.\n')
    text += '\nNegative effective inputs: all / unflagged-only interval counts and minima:\n\n'
    for r in rows.itertuples():
        pair = diagnostics[diagnostics.case_id == r.case_id].set_index('subset')
        a, u = pair.loc['all_reported_including_uncertain'], pair.loc['unflagged_intervals_only']
        text += (f'- **{r.case_id}:** {int(a.negative_input_intervals)}/{int(a.available_intervals)} / '
                 f'{int(u.negative_input_intervals)}/{int(u.available_intervals)}; '
                 f'{a.minimum_effective_net_input_m3_s:.2f} / {u.minimum_effective_net_input_m3_s:.2f} m³/s.\n')
    unflagged_negative = diagnostics.loc[diagnostics.subset == 'unflagged_intervals_only', 'negative_input_intervals'].sum()
    negative_note = ('Negative inputs also occur among unflagged intervals in this run.'
                     if unflagged_negative else 'No negative inputs occur among unflagged intervals in this run.')
    text += f"""
## Robust conclusions and unresolved interpretation

Retaining water raises level; the offset equals outstanding retained volume
**divided by assumed area**. Full repayment returns to the corresponding replay.
Those are structural model consequences. Height change, effective input and
bounds feasibility depend strongly on area; changing discharge source also changes
the effective input and reference volume. Equal total releases do not imply safe
levels. {negative_note}

The 28 Partial Day dates at 02LG006 affect 29 adjacent-center intervals; 02LG007
has no source flow flags. Excluding intervals from diagnostics is **not** a clean
re-simulation: later states can depend on earlier flagged input. No independent
validation is supplied by the alternative station. Real storage mapping, timing,
outlet completeness, routing and input validation remain unresolved. No Lake St.
Louis levels are predicted.

## Outputs and offline command

Configuration/provenance, resolved scales, `interval_inputs.csv`, `trajectories.csv`,
`release_accounting.csv`, `case_summary.csv` and `input_diagnostics.csv` retain all
assumptions, flags, modeled volumes and violations. Source-specific PNGs show
levels/releases/effective inputs/volume accounting; a third PNG compares scales.

```bash
uv run python -m reservoir_control.systems.ottawa.baskatong_scenario \\
  --pilot results/ottawa/baskatong_1970_observed_v1 \\
  --output results/ottawa/baskatong_scenario_review
```

Use `--config /path/to/new/scenario.json` with a fresh output folder to vary areas,
bounds or perturbations. No network calls or data-owner contact occur. Data:
Environment and Climate Change Canada HYDAT / MSC GeoMet and contributing agencies
as recorded in the preserved original manifest/station metadata.
"""
    return text


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, default=ROOT/'results/ottawa/baskatong_1970_observed_v1')
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.pilot.resolve()): raise ValueError('Scenario output must be outside the preserved observation bundle')
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError('Use a fresh scenario output directory')
    c = validate(json.loads(args.config.read_text()))
    before = snapshot(args.pilot)
    manifest = json.loads((args.pilot/'raw/manifest.json').read_text())
    for name, record in manifest['files'].items():
        if digest(args.pilot/'raw'/name) != record['sha256']: raise ValueError('Raw source checksum mismatch')
    original = json.loads((args.pilot/'provenance.json').read_text())
    if any(digest(args.pilot/name) != value for name, value in original['output_sha256'].items()):
        raise ValueError('Original pilot output checksum mismatch')
    o = c['observed_inputs']; s = c['linear_storage_assumption']
    level_records = extract(json.loads((args.pilot/'raw'/f"{o['level_station']}_daily.json").read_text()), o['level_station'])
    annual = level_records[(level_records.DATE >= o['first_date']) & (level_records.DATE <= o['last_interval_start_date'])]
    level_span = float(annual.LEVEL.max()-annual.LEVEL.min())
    if not np.isfinite(level_span) or level_span <= 0: raise ValueError('Positive observed annual level range required')
    args.output.mkdir(parents=True)
    (args.output/'scenario_config.json').write_text(json.dumps(c, indent=2, ensure_ascii=False)+'\n')
    resolved, summaries, inputs_all, trajectories, accounts, diagnostics = [], [], [], [], [], []
    for source in (o['primary_discharge_station'], o['sensitivity_discharge_station']):
        flows = extract(json.loads((args.pilot/'raw'/f'{source}_daily.json').read_text()), source)
        intervals = align_centers(level_records, flows, c)
        ledger = prescribe(intervals, c)
        for scale in s['scales']:
            area = scale['effective_area_m2']
            if area is None: area = s['inventory_anchor_m3']*scale['anchor_multiplier']/level_span
            mapping = LinearStorage(area, s['h_ref_m'])
            case_id = source+'__'+scale['id']
            case = dict(case_id=case_id, source=source, scale=scale['id'], effective_area_m2=area,
                        implied_K_m3=area*level_span, observed_level_range_m=level_span, h_ref_m=mapping.h_ref_m)
            inputs = effective_inputs(intervals, mapping)
            replay, alternative, result = evaluate_case(inputs, mapping, ledger, c)
            negative = negative_diagnostics(inputs, False)
            summaries.append(case | result | {k: v for k, v in negative.items() if k.startswith('negative_')})
            resolved.append(case)
            inputs_all.append(inputs.assign(case_id=case_id, source=source, scale=scale['id']))
            accounts.append(ledger.assign(date=inputs.date, model_interval_start=inputs.model_interval_start, model_interval_end=inputs.model_interval_end, duration_seconds=inputs.duration_seconds, case_id=case_id, source=source, scale=scale['id']))
            trajectories += [replay.assign(case_id=case_id, source=source, scale=scale['id'], experiment='historical_proxy_replay'),
                             alternative.assign(case_id=case_id, source=source, scale=scale['id'], experiment='retain_then_release')]
            diagnostics += [dict(case_id=case_id, **negative_diagnostics(inputs, clean)) for clean in (False, True)]
    inputs_all, trajectories, accounts = [pd.concat(x, ignore_index=True) for x in (inputs_all, trajectories, accounts)]
    summaries, diagnostics = pd.DataFrame(summaries), pd.DataFrame(diagnostics)
    for name, frame in [('interval_inputs', inputs_all), ('trajectories', trajectories),
                        ('release_accounting', accounts), ('case_summary', summaries), ('input_diagnostics', diagnostics)]:
        frame.to_csv(args.output/f'{name}.csv', index=False)
    (args.output/'resolved_scenarios.json').write_text(json.dumps(resolved, indent=2, allow_nan=False)+'\n')
    (args.output/'summary.json').write_text(json.dumps(dict(cases=summaries.to_dict('records'), independent_validation=False,
        accounting_check_only=True, source_files_unchanged=before == snapshot(args.pilot)), indent=2, allow_nan=False)+'\n')
    figures(inputs_all, trajectories, accounts, summaries, c, args.output)
    (args.output/'report.md').write_text(write_report(summaries, diagnostics, resolved, c))
    if before != snapshot(args.pilot): raise RuntimeError('Original observations changed')
    protected = [Path(__file__).with_name(name) for name in ('model.py', 'follower.py', 'reconstruction.py', 'historical_pilot.py')]
    provenance = dict(original_pilot_sha256=before, original_pilot_unchanged=True,
        implementation_sha256=digest(__file__), config_sha256=digest(args.config),
        unchanged_dependency_sha256={str(p.relative_to(ROOT)): digest(p) for p in protected},
        output_sha256=snapshot(args.output), network_access=False)
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(summaries[['case_id', 'max_level_increase_m', 'alternative_upper_excess_peak_m',
                     'alternative_feasible_under_assumptions', 'negative_input_intervals', 'remaining_unreleased_volume_m3']].to_string(index=False))
    print(f'Report: {args.output / "report.md"}')


if __name__ == '__main__': main()
