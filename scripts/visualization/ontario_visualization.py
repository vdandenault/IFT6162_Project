"""Build the local Ontario experiment playback and controller PNG diagrams."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INK = '#18354a'
BLUE = '#2784ae'
GREEN = '#24896c'
PURPLE = '#8464a7'


def figure(title, subtitle):
    fig, ax = plt.subplots(figsize=(16, 10), dpi=150)
    fig.patch.set_facecolor('#f5f8fc')
    ax.set(xlim=(0, 16), ylim=(0, 10))
    ax.axis('off')
    fig.subplots_adjust(left=.02, right=.98, top=.98, bottom=.02)
    ax.text(.35, 9.35, title, fontsize=25, color=INK, weight='bold')
    ax.text(.35, 8.8, subtitle, fontsize=13, color='#566d80')
    return fig, ax


def box(ax, x, y, width, height, title, text, color):
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle='round,pad=0.12',
                              facecolor='white', edgecolor=color, linewidth=2))
    ax.text(x+.18, y+height-.3, title, va='top', fontsize=16, color=color, weight='bold')
    ax.text(x+.18, y+height-.9, text, va='top', fontsize=13, color=INK, linespacing=1.6)


def arrow(ax, start, end, color=BLUE):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='-|>', color=color, lw=2, mutation_scale=19))


def diagrams(destination):
    destination.mkdir(parents=True, exist_ok=True)
    fig, ax = figure('Ontario MPC: plan releases, execute one, then replan',
                     'Implemented pilot · 48 quarter-month decisions per year · four-period horizon')
    box(ax, .45, 5.3, 4.0, 2.7, 'OBSERVE & FORECAST',
        'Ontario level + previous release\nPast Ontario supply and SLON\nLast observed ice and roughness\nPersist supply / SLON four periods', BLUE)
    box(ax, 5.6, 5.3, 4.3, 2.7, 'PLAN ON COPIED STATES',
        'Try four-release schedules\nApply Bv7 release limits\nPredict Ontario and Pointe-Claire\nKeep a beam of 12 schedules', PURPLE)
    box(ax, 11.1, 5.3, 4.35, 2.7, 'EXECUTE FIRST RELEASE',
        'Hold release for one quarter-month\nRealized supply updates Ontario\nOntario outflow + SLON route\nto Lake St. Louis / Pointe-Claire', GREEN)
    arrow(ax, (4.62, 6.65), (5.4, 6.65))
    arrow(ax, (10.08, 6.65), (10.9, 6.65), GREEN)
    box(ax, 5.6, 1.65, 4.3, 2.55, 'RANK THE SCHEDULES',
        'First seek both study caps\nIf none found: peak, then total excess\nThen rule-release tracking cost\n+ forecast terminal-level tracking', PURPLE)
    arrow(ax, (7.75, 5.13), (7.75, 4.38), PURPLE)
    ax.annotate('', xy=(11.0, 5.65), xytext=(10.0, 2.95),
                arrowprops=dict(arrowstyle='-|>', color=PURPLE, lw=2,
                                connectionstyle='angle,angleA=0,angleB=-90,rad=15'))
    ax.plot([13.3, 13.3, 2.45, 2.45], [8.15, 8.45, 8.45, 8.15], color=GREEN, lw=2)
    arrow(ax, (2.45, 8.25), (2.45, 8.03), GREEN)
    ax.text(6.5, 8.55, 'Updated physical state and observed history', fontsize=12, color=GREEN)
    ax.text(.5, .55,
        'Ontario cap 75.50 m · Pointe-Claire cap 22.10 m · fixed experimental model thresholds\n'
        'Stored water links decisions through time. More Ontario release can relieve the lake while raising downstream stage.',
        fontsize=13, color=INK, linespacing=1.7)
    fig.savefig(destination/'ontario_mpc_controller.png', facecolor=fig.get_facecolor())
    plt.close(fig)

    fig, ax = figure('PPO on Ontario: the planned learning application',
                     'Shared PPO implementation exists · Ontario observations, reward and training protocol still to be specified')
    box(ax, .45, 5.2, 4.0, 2.8, 'DECISION INFORMATION',
        'Ontario level and calendar\nPast release / supply / SLON\nAvailable ice and roughness info\nDefine and scale allowed inputs', BLUE)
    box(ax, 5.6, 5.2, 4.3, 2.8, 'ACTOR & CRITIC',
        'Actor proposes Ontario outflow\nCritic estimates future return\nTraining explores release choices\nEvaluate a frozen policy later', PURPLE)
    box(ax, 11.1, 5.2, 4.35, 2.8, 'GYMNASIUM → ENGINE',
        'Translate the requested action\nApply declared execution rules\nAdvance lake storage and routing\nObserve Ontario + downstream stage', GREEN)
    arrow(ax, (4.62, 6.6), (5.4, 6.6))
    arrow(ax, (10.08, 6.6), (10.9, 6.6), GREEN)
    box(ax, 5.6, 1.6, 7.9, 2.5, 'LEARN FROM COMPLETE TRANSITIONS',
        'Store observation, requested action, reward and next observation\n'
        'Specify reward from the agreed upstream / downstream objectives\n'
        'Estimate advantages, update actor and critic, collect more experience\n'
        'Preserve true storage and episode-boundary observations', BLUE)
    arrow(ax, (13.2, 5.05), (13.2, 4.25), GREEN)
    ax.plot([5.45, 4.95, 4.95], [2.8, 2.8, 6.2], color=BLUE, lw=2)
    arrow(ax, (4.95, 6.2), (5.42, 6.2), BLUE)
    ax.text(.55, 3.4, 'The training objective\nassigns credit to decisions\nwhose effects persist\ninto later periods.', color=INK, fontsize=13, linespacing=1.6)
    ax.text(.5, .55,
        'The physical problem is the same: retain flood headroom while managing downstream impacts and later storage.\n'
        'Ontario PPO results will follow the Gym adapter, reward definition and matched evaluation protocol.',
        fontsize=13, color=INK, linespacing=1.7)
    fig.savefig(destination/'ontario_ppo_controller.png', facecolor=fig.get_facecolor())
    plt.close(fig)


def load_data(diagnostics, comparison):
    base = pd.read_csv(comparison)
    settings = json.loads((diagnostics/'plan.json').read_text())['settings']
    arms = {'rule_curve': base.loc[base.policy.eq('rule_curve')],
            'mpc': pd.read_csv(diagnostics/'persistence.csv'),
            'oracle': pd.read_csv(diagnostics/'oracle_inflows.csv')}
    years = sorted(int(year) for year in base.year.unique())
    if set(years) != {1995, 2017, 2019}:
        raise ValueError('Expected the three declared Ontario case-study years')
    result = {'caps': {'ontario': settings['ontario_max_m'], 'downstream': settings['pointe_claire_max_m']},
              'years': years, 'policies': {}}
    for arm, frame in arms.items():
        records = {}
        for year in years:
            selected = frame.loc[frame.year.eq(year)].sort_values('quarter_month')
            if selected.quarter_month.tolist() != list(range(1, 49)):
                raise ValueError(f'Missing or duplicate quarter-month in {arm}, {year}')
            initial = float(base.loc[base.year.eq(year) & base.quarter_month.eq(1), 'ontario_start_m'].iloc[0])
            values = []
            for _, row in selected.iterrows():
                has_forecast = arm != 'rule_curve'
                values.append(dict(
                    period=int(row.quarter_month), start=initial,
                    ontario=float(row.ontario_end_m), downstream=float(row.pointe_claire_m),
                    release=float(row.release_m3_s), request=float(row.request_m3_s),
                    supply=float(row.realized_nts_m3_s), slon=float(row.realized_slon_m3_s),
                    feasible=bool(row.plan_feasible) if has_forecast else None,
                    ontarioForecast=[settings['ontario_max_m']-float(row[f'ontario_margin_qm{lead}_m']) for lead in range(1, 5)] if has_forecast else [],
                    downstreamForecast=[settings['pointe_claire_max_m']-float(row[f'pointe_claire_margin_qm{lead}_m']) for lead in range(1, 5)] if has_forecast else [],
                ))
                initial = float(row.ontario_end_m)
            records[str(year)] = values
        result['policies'][arm] = records
    source_paths = [diagnostics/'persistence.csv', diagnostics/'oracle_inflows.csv', comparison]
    result['sources'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostics', type=Path, default=ROOT/'results/loslr/mpc_diagnostics_v2')
    parser.add_argument('--comparison', type=Path, default=ROOT/'results/loslr/mpc_v2/comparison.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'background/ontario_simulator.html')
    args = parser.parse_args()
    data = load_data(args.diagnostics.resolve(), args.comparison.resolve())
    template = Path(__file__).with_name('ontario_simulator_template.html').read_text()
    payload = json.dumps(data, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    diagrams(args.output.parent/'images')
    args.output.write_text(template.replace('__EXPERIMENT_DATA__', payload))
    print(f'Built {args.output}: {len(data["years"])} years, {len(data["policies"])} controllers/diagnostics')


if __name__ == '__main__':
    main()
