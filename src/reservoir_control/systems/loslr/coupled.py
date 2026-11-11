"""Ontario–Ottawa research coupling and planning adapter, separate from algorithms."""
from dataclasses import dataclass, replace
import json
from pathlib import Path
import numpy as np
from reservoir_control.controllers.mpc.bilevel import NestedResponseSearch
from reservoir_control.systems.ottawa.model import (
    OttawaParameters, OttawaState, OttawaForcing, AggregateStorageModel,
)
from reservoir_control.systems.ottawa.follower import OttawaFollower
from .causal import CausalOntarioSession
from .data import forcing_from_row
from .model import OntarioModel, Plan2014, ReleaseRequest
from .planning import OntarioPlanningAdapter

ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT/'configs/studies/loslr/coupled_v1.json'
DURATION = 365*86400/48


def load_protocol(path=CONFIG):
    c = json.loads(Path(path).read_text())
    numeric = [v for v in c.values() if isinstance(v, (int, float))]
    if c['schema_version'] != 1 or not np.isfinite(numeric).all():
        raise ValueError('Supported finite study settings required')
    if c['horizon'] != 4 or c['downstream_mode'] != 'plan2014_f_tiers':
        raise ValueError('This adapter implements four-period F-tier planning')
    if (not 0 < c['controlled_fraction'] < 1 or c['ottawa_capacity_m3'] <= 0
            or not 0 < c['ottawa_initial_fraction'] < 1
            or c['ottawa_max_release_m3_s'] <= 0
            or c['routing_lag_periods'] != 0 or c['routing_gain'] != 1):
        raise ValueError('Valid aggregate settings and explicit zero-lag unit-gain routing required')
    for key in ('flood_weight', 'low_water_weight', 'release_change_weight',
                'terminal_weight', 'follower_service_weight', 'follower_storage_weight',
                'follower_flood_weight', 'follower_terminal_weight'):
        if c[key] < 0: raise ValueError('Nonnegative objective weights required')
    train = set(range(c['training_years'][0], c['training_years'][1]+1))
    dev = set(range(c['development_years'][0], c['development_years'][1]+1)) | set(c['development_events'])
    test = set(c['reserved_test_years'])
    if train & dev or train & test or dev & test:
        raise ValueError('Training, development and test membership must be disjoint')
    if not set(c['pilot_years']) <= dev:
        raise ValueError('Pilot cases must belong to development')
    return c


def f_limit_target(ontario_beginning_m):
    """Plan 2014 Annex B, Table B4; metre IGLD 1985 targets."""
    x = float(ontario_beginning_m)
    if not np.isfinite(x): raise ValueError('Finite beginning level required')
    return (22.10 if x < 75.30 else 22.20 if x < 75.37 else 22.33 if x < 75.50
            else 22.40 if x < 75.60 else 22.48)


def operating_cost(start_level, end_level, downstream, release, previous_release,
                   c, *, terminal=False):
    upstream = max(0., end_level-c['ontario_flood_cap_m'])
    down = max(0., downstream-f_limit_target(start_level))
    low = max(0., c['ontario_low_water_m']-end_level)
    components = dict(flood_cost=c['flood_weight']*(upstream**2+down**2),
                      low_water_cost=c['low_water_weight']*low**2,
                      release_change_cost=c['release_change_weight']*((release-previous_release)/1000)**2,
                      terminal_cost=c['terminal_weight']*(end_level-c['terminal_target_m'])**2 if terminal else 0.)
    return sum(components.values()), components


def start_states(model, frame, years):
    """Cache common annual Ontario starts using causal rule-curve warmup only."""
    session = CausalOntarioSession(model, frame)
    rule = Plan2014(model.reference)
    wanted, found = set(years), {}
    while not session.done:
        decision, time = session.information()
        if time.year in wanted and time.period == 1:
            found[time.year] = (session.state, session.index)
            if len(found) == len(wanted): break
        session.step(rule.action(session.state, decision.forecast, time))
    if set(found) != wanted: raise ValueError('Requested complete years are unavailable')
    return found


@dataclass(frozen=True)
class CoupledPlan:
    requests: tuple[float, ...]
    follower: object
    cost: float
    max_excess_m: float
    total_excess_m: float
    feasible: bool
    predicted_releases: tuple[float, ...]
    candidates: int = 1

    @property
    def rank(self):
        return (not self.feasible, self.max_excess_m if not self.feasible else 0.,
                self.total_excess_m if not self.feasible else 0., self.cost, self.requests)


class CoupledPlanningAdapter:
    """Domain-specific rollouts and response mapping for finite leader search."""
    def __init__(self, model, settings):
        self.model, self.settings = model, settings
        self.rule = Plan2014(model.reference)
        self.physical = OntarioPlanningAdapter(model)
        self.follower = OttawaFollower(settings)

    def neutral(self, state, decision, time):
        actions = []
        for estimate, when in self.physical.forecast_stages(decision, time):
            request = self.rule.action(state, estimate.forecast, when)
            actions.append(request.m3_s)
            state = self.physical.forecast_step(state, request, estimate, when).state
        return np.clip(np.asarray(actions), 0., 11500.)

    def response(self, state, storage, decision, schedule):
        n = len(schedule); c = self.settings
        slon = max(0., decision.forecast.slon_m3_s)
        # Freeze hydraulic target at decision-time beginning level for convexity.
        capacity = 604*(f_limit_target(state.level_m)-16.57)**(1/.58)/decision.pointe_claire_roughness
        return self.follower.solve(storage, np.full(n, slon*c['controlled_fraction']),
                                   np.full(n, slon*(1-c['controlled_fraction'])), schedule,
                                   np.full(n, capacity), DURATION)

    def evaluate(self, state, storage, decision, time, schedule, *, response=None):
        c = self.settings
        if response is None: response = self.response(state, storage, decision, schedule)
        costs, excess, releases = 0., [], []
        for i, (estimate, when) in enumerate(self.physical.forecast_stages(decision, time)):
            slon = max(0., decision.forecast.slon_m3_s)*(1-c['controlled_fraction']) + response.release_m3_s[i]
            estimate = replace(estimate, forecast=replace(estimate.forecast, slon_m3_s=slon))
            step = self.physical.forecast_step(state, ReleaseRequest(float(schedule[i])), estimate, when)
            upstream = max(0., step.state.level_m-c['ontario_flood_cap_m'])
            downstream = max(0., step.pointe_claire_m-f_limit_target(state.level_m))
            excess.extend((upstream, downstream))
            cost, _ = operating_cost(state.level_m, step.state.level_m, step.pointe_claire_m,
                                     step.release_m3_s, state.previous_release_m3_s, c,
                                     terminal=(i == len(schedule)-1))
            costs += cost; releases.append(step.release_m3_s); state = step.state
        return CoupledPlan(tuple(map(float, schedule)), response, costs, max(excess),
                           sum(excess), max(excess) <= c['stage_rounding_allowance_m'], tuple(releases))

    def candidates(self, neutral):
        for offset in (-1400., -700., -350., 0., 350., 700., 1400.):
            yield np.clip(neutral+offset, 0., 11500.)
        # Early release/hold patterns can express timing rather than just a constant shift.
        for offset in (-1400., -700., 700., 1400.):
            for count in (1, 2):
                pattern = np.r_[np.full(count, offset), np.zeros(4-count)]
                yield np.clip(neutral+pattern, 0., 11500.)

    def plan(self, state, storage, decision, time, *, anticipating):
        neutral = self.neutral(state, decision, time)
        fixed = None if anticipating else self.response(state, storage, decision, neutral)
        search = NestedResponseSearch(
            lambda schedule: self.response(state, storage, decision, schedule),
            lambda schedule, response: self.evaluate(state, storage, decision, time,
                                                     schedule, response=response),
            lambda plan: plan.rank)
        best, count = search.plan(self.candidates(neutral), fixed_response=fixed)
        return replace(best, candidates=count)


class CoupledSession:
    """One common responsive-follower execution environment for every controller."""
    def __init__(self, model, frame, initial, settings):
        # Copy a cached state into a session without replaying 100+ years per episode.
        self.session = object.__new__(CausalOntarioSession)
        self.session.model, self.session._frame = model, frame
        self.session.state, self.session.index = initial
        self.settings = settings
        self.storage_m3 = settings['ottawa_capacity_m3']*settings['ottawa_initial_fraction']
        self.adapter = CoupledPlanningAdapter(model, settings)
        self.ottawa = AggregateStorageModel(OttawaParameters(
            0., settings['ottawa_capacity_m3'], 0., settings['ottawa_max_release_m3_s']))
        self.elapsed = 0

    def information(self): return self.session.information()

    def step(self, schedule):
        if self.elapsed >= 48: raise RuntimeError('Episode is complete')
        schedule = np.asarray(schedule, float)
        if schedule.shape != (4,) or not np.isfinite(schedule).all() or np.min(schedule) < 0 or np.max(schedule) > 11500:
            raise ValueError('Four finite release requests within 0–11500 m3/s required')
        state = self.session.state
        decision, time = self.information()
        response = self.adapter.response(state, self.storage_m3, decision, schedule)
        raw = self.session._frame.iloc[self.session.index]
        forcing = forcing_from_row(raw)
        controlled = max(0., forcing.slon_m3_s)*self.settings['controlled_fraction']
        uncontrolled = max(0., forcing.slon_m3_s)*(1-self.settings['controlled_fraction'])
        ottawa_state = OttawaState(self.storage_m3)
        bounds = self.ottawa.feasible_release_bounds(ottawa_state, OttawaForcing(controlled), time)
        if bounds is None: raise RuntimeError('Ottawa physical feasibility failed; emergency spill unspecified')
        q = float(np.clip(response.release_m3_s[0], *bounds))
        outlet = self.ottawa.transition(ottawa_state, q, OttawaForcing(controlled), time)
        self.storage_m3 = outlet.state.storage_m3
        estimated_slon = max(0., decision.forecast.slon_m3_s)*(1-self.settings['controlled_fraction']) + response.release_m3_s[0]
        decision = replace(decision, forecast=replace(decision.forecast, slon_m3_s=estimated_slon))
        forcing = replace(forcing, slon_m3_s=uncontrolled+q)
        result = self.session.model.transition(state, ReleaseRequest(float(schedule[0])), forcing, time, decision=decision)
        self.session.state = result.state; self.session.index += 1; self.elapsed += 1
        release = result.metrics['executed_release_m3_s']
        downstream = result.metrics['pointe_claire_level_m']
        cost, components = operating_cost(state.level_m, result.state.level_m, downstream,
                                          release, state.previous_release_m3_s, self.settings,
                                          terminal=self.elapsed == 48)
        return dict(year=time.year, quarter_month=time.period, ontario_start_m=state.level_m,
                    ontario_end_m=result.state.level_m, pointe_claire_m=downstream,
                    downstream_target_m=f_limit_target(state.level_m),
                    release_m3_s=release, requested_release_m3_s=float(schedule[0]),
                    release_change_m3_s=abs(release-state.previous_release_m3_s),
                    ottawa_release_m3_s=q, ottawa_storage_m3=self.storage_m3,
                    ottawa_inflow_m3_s=controlled, uncontrolled_m3_s=uncontrolled,
                    ottawa_balance_error_m3=outlet.metrics['balance_error_m3'],
                    ottawa_safety_adjustment_m3_s=q-response.release_m3_s[0],
                    follower_service_shortfall_m3_s=max(0., controlled-q),
                    follower_solver_success=response.response.solver_success,
                    follower_kkt_residual=max(response.response.primal_residual,
                                               response.response.stationarity_residual,
                                               response.response.complementarity_residual),
                    forecast_slon_m3_s=estimated_slon, realized_slon_m3_s=uncontrolled+q,
                    cost=cost, reward=-cost, **components)
