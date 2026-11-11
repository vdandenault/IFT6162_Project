"""Ontario physics, candidate releases and objectives for the MPC planning adapter."""

from dataclasses import dataclass, replace
from math import floor, isfinite

from reservoir_control.core.contracts import DecisionTime
from reservoir_control.systems.loslr.data import BALANCE_CONVERSION
from reservoir_control.systems.loslr.model import DecisionInputs, OntarioModel, OntarioState, Plan2014, ReleaseRequest


@dataclass(frozen=True)
class MPCSettings:
    ontario_max_m: float = 75.50
    pointe_claire_max_m: float = 22.10
    beam_width: int = 12
    tracking_weight: float = 1.0
    terminal_weight: float = 1.0


@dataclass(frozen=True)
class ForecastStep:
    state: OntarioState
    release_m3_s: float
    pointe_claire_m: float


@dataclass(frozen=True)
class MPCPlan:
    actions: tuple[ReleaseRequest, ...]
    steps: tuple[ForecastStep, ...]
    tracking_cost: float
    terminal_cost: float
    max_excess_m: float
    total_excess_m: float

    @property
    def objective(self) -> float:
        return self.tracking_cost + self.terminal_cost

    @property
    def feasible(self) -> bool:
        return self.max_excess_m <= 1e-9


class OntarioPlanningAdapter:
    """Translate Ontario forecasts and physical rules into a search problem."""

    def __init__(self, model: OntarioModel, settings: MPCSettings = MPCSettings()):
        numeric = (settings.ontario_max_m, settings.pointe_claire_max_m,
                   settings.tracking_weight, settings.terminal_weight)
        if (settings.beam_width < 1 or not all(map(isfinite, numeric))
                or settings.pointe_claire_max_m <= 16.57
                or min(settings.tracking_weight, settings.terminal_weight) < 0):
            raise ValueError("Invalid Ontario MPC settings")
        self.model = model
        self.settings = settings
        self.rule = Plan2014(model.reference)

    def forecast_step(self, state, action, decision, time):
        limits = self.model.release_limits(state, action, time, decision=decision)
        release = float(limits["ontFlow"]) * 10
        following = self.model.advance_storage(
            state, release, decision.forecast.nts_m3_s[0], decision.ice, time,
        )
        # Same Pointe-Claire rating and rounding as the pinned stlaw routing.
        combined = release + decision.forecast.slon_m3_s
        stage = self.model.reference.utils.round_d(
            16.57 + (decision.pointe_claire_roughness * combined / 604.0) ** 0.58, 2,
        )
        return ForecastStep(following, release, stage)

    def _requests(self, state, rule_action, decision):
        settings = self.settings
        downstream_cap = (
            604.0 * (settings.pointe_claire_max_m - 16.57) ** (1 / 0.58)
            / decision.pointe_claire_roughness - decision.forecast.slon_m3_s
        )
        upstream_need = (
            decision.forecast.nts_m3_s[0]
            + (state.level_m - settings.ontario_max_m) * 10 * BALANCE_CONVERSION
        )
        requests = {0.0, 11500.0, state.previous_release_m3_s, upstream_need,
                    10.0 * floor(downstream_cap / 10)}
        requests.update(rule_action.m3_s + offset for offset in (-1400, -700, -350, 350, 700, 1400))
        yield rule_action
        for release in sorted({max(0.0, min(11500.0, request)) for request in requests}):
            yield ReleaseRequest(release)

    def forecast_stages(self, decision: DecisionInputs, start: DecisionTime):
        forecast = decision.forecast
        for lead in range(len(forecast.nts_m3_s)):
            year_offset, period = divmod(start.period - 1 + lead, 48)
            time = DecisionTime(start.index + lead, start.year + year_offset,
                                period + 1, start.duration_seconds)
            if lead == 0:
                estimate = decision
            else:
                remaining = forecast.nts_m3_s[lead:]
                estimate = replace(decision, forecast=replace(
                    forecast, nts_m3_s=remaining + (remaining[-1],) * lead,
                ))
            yield estimate, time

    def prepare(self, initial: OntarioState, decision: DecisionInputs, start: DecisionTime):
        stages = tuple(self.forecast_stages(decision, start))
        target_state = initial
        for estimate, time in stages:
            rule_action = self.rule.action(target_state, estimate.forecast, time)
            target_state = self.forecast_step(target_state, rule_action, estimate, time).state
        return OntarioPlanningProblem(initial, stages, target_state)

    def expand(self, path, stage, *, final, problem):
        estimate, time = stage
        state = path.steps[-1].state if path.steps else problem.initial_state
        rule_action = self.rule.action(state, estimate.forecast, time)
        reference = self.forecast_step(state, rule_action, estimate, time)
        seen = set()
        for action in self._requests(state, rule_action, estimate):
            step = self.forecast_step(state, action, estimate, time)
            if step.release_m3_s in seen:
                continue
            seen.add(step.release_m3_s)
            upstream = max(0.0, step.state.level_m - self.settings.ontario_max_m)
            downstream = max(0.0, step.pointe_claire_m - self.settings.pointe_claire_max_m)
            tracking = path.tracking_cost + self.settings.tracking_weight * ((step.release_m3_s - reference.release_m3_s) / 1000) ** 2
            terminal = 0.0
            if final:
                terminal = self.settings.terminal_weight * ((step.state.level_m - problem.target_state.level_m) / 0.10) ** 2
            yield MPCPlan(
                path.actions + (action,), path.steps + (step,), tracking, terminal,
                max(path.max_excess_m, upstream, downstream),
                path.total_excess_m + upstream + downstream,
            )

    @staticmethod
    def rank(path):
        # Feasible schedules take priority, followed by peak and total excess.
        return (not path.feasible, 0.0 if path.feasible else path.max_excess_m,
                0.0 if path.feasible else path.total_excess_m, path.objective)


@dataclass(frozen=True)
class OntarioPlanningProblem:
    initial_state: OntarioState
    stages: tuple
    target_state: OntarioState
    initial_plan: MPCPlan = MPCPlan((), (), 0.0, 0.0, 0.0, 0.0)
