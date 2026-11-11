"""Explicit experimental low-water operating rule, separate from hydraulics.

Only current level is used. The rule has priority over minimum-flow service;
shortfalls remain logged. No operator calibration or unconditional safety claim.
"""
from dataclasses import dataclass
import math

from reservoir_control.systems.lake_como.environment import settings
from reservoir_control.systems.lake_como.splits import split_years
from reservoir_control.systems.lake_como.simulation.outlet import OutletSpec, OutletLake, OutletModel, OutletEnv, OutletTrainingEpisodes


@dataclass(frozen=True)
class LowWaterRule:
    floor_m: float = -.5
    reserve_hours: float = 24.

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.floor_m, self.reserve_hours)):
            raise ValueError('Finite low-water parameters required')
        if self.reserve_hours < 1:
            raise ValueError('Reserve horizon must cover the hourly integrator')

    def release_cap(self, level_m, area_m2):
        if not all(math.isfinite(v) for v in (level_m, area_m2)) or area_m2 <= 0:
            raise ValueError('Finite level and positive area required')
        return max(0., area_m2*(level_m-self.floor_m)/(self.reserve_hours*3600))


class ProtectedOutletLake(OutletLake):
    def __init__(self, params, spec, rule):
        if spec.capacity != 'published_full_open':
            raise ValueError('Protection ablation uses the published capacity candidate')
        super().__init__(params, spec)
        if not spec.lower_evaluation_m < rule.floor_m < spec.upper_evaluation_m:
            raise ValueError('Operating floor must be inside the evaluation envelope')
        self.rule = rule

    def unprotected_release(self, request, storage, day):
        return super().actual_release(request, storage, day)

    def actual_release(self, request, storage, day):
        hydraulic = self.unprotected_release(request, storage, day)
        cap = self.rule.release_cap(self.storage_to_level(storage), self.surface)
        return min(hydraulic, cap)

    def integrate_interval(self, storage, request, inflow, day, *,
                           duration_seconds=86400., substeps=24):
        if not isinstance(substeps, int) or substeps < 1 or duration_seconds/substeps > self.rule.reserve_hours*3600:
            raise ValueError('Integration step must not exceed reserve horizon')
        return super().integrate_interval(storage, request, inflow, day,
            duration_seconds=duration_seconds, substeps=substeps)


class ProtectedOutletModel(OutletModel):
    def __init__(self, params, spec=OutletSpec(), rule=LowWaterRule()):
        super().__init__(params, spec)
        self.lake = ProtectedOutletLake(params['lake_params'], spec, rule)

    def advance(self, state, request, inflow, demand, day, evaporation_mm_day=0.):
        # The level-only proof assumes nonnegative net external input. Supplied
        # evaporation is not silently ignored or compensated using future inflow.
        if evaporation_mm_day != 0:
            raise ValueError('This protection experiment requires zero evaporation')
        hydraulic = self.lake.unprotected_release(request, state.storage, day)
        cap = self.lake.rule.release_cap(self.lake.storage_to_level(state.storage), self.lake.surface)
        following, row = super().advance(state, request, inflow, demand, day)
        row.update(unprotected_release_m3_s=float(hydraulic),
                   low_water_release_cap_m3_s=cap,
                   low_water_intervened=bool(hydraulic > row['release_m3_s']+1e-10),
                   operationally_curtailed_m3=(hydraulic-row['release_m3_s'])*3600,
                   operating_minimum_shortfall_m3=max(0., self.lake.min_release(state.storage,day)-row['release_m3_s'])*3600)
        return following, row


class ProtectedOutletEnv(OutletEnv):
    def __init__(self, params, years, *, outlet=OutletSpec(), rule=LowWaterRule(), **kwargs):
        super().__init__(params, years, outlet=outlet, **kwargs)
        if rule is not None:
            if any(values.any() for values in self.evaps.values()):
                raise ValueError('This protection experiment requires zero evaporation')
            self.model = ProtectedOutletModel(params, outlet, rule)


def make_protected_outlet_env(split='train', years=(1946,), **kwargs):
    if any(year not in split_years(split) for year in years):
        raise ValueError('Years must stay within the selected split')
    return ProtectedOutletEnv(settings(split), years, **kwargs)


class ProtectedOutletTrainingEpisodes(OutletTrainingEpisodes):
    def _build(self, years):
        return ProtectedOutletEnv(self.params, years, decision_hours=self.protocol.decision_hours,
            terminal_mode=self.protocol.terminal_mode,
            initial_level_range=self.protocol.initial_level_range, **self.kwargs)
