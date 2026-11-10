"""Opt-in outlet-capacity ablation; never replaces the pinned benchmark.

The published fully-open curve is not a gate-position model. Its experiment
envelope is declared explicitly, not asserted to be a source-certified validity
range. Outside it the candidate fails closed for research use (no extrapolation).
"""
from dataclasses import dataclass
import math

from gym4real.envs.dam.lake import Lake
from lake_como.environment import settings
from lake_como.splits import split_years
from .environment import OperationalEnv
from .model import OperationalModel, Operations
from .protocol import TrainingEpisodes


class OutletDomainError(ValueError):
    """An experiment left the declared outlet-curve evaluation envelope."""


@dataclass(frozen=True)
class OutletSpec:
    capacity: str = 'published_full_open'
    area_m2: float = 145900000.
    storage_reference_m: float = -.5
    lower_evaluation_m: float = -.705
    upper_evaluation_m: float = 2.64

    def __post_init__(self):
        if self.capacity not in ('legacy', 'published_full_open'):
            raise ValueError('Unknown outlet capacity')
        numbers = (self.area_m2, self.storage_reference_m,
                   self.lower_evaluation_m, self.upper_evaluation_m)
        if not all(math.isfinite(v) for v in numbers) or self.area_m2 <= 0:
            raise ValueError('Finite parameters and positive area required')
        if not -2.5 < self.lower_evaluation_m < self.upper_evaluation_m:
            raise ValueError('Ordered evaluation envelope above curve zero required')


class OutletLake(Lake):
    def __init__(self, params, spec=OutletSpec()):
        super().__init__(params)
        self.spec = spec
        self.surface = spec.area_m2
        if (self.C_r, self.alpha, self.beta) != (33.37, -2.5, 2.015):
            raise ValueError('Published candidate requires its documented coefficients')

    def storage_to_level(self, storage):
        return storage / self.surface + self.spec.storage_reference_m

    def level_to_storage(self, level):
        return self.surface * (level - self.spec.storage_reference_m)

    def check_domain(self, level):
        if not math.isfinite(level):
            raise OutletDomainError('Nonfinite level')
        if self.spec.capacity == 'published_full_open' and not (
                self.spec.lower_evaluation_m-1e-12 <= level <= self.spec.upper_evaluation_m+1e-12):
            raise OutletDomainError(f'Level {level:.9g} m outside declared experiment envelope; no extrapolation')

    def max_release(self, storage, cday):
        level = self.storage_to_level(storage)
        self.check_domain(level)
        if self.spec.capacity == 'legacy':
            return super().max_release(storage, cday)
        return self.C_r * (level - self.alpha)**self.beta

    # min_release intentionally inherits the original operating policy:
    # zero minimum <= -.5, MEF through 1.25, fully open above 1.25 m.
    # These thresholds are separate from the storage reference and capacity.

    def integrate_interval(self, storage, request, inflow, day, *,
                           duration_seconds=86400., substeps=24):
        """Explicit balance integration for convergence checks; no hidden flows."""
        if (not isinstance(substeps, int) or substeps < 1 or
                not all(math.isfinite(v) for v in (storage, request, inflow, duration_seconds))
                or min(request, inflow) < 0 or duration_seconds <= 0 or not 1 <= day <= 365):
            raise ValueError('Finite nonnegative inputs, valid day and positive integration dimensions required')
        dt = duration_seconds/substeps
        volume = 0.
        start = storage
        self.check_domain(self.storage_to_level(storage))
        for _ in range(substeps):
            release = self.actual_release(request, storage, day)
            storage += dt*(inflow-release)
            volume += dt*release
            self.check_domain(self.storage_to_level(storage))
        return dict(storage_m3=storage, release_volume_m3=volume,
                    mean_release_m3_s=volume/duration_seconds,
                    balance_error_m3=storage-start-duration_seconds*inflow+volume)


class OutletModel(OperationalModel):
    def __init__(self, params, spec=OutletSpec()):
        # First ablation deliberately disables the uncalibrated mechanisms.
        super().__init__(params, Operations())
        self.lake = OutletLake(params['lake_params'], spec)

    def advance(self, state, request, inflow, demand, day, evaporation_mm_day=0.):
        self.lake.check_domain(self.lake.storage_to_level(state.storage))
        following, row = super().advance(state, request, inflow, demand, day, evaporation_mm_day)
        self.lake.check_domain(row['level_m'])
        row.update(outlet_capacity_m3_s=float(self.lake.max_release(state.storage, day)),
                   below_storage_reference=bool(state.storage < 0))
        return following, row


class OutletEnv(OperationalEnv):
    """Same observations/actions/reward code; only declared geometry/capacity vary."""
    def __init__(self, params, years, *, outlet=OutletSpec(), **kwargs):
        if 'operations' in kwargs:
            raise ValueError('Outlet ablation fixes ramp/routing disabled')
        super().__init__(params, years, **kwargs)
        self.model = OutletModel(params, outlet)
        for bound in self.initial_level_range:
            self.model.lake.check_domain(bound)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # Negative relative storage is bookkeeping, not an empty lake.
        info['below_storage_reference'] = info.pop('relative_storage_outside_model_domain')
        info['outlet_capacity_model'] = self.model.lake.spec.capacity
        info['outlet_extrapolated'] = False
        return obs, reward, terminated, truncated, info


def make_outlet_env(split='train', years=(1946,), **kwargs):
    if any(year not in split_years(split) for year in years):
        raise ValueError('Years must remain within the selected split')
    return OutletEnv(settings(split), years, **kwargs)


class OutletTrainingEpisodes(TrainingEpisodes):
    def _build(self, years):
        return OutletEnv(self.params, years, decision_hours=self.protocol.decision_hours,
                         terminal_mode=self.protocol.terminal_mode,
                         initial_level_range=self.protocol.initial_level_range, **self.kwargs)
