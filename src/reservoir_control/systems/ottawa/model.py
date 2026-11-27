from dataclasses import dataclass
from math import isfinite

from reservoir_control.core.contracts import DecisionTime, Transition


def _finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class OttawaParameters:
    """Explicit study inputs; these values have no built-in physical provenance."""

    minimum_storage_m3: float
    maximum_storage_m3: float
    minimum_release_m3_s: float
    maximum_release_m3_s: float

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            _finite(name, value)
        if not 0 <= self.minimum_storage_m3 < self.maximum_storage_m3:
            raise ValueError("storage bounds must satisfy 0 <= minimum < maximum")
        if not 0 <= self.minimum_release_m3_s <= self.maximum_release_m3_s:
            raise ValueError("release bounds must satisfy 0 <= minimum <= maximum")


@dataclass(frozen=True)
class OttawaState:
    """Useful storage in m³, not a measured gauge elevation."""

    storage_m3: float

    def __post_init__(self) -> None:
        _finite("storage_m3", self.storage_m3)


@dataclass(frozen=True)
class OttawaForcing:
    """Period-mean net inflow at the aggregate boundary, including stated losses."""

    net_inflow_m3_s: float

    def __post_init__(self) -> None:
        _finite("net_inflow_m3_s", self.net_inflow_m3_s)


@dataclass(frozen=True)
class AggregateStorageModel:
    parameters: OttawaParameters

    def feasible_release_bounds(
        self, state: OttawaState, forcing: OttawaForcing, time: DecisionTime
    ) -> tuple[float, float] | None:
        """Intersect outlet limits and one-step storage feasibility; never spill."""
        p = self.parameters
        if not p.minimum_storage_m3 <= state.storage_m3 <= p.maximum_storage_m3:
            raise ValueError("initial storage is outside declared bounds")
        lower = max(
            p.minimum_release_m3_s,
            forcing.net_inflow_m3_s
            + (state.storage_m3 - p.maximum_storage_m3) / time.duration_seconds,
        )
        upper = min(
            p.maximum_release_m3_s,
            forcing.net_inflow_m3_s
            + (state.storage_m3 - p.minimum_storage_m3) / time.duration_seconds,
        )
        return None if lower > upper else (lower, upper)

    def transition(
        self,
        state: OttawaState,
        action: float,
        forcing: OttawaForcing,
        time: DecisionTime,
    ) -> Transition[OttawaState]:
        """Advance a feasible release in m³/s without mutating the supplied state."""
        _finite("release_m3_s", action)
        bounds = self.feasible_release_bounds(state, forcing, time)
        if bounds is None:
            raise ValueError("no feasible release; spill/emergency rules are unspecified")
        if not bounds[0] <= action <= bounds[1]:
            raise ValueError("release violates outlet or next-storage bounds")
        inflow_volume = forcing.net_inflow_m3_s * time.duration_seconds
        release_volume = action * time.duration_seconds
        following = OttawaState(state.storage_m3 + inflow_volume - release_volume)
        return Transition(
            state=following,
            metrics={
                "storage_m3": following.storage_m3,
                "net_inflow_m3": inflow_volume,
                "release_m3": release_volume,
                "release_m3_s": action,
                "balance_error_m3": (
                    following.storage_m3 - state.storage_m3
                    - inflow_volume + release_volume
                ),
            },
            info={"calibrated": False, "routing_included": False},
        )
