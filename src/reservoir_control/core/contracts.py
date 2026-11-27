from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Generic, Mapping, Protocol, TypeVar

State = TypeVar("State")
Action = TypeVar("Action", contravariant=True)
Forcing = TypeVar("Forcing", contravariant=True)
Observation = TypeVar("Observation")


@dataclass(frozen=True)
class DecisionTime:
    """One model transition, with a system-defined period within a year.

    Duration follows the chosen physical model's convention; adapters must not
    infer it from period numbers or silently replace a reference convention.
    """

    index: int
    year: int
    period: int
    duration_seconds: float

    def __post_init__(self):
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("Decision index must be a nonnegative integer")
        if isinstance(self.year, bool) or not isinstance(self.year, int):
            raise ValueError("Year must be an integer")
        if isinstance(self.period, bool) or not isinstance(self.period, int) or self.period < 1:
            raise ValueError("Period must be a positive system-defined integer")
        if not isfinite(self.duration_seconds) or self.duration_seconds <= 0:
            raise ValueError("Transition duration must be finite and positive")


@dataclass(frozen=True)
class Transition(Generic[State]):
    state: State
    metrics: Mapping[str, float] = field(default_factory=dict)
    info: Mapping[str, Any] = field(default_factory=dict)


class PlanningModel(Protocol[State, Action, Forcing]):
    """Physical transition with explicit forcing and no live-environment access."""

    def transition(
        self, state: State, action: Action, forcing: Forcing, time: DecisionTime
    ) -> Transition[State]: ...


@dataclass(frozen=True)
class EpisodeStart(Generic[Observation]):
    observation: Observation
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeStep(Generic[Observation]):
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)


class EnvironmentSession(Protocol[Observation, Action]):
    """Application episode semantics; implemented outside the physical engine."""

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> EpisodeStart[Observation]: ...
    def step(self, action: Action) -> EpisodeStep[Observation]: ...
    def close(self) -> None: ...
