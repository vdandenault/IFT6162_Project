from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable, Mapping

from reservoir_control.core.contracts import EnvironmentSession, EpisodeStart, EpisodeStep


@dataclass(frozen=True)
class StepRecord:
    action: Any
    result: EpisodeStep


@dataclass(frozen=True)
class EpisodeRecord:
    initial: EpisodeStart
    steps: tuple[StepRecord, ...]

    @property
    def total_reward(self) -> float:
        return sum(row.result.reward for row in self.steps)


class IncompleteEpisodeError(RuntimeError):
    """A safety budget was reached; this is not an environment truncation."""


def run_episode(
    session: EnvironmentSession,
    controller: Callable[[Any, Mapping[str, Any]], Any],
    *,
    max_steps: int,
    seed: int | None = None,
    options: dict | None = None,
) -> EpisodeRecord:
    """Execute either controller through a session without a Gym dependency.

    The controller sees only the session's declared observation and public info.
    Recorded transitions are copied so later simulator updates cannot mutate them.
    The caller owns session construction, close and controller reset/lifecycle.
    """
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("A positive integer episode safety budget is required")
    start = session.reset(seed=seed, options=options)
    observation, info = start.observation, start.info
    records = []
    initial = deepcopy(start)
    for _ in range(max_steps):
        action = controller(deepcopy(observation), deepcopy(info))
        saved_action = deepcopy(action)
        result = session.step(action)
        if not isfinite(result.reward):
            raise ValueError("Cannot aggregate a nonfinite reward")
        records.append(StepRecord(saved_action, deepcopy(result)))
        if result.terminated or result.truncated:
            return EpisodeRecord(initial, tuple(records))
        observation, info = result.observation, result.info
    raise IncompleteEpisodeError("Episode did not end within the declared safety budget")
