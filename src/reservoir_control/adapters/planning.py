from copy import deepcopy
from typing import Generic, Sequence, TypeVar

from reservoir_control.core.contracts import DecisionTime, PlanningModel, Transition

State = TypeVar("State")
Action = TypeVar("Action")
Forcing = TypeVar("Forcing")


class PlanningAdapter(Generic[State, Action, Forcing]):
    def __init__(self, model: PlanningModel[State, Action, Forcing]):
        self.model = model

    def transition(
        self, state: State, action: Action, forcing: Forcing, time: DecisionTime
    ) -> Transition[State]:
        return self.model.transition(deepcopy(state), deepcopy(action), deepcopy(forcing), time)

    def rollout(
        self,
        state: State,
        actions: Sequence[Action],
        forcings: Sequence[Forcing],
        times: Sequence[DecisionTime],
    ) -> tuple[Transition[State], ...]:
        if not (len(actions) == len(forcings) == len(times)):
            raise ValueError("Actions, explicit forcings and decision times must align")
        if any(b.index != a.index + 1 for a, b in zip(times, times[1:])):
            raise ValueError("Planning decision indices must be consecutive")

        transitions = []
        current = deepcopy(state)
        for action, forcing, time in zip(actions, forcings, times):
            result = self.transition(current, action, forcing, time)
            transitions.append(deepcopy(result))
            current = result.state
        return tuple(transitions)
