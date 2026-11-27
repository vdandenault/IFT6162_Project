from typing import Any, Iterable, Protocol, Sequence


class PreparedProblem(Protocol):
    initial_plan: Any
    stages: Sequence[Any]


class PlanningAdapter(Protocol):
    def prepare(self, initial, decision, start) -> PreparedProblem: ...
    def expand(self, path, stage, *, final: bool, problem: PreparedProblem) -> Iterable[Any]: ...
    def rank(self, path) -> tuple: ...


class RecedingHorizonSearch:
    """Retain the best partial schedules and return a complete action plan.

    The adapter supplies transitions, candidate actions, objectives, constraints
    and ranking. The application executes the first action and replans.
    """

    def __init__(self, adapter: PlanningAdapter, *, beam_width: int = 12):
        if isinstance(beam_width, bool) or not isinstance(beam_width, int) or beam_width < 1:
            raise ValueError("Beam width must be a positive integer")
        self.adapter = adapter
        self.beam_width = beam_width

    def plan(self, initial, decision, start):
        problem = self.adapter.prepare(initial, decision, start)
        if not problem.stages:
            raise ValueError("At least one forecast stage is required")
        paths = [problem.initial_plan]
        for depth, stage in enumerate(problem.stages, start=1):
            candidates = [
                candidate
                for path in paths
                for candidate in self.adapter.expand(
                    path, stage, final=depth == len(problem.stages), problem=problem
                )
            ]
            if not candidates:
                raise RuntimeError("The planning adapter produced no candidate schedules")
            paths = sorted(candidates, key=self.adapter.rank)[: self.beam_width]
        return paths[0]
