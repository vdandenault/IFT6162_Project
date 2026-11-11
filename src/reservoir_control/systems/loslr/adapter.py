"""Replay session and state construction; training reward is deliberately undefined."""
from __future__ import annotations


import pandas as pd

from reservoir_control.core.contracts import DecisionTime, Transition
from .data import forcing_from_row
from .model import OntarioModel, OntarioState, ReleaseRequest


def decision_time(row, index: int) -> DecisionTime:
    # A fixed nominal 365-day reporting duration is useful metadata, not a replacement for native
    # balance conversion. Do not interpret it as resolved daily hydrology.
    year = int(row["Year"])
    return DecisionTime(index, year, int(row["QM"]),
                        365.0 * 86400.0 / 48)


def initial_state(frame: pd.DataFrame) -> OntarioState:
    if len(frame) < 49:
        raise ValueError("Reference initialization requires 48 preceding input rows")
    return OntarioState(74.55, 5950.0, 2,
                        tuple(float(x) for x in frame.ontLevelMOQ.iloc[:48]))


class OntarioSession:
    """Live replay, distinct from pure planning transitions and from Gymnasium.

    Forecasts are returned separately. Physical forcing remains private to step;
    no reward or PPO observation schema is invented for the baseline milestone.
    """

    def __init__(self, model: OntarioModel, frame: pd.DataFrame):
        self.model = model
        self._frame = frame.copy(deep=True)
        self.reset()

    def reset(self) -> OntarioState:
        self.state = initial_state(self._frame)
        self.index = 48
        return self.state

    @property
    def done(self) -> bool:
        return self.index >= len(self._frame)

    def information(self):
        if self.done:
            raise RuntimeError("Replay is complete")
        row = self._frame.iloc[self.index]
        return forcing_from_row(row).forecast, decision_time(row, self.index - 48)

    def step(self, action: ReleaseRequest | float) -> Transition[OntarioState]:
        if self.done:
            raise RuntimeError("Replay is complete; reset before stepping again")
        row = self._frame.iloc[self.index]
        result = self.model.transition(self.state, action, forcing_from_row(row),
                                       decision_time(row, self.index - 48))
        self.state = result.state
        self.index += 1
        return result
