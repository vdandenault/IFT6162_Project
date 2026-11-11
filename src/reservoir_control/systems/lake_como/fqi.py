"""Como past-inflow observations and action mapping for fitted Q iteration."""
import numpy as np
from reservoir_control.controllers.fqi.agent import state_actions


class FQIController:
    """Build the same past-inflow observation from completed-day telemetry."""

    def __init__(
        self, model, normalized_actions, physical_high, fallback, safety_filter=None
    ):
        self.model = model
        self.normalized_actions = np.asarray(normalized_actions, dtype=np.float32)
        self.physical_high = float(physical_high)
        self.fallback = float(fallback)
        self.filter = safety_filter
        self.previous_storage = None
        self.previous_day = None

    def __call__(self, observation, info):
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (4,) or not np.isfinite(obs).all():
            raise ValueError("Finite four-feature base observation required")
        day = (
            int(np.rint((np.arctan2(obs[1], obs[2]) % (2 * np.pi)) * 365 / (2 * np.pi)))
            % 365
            + 1
        )
        if day == 1:
            self.previous_storage = None
        elif self.previous_day != day - 1:
            raise ValueError("Consecutive days required")
        storage = float(info["storage"])
        past = (
            self.fallback
            if self.previous_storage is None
            else max(
                0.0, (storage - self.previous_storage) / 86400 + float(info["release"])
            )
        )
        state = np.r_[obs, past].astype(np.float32)
        q = self.model.predict(state_actions(state[None, :], self.normalized_actions))
        if not np.isfinite(q).all():
            raise ValueError("Nonfinite Q prediction")
        # Ascending release grid breaks ties toward lower requested release.
        action = np.array(
            [self.normalized_actions[int(q.argmax())] * self.physical_high],
            dtype=np.float32,
        )
        self.previous_storage = storage
        self.previous_day = day
        return self.filter(action, state, info) if self.filter is not None else action
