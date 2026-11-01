"""Common finite-scenario flood filter; conditional checks, not robust guarantees."""
import numpy as np
from lake_como.flood_feasibility import grid_rollout
from lake_como.splits import split_years


def fit_error_limits(inflows, climatology):
    if set(inflows) != set(split_years("train")):
        raise ValueError("Training years only")
    matrix = np.array([inflows[y][:365] for y in split_years("train")], dtype=float)
    errors = np.c_[
        matrix[:, 0] - climatology[0], matrix[:, 1:] - matrix[:, :-1]
    ].ravel()
    return tuple(float(x) for x in np.quantile(errors, [0.01, 0.99], method="higher"))


def choose_request(requests, base_request, end_levels, peak_levels, threshold):
    """Closest sampled safe request, else minimum peak/end violation recovery."""
    worst_peak = peak_levels.max(axis=0)
    worst_end = end_levels.max(axis=0)
    safe = worst_peak <= threshold - 1e-7
    distance = abs(requests - base_request)
    if safe.any():
        indices = np.flatnonzero(safe)
        index = int(indices[np.lexsort((requests[indices], distance[indices]))[0]])
    else:
        # Starting violation may make peak ties unavoidable; endpoint breaks ties.
        index = int(np.lexsort((requests, distance, worst_end, worst_peak))[0])
    return index, bool(safe.any()), float(worst_peak[index]), float(worst_end[index])


class FloodSafetyFilter:
    """Use only raw current level, calendar, and completed-day telemetry.

    Scenario set: five equally spaced training-error interval scenarios, plus
    persistence. Does not certify intervening inflows or state-measurement error.
    """

    def __init__(self, model, fallback, error_limits, points=401):
        if (
            points < 2
            or not np.isfinite(error_limits).all()
            or len(error_limits) != 2
            or error_limits[0] > error_limits[1]
        ):
            raise ValueError("Invalid scenario/grid parameters")
        if not np.isfinite(fallback) or fallback < 0:
            raise ValueError("Invalid fallback")
        self.model = model
        self.fallback = float(fallback)
        self.error_limits = error_limits
        self.grid = (
            np.linspace(model.low, model.high, points).astype(np.float32).astype(float)
        )
        self.previous_storage = None
        self.previous_day = None
        self.diagnostics = []

    def __call__(self, proposed, observation, info):
        obs = np.asarray(observation)
        if obs.shape not in [(4,), (5,)] or not np.isfinite(obs).all():
            raise ValueError("Finite raw observation required")
        day = (
            int(np.rint((np.arctan2(obs[1], obs[2]) % (2 * np.pi)) * 365 / (2 * np.pi)))
            % 365
            + 1
        )
        storage = float(info["storage"])
        if day == 1:
            self.previous_storage = None
        elif self.previous_day != day - 1:
            raise ValueError("Consecutive daily observations required")
        past = (
            self.fallback
            if self.previous_storage is None
            else max(
                0.0, (storage - self.previous_storage) / 86400 + float(info["release"])
            )
        )
        lower = max(0.0, past + self.error_limits[0])
        upper = max(lower, past + self.error_limits[1])
        scenarios = np.unique(np.r_[np.linspace(lower, upper, 5), past])
        if not np.isfinite(past) or not np.isfinite(storage):
            raise ValueError("Nonfinite telemetry")
        raw = float(np.asarray(proposed).item())
        if not np.isfinite(raw):
            raise ValueError("Nonfinite proposal")
        base = float(np.float32(np.clip(raw, self.model.low, self.model.high)))
        requests = np.unique(np.r_[self.grid, base])
        # Planning precision matches MPC: raw float32 level, not exact storage info.
        starting_storage = self.model.lake.level_to_storage(float(obs[0]))
        end, peak = grid_rollout(
            self.model.lake,
            np.full(len(scenarios), starting_storage),
            scenarios,
            np.full(len(scenarios), day),
            requests,
        )
        if not np.isfinite(end).all() or not np.isfinite(peak).all():
            raise ValueError("Nonfinite prediction")
        index, feasible, selected_peak, selected_end = choose_request(
            requests, base, end, peak, self.model.flood_level
        )
        chosen = requests[index]
        self.diagnostics.append(
            dict(
                day=day,
                raw_proposal_m3_s=raw,
                nominal_proposal_m3_s=base,
                filtered_request_m3_s=chosen,
                intervened=bool(abs(chosen - base) > 1e-5),
                sampled_safe_action_found=feasible,
                starts_above_threshold=bool(float(obs[0]) > self.model.flood_level),
                forecast_past_inflow_m3_s=past,
                scenario_lower_m3_s=float(scenarios.min()),
                scenario_upper_m3_s=float(scenarios.max()),
                scenario_count=len(scenarios),
                selected_worst_predicted_peak_m=selected_peak,
                selected_worst_predicted_endpoint_m=selected_end,
            )
        )
        self.previous_storage = storage
        self.previous_day = day
        return np.array([chosen], dtype=np.float32)
