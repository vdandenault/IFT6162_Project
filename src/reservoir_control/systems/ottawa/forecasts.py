import numpy as np


def inflow_forecast(mode, realized_past, initial_estimate, horizon, rng, sigma, *, perfect_values=None):
    if mode == 'perfect':
        if perfect_values is None or len(perfect_values) != horizon:
            raise ValueError('Explicit perfect-forecast values required')
        return np.asarray(perfect_values, float).copy()
    if mode != 'noisy':
        raise ValueError('Unknown forecast mode')
    last = float(realized_past[-1]) if len(realized_past) else initial_estimate
    return np.full(horizon, last) + rng.normal(0., sigma, horizon)


def forecast_bank(realized, mode, seed, initial_estimate, sigma, maximum_horizon=12):
    """Precompute all decision times before any control actions or early stopping.

    Noisy values at time k depend exclusively on observations strictly before k.
    Draw a fixed maximum-horizon row even at the endpoint. Unused cells are NaN.
    """
    bank = np.full((len(realized), maximum_horizon), np.nan)
    rng = np.random.default_rng(seed)
    for k in range(len(realized)):
        horizon = min(maximum_horizon, len(realized) - k)
        if mode == 'noisy':
            row = inflow_forecast(mode, realized[:k], initial_estimate, maximum_horizon, rng, sigma)
        else:
            row = inflow_forecast(
                mode, [], initial_estimate, horizon, rng, sigma,
                perfect_values=realized[k:k + horizon],
            )
        bank[k, :horizon] = row[:horizon]
    bank.flags.writeable = False
    return bank


def recession_bank(observed, persistence, *, jump_minimum_m3_s=None):
    """Keep lead one, then let positive departures from recent history decay."""
    if len(observed) != len(persistence):
        raise ValueError("Observation series and forecast bank must share the same clock")
    bank = np.array(persistence, copy=True)
    decay = 1.0 - np.exp2(-np.arange(bank.shape[1]))
    for decision in range(2, len(observed)):
        last = observed[decision - 1]
        earlier = observed[max(0, decision - 4):decision - 1]
        if jump_minimum_m3_s is not None and last - max(earlier) < jump_minimum_m3_s:
            continue
        departure = max(0.0, last - float(np.median(earlier)))
        bank[decision] -= departure * decay
    bank.flags.writeable = False
    return bank
