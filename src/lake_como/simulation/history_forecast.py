"""Causal seven-day analogues fitted only to a declared prior training prefix."""
import numpy as np

from .protected_mpc import CausalForecastBank
from .precision_mpc import PrecisionMPC

METHODS = ('baseline', 'history', 'history_stratified', 'history_scaled_stratified')


def features(history):
    values = np.asarray(history, float)
    if values.shape != (7,) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Seven finite nonnegative past observations required')
    x = np.log1p(values)
    return np.array([x[-1], x[-3:].mean(), x.mean(), x[-3:].mean()-x[:3].mean()])


class HistoryForecastBank(CausalForecastBank):
    def __init__(self, inflows, targets, *, before_year, method):
        if method not in METHODS:
            raise ValueError('Declared forecast method required')
        super().__init__(inflows, targets, before_year=before_year)
        self.method = method
        self.history = np.array([], float)
        self.starts = np.arange(7, len(self.flat)-7+1)
        self.past = np.array([features(self.flat[s-7:s]) for s in self.starts])
        self.scale = np.maximum(self.past.std(axis=0), 1e-6)
        self.paths = self.flat[self.starts[:, None]+np.arange(7)]
        self.pools = {}
        for day in range(1, 366):
            distance = np.abs(self.starts%365-(day-1))
            self.pools[day] = np.flatnonzero(np.minimum(distance, 365-distance) <= 15)
        for a in [self.starts, self.past, self.scale, self.paths]:
            a.setflags(write=False)

    def set_history(self, values):
        values = np.asarray(values, float)
        if values.ndim != 1 or len(values)>7 or not np.isfinite(values).all() or (values<0).any():
            raise ValueError('At most seven finite nonnegative past inflows required')
        self.history = values.copy()

    def forecast(self, kind, day, last_inflow, *, horizon=7, count=5, seed=901):
        if self.method == 'baseline' or len(self.history)<7 or kind != 'conditional':
            flows, targets, meta = super().forecast(kind, day, last_inflow, horizon=horizon, count=count, seed=seed)
            return flows, targets, dict(meta, history_method=self.method, cold_history=len(self.history)<7)
        if not 1<=day<=365 or not 1<=horizon<=7 or count!=5:
            raise ValueError('Daily calendar, horizon <=7 and five scenarios required')
        if not np.isfinite(last_inflow) or last_inflow<0 or self.history[-1] != last_inflow:
            raise ValueError('History must end at the currently observed past inflow')
        indices = self.pools[day]
        distance = np.mean(((self.past[indices]-features(self.history))/self.scale)**2, axis=1)
        pool = indices[np.argsort(distance, kind='stable')[:128]]
        paths = self.paths[pool].copy()
        if self.method == 'history_scaled_stratified':
            multiplier = np.clip((last_inflow+1)/(self.flat[self.starts[pool]-1]+1), .5, 2.)
            paths *= multiplier[:, None]
        rng = np.random.default_rng(seed)
        if 'stratified' in self.method:
            # Near-equal-size strata by training seven-day volume; each scenario keeps
            # its historical temporal shape. Every stratum has equal MPC weight.
            order = np.argsort(paths.sum(axis=1), kind='stable')
            strata = np.array_split(order, 5)
            chosen = np.array([rng.choice(group) for group in strata])
        else:
            chosen = rng.choice(len(pool), count, replace=True)
        calendar = (day-1+np.arange(horizon))%365
        return paths[chosen, :horizon].copy(), self.target_mean[calendar].copy(), dict(
            training_years=list(self.years), source_start_indices=self.starts[pool[chosen]].tolist(),
            conditioning_pool_size=len(pool), history_method=self.method, cold_history=False,
            observed_history=self.history.tolist(),
            sampling='Prior analogues; stratified variants represent near-equal-volume-rank strata, not calibrated risk guarantees')


class HistoryMPC(PrecisionMPC):
    """Same MPC and observations; retain only flows actually returned in obs[4]."""
    def __init__(self, params, bank, **kwargs):
        self.history_bank = bank
        super().__init__(params, bank, precision_safe=True, conservative=True, **kwargs)

    def reset(self):
        super().reset()
        self.observed_flows = []
        self.last_history_hour = None
        self.primed_hour = None

    def prime_history(self, values, hour):
        if not isinstance(hour, int) or hour<0 or hour%24 or len(values)!=min(7, hour//24):
            raise ValueError('Aligned clock and exactly the available trailing history required')
        self.history_bank.set_history(values)
        self.observed_flows = list(map(float, values))
        self.primed_hour = hour
        self.last_history_hour = None

    def __call__(self, observation, info):
        hour = int(info['elapsed_hours'])
        if self.primed_hour == hour:
            if hour and self.observed_flows[-1] != float(observation[4]):
                raise ValueError('Primed history must match the observed past inflow')
            self.primed_hour = None
        elif self.last_history_hour is None:
            if hour != 0:
                raise ValueError('Prime observed history before a mid-episode branch')
            self.observed_flows = []  # reset fallback is not a measured inflow
        elif self.last_history_hour == hour-24:
            self.observed_flows = (self.observed_flows+[float(observation[4])])[-7:]
        else:
            raise ValueError('Consecutive daily observations required')
        self.history_bank.set_history(self.observed_flows)
        action = super().__call__(observation, info)
        self.last_history_hour = hour
        return action
