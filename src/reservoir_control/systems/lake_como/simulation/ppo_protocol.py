"""Matched two-year PPO task and causal packing of MPC's available history."""
import numpy as np
from reservoir_control.systems.lake_como.simulation.low_water import ProtectedOutletTrainingEpisodes
from reservoir_control.systems.lake_como.simulation.protocol import TaskProtocol
from reservoir_control.systems.lake_como.simulation.protected_mpc import CausalForecastBank

OBSERVATION_NAMES=('level','sin_season','cos_season','past_target_ema','last_inflow',
    'last_command','river_storage','remaining_fraction','inflow_lag2','inflow_lag3',
    'inflow_lag4','inflow_lag5','inflow_lag6','inflow_lag7','observed_history_fraction')
SCALES=(1.,1.,1.,100.,100.,250.,1e7,1.,100.,100.,100.,100.,100.,100.,1.)


class MatchedTrainingEpisodes(ProtectedOutletTrainingEpisodes):
    def __init__(self):
        super().__init__(TaskProtocol(episode_years=2,burn_in_days=0,decision_hours=24,
            terminal_mode='finite_horizon',initial_level_range=(.35,.35)))

    def _build(self,years):
        if not hasattr(self,'prior_last_flow'):
            self.prior_last_flow=CausalForecastBank(self.params['inflow'],self.params['demand'],before_year=1988).initial_last_inflow
        env=super()._build(years);env.fallback=self.prior_last_flow
        return env


class ObservationHistory:
    """Preserve base eight features and add only older returned obs[4] values.

    Missing lags use the explicitly unmeasured reset prior; the count indicator
    distinguishes padding. Calendar rollover never resets history.
    """
    def __init__(self):self.reset()

    def reset(self):
        self.history=[];self.last_hour=None;self.fallback=None;self.last_raw=None;self.last_encoded=None

    def encode(self,observation,info):
        raw=np.asarray(observation,np.float32);hour=int(info['elapsed_hours'])
        if raw.shape!=(8,) or not np.isfinite(raw).all() or hour<0 or hour%24:
            raise ValueError('Finite daily eight-feature observation required')
        if self.last_hour==hour:
            if not np.array_equal(raw,self.last_raw):raise ValueError('Different observation at the same clock')
            return self.last_encoded.copy()
        if self.last_hour is None:
            if hour!=0:raise ValueError('History must start at the original episode reset')
            self.fallback=float(raw[4])
        elif hour==self.last_hour+24:
            self.history=(self.history+[float(raw[4])])[-7:]
        else:raise ValueError('Consecutive daily observations required; reset explicitly between episodes')
        older=[self.history[-lag] if len(self.history)>=lag else self.fallback for lag in range(2,8)]
        encoded=np.r_[raw,np.asarray(older,np.float32),np.float32(len(self.history)/7)].astype(np.float32)
        self.last_hour=hour;self.last_raw=raw.copy();self.last_encoded=encoded.copy()
        return encoded
