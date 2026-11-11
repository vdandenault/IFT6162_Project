"""Declared initial-state, spin-up and terminal semantics for training episodes."""
from dataclasses import dataclass,asdict
import gymnasium as gym
import numpy as np
from reservoir_control.systems.lake_como.environment import settings
from reservoir_control.systems.lake_como.splits import split_years
from reservoir_control.systems.lake_como.simulation.environment import OperationalEnv


@dataclass(frozen=True)
class TaskProtocol:
    episode_years: int = 2
    burn_in_days: int = 30
    burn_in_request_m3_s: float = 100.
    decision_hours: int = 24
    terminal_mode: str = 'finite_horizon'
    gamma_per_day: float = .995
    initial_level_range: tuple = (.35,.35)

    def __post_init__(self):
        if not isinstance(self.episode_years,int) or self.episode_years<1:raise ValueError('Positive integer year count required')
        if not isinstance(self.burn_in_days,int) or not 0<=self.burn_in_days<365*self.episode_years:raise ValueError('Invalid spin-up length')
        if self.decision_hours not in [1,3,6,12,24]:raise ValueError('Invalid decision interval')
        if self.terminal_mode not in ['finite_horizon','continuing_record']:raise ValueError('Unknown terminal rule')
        if not np.isfinite([self.gamma_per_day,self.burn_in_request_m3_s]).all() or not 0<self.gamma_per_day<=1:
            raise ValueError('Invalid discount/request')
        bounds=np.asarray(self.initial_level_range,float)
        if bounds.shape!=(2,) or not np.isfinite(bounds).all() or bounds[0]>bounds[1]:raise ValueError('Ordered initial levels required')

    @property
    def gamma_per_decision(self):return self.gamma_per_day**(self.decision_hours/24)


class TrainingEpisodes(gym.Env):
    """Seeded contiguous windows from chronological TRAIN years only.

    Fixed-action spin-up preserves physical/actuator/river state, is charged to
    interaction accounting, and contributes no replay rows or scored reward.
    """
    metadata={'render_modes':[]}

    def __init__(self,protocol=TaskProtocol(),**kwargs):
        super().__init__();self.protocol=protocol;self.params=settings('train');self.kwargs=kwargs
        years=split_years('train');n=protocol.episode_years
        self.windows=[tuple(years[i:i+n]) for i in range(len(years)-n+1) if all(b==a+1 for a,b in zip(years[i:i+n],years[i+1:i+n]))]
        if not self.windows:raise ValueError('No training-only window of this length')
        self.current=self._build(self.windows[0]);self.action_space=self.current.action_space;self.observation_space=self.current.observation_space
        self.burn_in_transitions=0;self.agent_transitions=0;self.episodes=0
        if not self.action_space.low[0]<=protocol.burn_in_request_m3_s<=self.action_space.high[0]:raise ValueError('Spin-up request outside nominal bounds')

    def _build(self,years):
        return OperationalEnv(self.params,years,decision_hours=self.protocol.decision_hours,
            terminal_mode=self.protocol.terminal_mode,initial_level_range=self.protocol.initial_level_range,**self.kwargs)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        if options:raise ValueError('Training windows are selected by the declared seed')
        self.current.close();window=self.windows[int(self.np_random.integers(len(self.windows)))]
        self.current=self._build(window)
        obs,info=self.current.reset(seed=int(self.np_random.integers(2**31)))
        for _ in range(self.protocol.burn_in_days*24//self.protocol.decision_hours):
            obs,_,term,trunc,info=self.current.step(np.array([self.protocol.burn_in_request_m3_s],np.float32))
            assert not(term or trunc);self.burn_in_transitions+=1
        self.episodes+=1
        info=dict(info,episode_years=list(window),burn_in_transitions=self.burn_in_transitions,protocol=asdict(self.protocol))
        return obs,info

    def step(self,action):
        result=self.current.step(action);self.agent_transitions+=1;return result

    def close(self):self.current.close()


def replay_terminal(terminated,truncated):
    """Bootstrap through artificial record truncation; finite task ends terminate."""
    return bool(terminated)
