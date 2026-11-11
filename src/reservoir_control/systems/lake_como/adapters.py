"""Planning and Gymnasium adapters around the Como physical simulator.

This module composes the system-specific Como implementation. Shared contracts and
controllers must not import it. Physical parameters remain owned by that system.
"""
from dataclasses import dataclass
from copy import deepcopy
from reservoir_control.adapters.gymnasium import GymnasiumAdapter
from reservoir_control.adapters.planning import PlanningAdapter
from reservoir_control.core import DecisionTime, EpisodeStart, EpisodeStep, Transition
from reservoir_control.systems.lake_como.simulation.environment import OperationalEnv
from reservoir_control.systems.lake_como.simulation.model import OperationalModel, Operations, WaterState


@dataclass(frozen=True)
class ComoForcing:
    inflow_m3_s: float
    demand_m3_s: float
    model_day: int
    evaporation_mm_day: float = 0.


class ComoPhysicalModel:
    """Hourly physical boundary; does not infer or hold future forcing records."""
    def __init__(self, params, *, operations=Operations()):
        self.model = OperationalModel(deepcopy(params), operations)

    def transition(self, state: WaterState, action: float, forcing: ComoForcing,
                   time: DecisionTime) -> Transition[WaterState]:
        if time.duration_seconds != 3600:
            raise ValueError('The preserved Como physical engine advances one hour')
        if time.period != forcing.model_day:
            raise ValueError('Decision period and Como forcing day must agree')
        following, metrics = self.model.advance(state, float(action), forcing.inflow_m3_s,
            forcing.demand_m3_s, forcing.model_day, forcing.evaporation_mm_day)
        return Transition(following, metrics, {'time': time})


class ComoPlanningAdapter(PlanningAdapter):
    def __init__(self, params, *, operations=Operations()):
        super().__init__(ComoPhysicalModel(params, operations=operations))


class ComoSession:
    """Translate existing episode semantics without rewriting frozen behavior."""
    def __init__(self, env: OperationalEnv):
        self.env = env

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        return EpisodeStart(observation, info)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return EpisodeStep(observation, reward, terminated, truncated, info)

    def close(self):
        self.env.close()


class ComoGymnasiumAdapter(GymnasiumAdapter):
    """Gym adapter preserving Como observations, reward and episode endings."""
    def __init__(self, params, years, **kwargs):
        env = OperationalEnv(params, years, **kwargs)
        super().__init__(ComoSession(env), action_space=env.action_space,
                         observation_space=env.observation_space)

    @property
    def physical_state(self):
        return deepcopy(self.session.env.state)
