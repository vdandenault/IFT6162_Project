"""Lake Como release units for the shared SAC implementation."""
from reservoir_control.controllers.sac.agent import SACAgent as SharedSACAgent
from reservoir_control.controllers.sac.agent import SACConfig


class SACAgent(SharedSACAgent):
    def __init__(self, config=SACConfig(), *, observation_mean=None,
                 observation_scale=None, action_low=(0.0,), action_high=(491.61,), seed=0):
        super().__init__(config, observation_mean=observation_mean,
                         observation_scale=observation_scale, action_low=action_low,
                         action_high=action_high, seed=seed)
