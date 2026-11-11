"""Lake Como observation scaling for the shared PPO implementation."""
from reservoir_control.controllers.ppo.agent import PPOAgent as SharedPPOAgent
from reservoir_control.controllers.ppo.agent import PPOConfig, gae, log_probability
from reservoir_control.systems.lake_como.simulation.ppo_protocol import SCALES


class PPOAgent(SharedPPOAgent):
    def __init__(self, config=PPOConfig(), *, action_low, action_high, seed=123,
                 observation_scales=None):
        scales = SCALES if observation_scales is None else observation_scales
        super().__init__(config, action_low=action_low, action_high=action_high,
                         seed=seed, observation_scales=scales)
