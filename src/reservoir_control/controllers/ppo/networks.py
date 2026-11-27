import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def log_probability(distribution, latent):
    jacobian = 2 * (math.log(2) - latent - F.softplus(-2 * latent))
    return (distribution.log_prob(latent) - jacobian).sum(dim=-1)


class Networks(nn.Module):
    def __init__(self, config, action_low, action_high, observation_scales):
        super().__init__()
        scales = (
            np.ones(config.observation_dim)
            if observation_scales is None
            else np.asarray(observation_scales)
        )
        if (
            scales.shape != (config.observation_dim,)
            or not np.isfinite(scales).all()
            or (scales <= 0).any()
        ):
            raise ValueError("One finite positive scale per observation is required")
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32))
        self.register_buffer("action_low", torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))

        def mlp():
            net = nn.Sequential(
                nn.Linear(config.observation_dim, config.hidden),
                nn.Tanh(),
                nn.Linear(config.hidden, config.hidden),
                nn.Tanh(),
                nn.Linear(config.hidden, 1),
            )
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, math.sqrt(2))
                    nn.init.zeros_(layer.bias)
            return net

        self.actor = mlp()
        self.critic = mlp()
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.orthogonal_(self.critic[-1].weight, 1.0)
        self.log_std = nn.Parameter(torch.full((1,), config.initial_log_std))

    def distribution(self, observations):
        mean = self.actor(observations / self.scales)
        return torch.distributions.Normal(mean, self.log_std.clamp(-5, 1).exp().expand_as(mean))

    def value(self, observations):
        return self.critic(observations / self.scales).squeeze(-1)

    def physical(self, latent):
        return self.action_low + (torch.tanh(latent) + 1) * 0.5 * (
            self.action_high - self.action_low
        )
