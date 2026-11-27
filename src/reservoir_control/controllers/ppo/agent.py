import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PPOConfig:
    observation_dim: int = 15
    hidden: int = 64
    rollout: int = 2048
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    max_gradient_norm: float = 0.5
    target_kl: float = 0.03
    reward_scale: float = 0.01
    initial_log_std: float = -0.5

    def __post_init__(self):
        if min(self.observation_dim, self.hidden, self.rollout, self.epochs, self.batch_size) < 1:
            raise ValueError("Declared observation size and positive training dimensions required")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1 or self.reward_scale <= 0:
            raise ValueError("Valid discount, GAE and fixed reward units required")


def gae(rewards, values, next_values, terminated, truncated, gamma, lam):
    arrays = [np.asarray(a) for a in [rewards, values, next_values, terminated, truncated]]
    if not arrays or any(a.shape != arrays[0].shape for a in arrays) or arrays[0].ndim != 1:
        raise ValueError("Aligned scalar transition arrays required")
    rewards, values, next_values, terminated, truncated = arrays
    if not all(np.isfinite(a).all() for a in [rewards, values, next_values]):
        raise ValueError("Finite GAE inputs required")
    advantage = np.zeros(len(rewards), np.float64)
    next_advantage = 0.0
    for step in range(len(rewards) - 1, -1, -1):
        delta = (
            float(rewards[step])
            + gamma * (not bool(terminated[step])) * float(next_values[step])
            - float(values[step])
        )
        # Bootstrap truncations from their real stored observation, but never
        # carry GAE across a reset. True finite-task ends do not bootstrap.
        next_advantage = (
            delta
            + gamma * lam * (not (bool(terminated[step]) or bool(truncated[step]))) * next_advantage
        )
        advantage[step] = next_advantage
    return advantage.astype(np.float32), (advantage + values).astype(np.float32)


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


class PPOAgent:
    def __init__(
        self, config=PPOConfig(), *, action_low, action_high, seed=123, observation_scales=None
    ):
        torch.manual_seed(seed)
        self.config = config
        self.seed = seed
        self.model = Networks(config, action_low, action_high, observation_scales)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate, eps=1e-5
        )
        self.shuffle = np.random.default_rng(seed + 10000)
        self.updates = 0
        self.steps = 0

    @torch.no_grad()
    def act(self, observation, *, deterministic=False):
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        distribution = self.model.distribution(observation_tensor)
        latent = distribution.mean if deterministic else distribution.sample()
        log_prob = log_probability(distribution, latent)
        value = self.model.value(observation_tensor)
        action = self.model.physical(latent).reshape(-1)
        if not torch.isfinite(action).all():
            raise ValueError("Nonfinite PPO action")
        return (
            action.numpy().copy(),
            latent.reshape(-1).numpy().copy(),
            float(log_prob.item()),
            float(value.item()),
        )

    @torch.no_grad()
    def value(self, observation):
        return float(
            self.model.value(
                torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
            ).item()
        )

    def update(self, rollout):
        config = self.config
        advantage, returns = gae(
            rollout["reward"],
            rollout["value"],
            rollout["next_value"],
            rollout["terminated"],
            rollout["truncated"],
            config.gamma,
            config.gae_lambda,
        )
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        observations = torch.tensor(np.asarray(rollout["observation"]), dtype=torch.float32)
        latents = torch.tensor(np.asarray(rollout["latent"]), dtype=torch.float32)
        old_log_probs = torch.tensor(rollout["log_probability"], dtype=torch.float32)
        normalized_advantages = torch.tensor(advantage)
        return_targets = torch.tensor(returns)
        minibatch_metrics = []
        kl_early_stop = False
        for _ in range(config.epochs):
            indices = self.shuffle.permutation(len(observations))
            for start in range(0, len(observations), config.batch_size):
                batch = indices[start : start + config.batch_size]
                distribution = self.model.distribution(observations[batch])
                log_probs = log_probability(distribution, latents[batch])
                log_ratio = log_probs - old_log_probs[batch]
                ratio = log_ratio.exp()
                kl = ((ratio - 1) - log_ratio).mean()
                if not torch.isfinite(ratio).all():
                    raise ValueError("Nonfinite PPO probability ratio")
                if kl.item() > config.target_kl:
                    kl_early_stop = True
                    break
                batch_advantages = normalized_advantages[batch]
                clipped_ratio = ratio.clamp(1 - config.clip_ratio, 1 + config.clip_ratio)
                policy = -torch.minimum(
                    ratio * batch_advantages, clipped_ratio * batch_advantages
                ).mean()
                value = F.mse_loss(self.model.value(observations[batch]), return_targets[batch])
                entropy = distribution.entropy().sum(dim=-1).mean()
                loss = (
                    policy + config.value_coefficient * value - config.entropy_coefficient * entropy
                )
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite PPO loss")
                self.optimizer.zero_grad()
                loss.backward()
                actor_params = [*self.model.actor.parameters(), self.model.log_std]
                actor_norm = float(
                    torch.sqrt(
                        sum((p.grad.square().sum() for p in actor_params if p.grad is not None))
                    ).item()
                )
                critic_norm = float(
                    torch.sqrt(
                        sum(
                            (
                                p.grad.square().sum()
                                for p in self.model.critic.parameters()
                                if p.grad is not None
                            )
                        )
                    ).item()
                )
                norm = nn.utils.clip_grad_norm_(self.model.parameters(), config.max_gradient_norm)
                if not np.isfinite([actor_norm, critic_norm, float(norm)]).all():
                    raise ValueError("Nonfinite PPO gradient")
                self.optimizer.step()
                self.updates += 1
                if not all(torch.isfinite(p).all() for p in self.model.parameters()):
                    raise ValueError("Nonfinite PPO parameters")
                minibatch_metrics.append(
                    dict(
                        policy_loss=float(policy.item()),
                        value_loss=float(value.item()),
                        approx_kl=float(kl.item()),
                        clip_fraction=float(
                            ((ratio - 1).abs() > config.clip_ratio).float().mean().item()
                        ),
                        actor_gradient_norm=actor_norm,
                        critic_gradient_norm=critic_norm,
                        gradient_norm=float(norm),
                        gaussian_entropy=float(entropy.item()),
                    )
                )
            if kl_early_stop:
                break
        metrics = (
            {
                key: float(np.mean([row[key] for row in minibatch_metrics]))
                for key in minibatch_metrics[0]
            }
            if minibatch_metrics
            else {}
        )
        metrics.update(
            minibatch_updates=len(minibatch_metrics),
            kl_early_stop=kl_early_stop,
            return_min=float(returns.min()),
            return_max=float(returns.max()),
            log_std=float(self.model.log_std.item()),
        )
        return metrics

    def save(self, path):
        torch.save(
            dict(
                config=asdict(self.config),
                seed=self.seed,
                model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict(),
                updates=self.updates,
                steps=self.steps,
            ),
            path,
        )

    @classmethod
    def load(cls, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = state["model"]
        agent = cls(
            PPOConfig(**state["config"]),
            action_low=model["action_low"].numpy(),
            action_high=model["action_high"].numpy(),
            seed=state["seed"],
            observation_scales=model["scales"].numpy(),
        )
        agent.model.load_state_dict(model, strict=True)
        agent.model.eval()
        agent.updates = state["updates"]
        agent.steps = state["steps"]
        return agent
