"""Readable twin-Q SAC with automatic temperature; no simulator planning calls.

Replay holds raw observations, normalized requested actions in [-1, 1], and
original rewards. Normalization is fixed by the caller using training data only.
Both termination and truncation stop bootstrap for our annual-reset protocol.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SACConfig:
    """Starting implementation defaults, not a selected reservoir study config."""
    observation_dim: int = 5
    action_dim: int = 1
    hidden_sizes: tuple[int, ...] = (64, 64)
    gamma: float = 0.995
    tau: float = 0.005
    learning_rate: float = 3e-4
    initial_alpha: float = 0.2
    target_entropy: float | None = None  # defaults to -action_dim, normalized units
    reward_scale: float = 0.001

    def __post_init__(self):
        if (self.observation_dim < 1 or self.action_dim < 1
                or not self.hidden_sizes or any(n < 1 for n in self.hidden_sizes)):
            raise ValueError("Positive observation/action/hidden dimensions required")
        if not (0 <= self.gamma <= 1 and 0 < self.tau <= 1):
            raise ValueError("Invalid gamma or tau")
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.learning_rate, self.initial_alpha, self.reward_scale)):
            raise ValueError("Finite positive learning rate, alpha, reward scale required")
        if self.target_entropy is not None and not math.isfinite(self.target_entropy):
            raise ValueError("Target entropy must be finite")


class ReplayBuffer:
    """Uniform ring buffer, with an independent sampling RNG and copied inputs."""
    def __init__(self, capacity, observation_dim, action_dim, seed=0):
        if min(capacity, observation_dim, action_dim) < 1:
            raise ValueError("Positive replay dimensions required")
        self.capacity, self.position, self.size = capacity, 0, 0
        self.rng = np.random.default_rng(seed)
        self.data = {
            "observations": np.empty((capacity, observation_dim), np.float32),
            "actions": np.empty((capacity, action_dim), np.float32),
            "rewards": np.empty((capacity, 1), np.float32),
            "next_observations": np.empty((capacity, observation_dim), np.float32),
            "ends": np.empty((capacity, 1), np.float32),
        }

    def __len__(self):
        return self.size

    def add(self, observation, action, reward, next_observation, terminated, truncated):
        values = dict(observations=observation, actions=action, rewards=[reward],
                      next_observations=next_observation,
                      ends=[bool(terminated or truncated)])
        converted = {}
        for key, value in values.items():
            arr = np.asarray(value, dtype=np.float32)
            if arr.shape != self.data[key].shape[1:] or not np.isfinite(arr).all():
                raise ValueError(f"Invalid replay {key}: shape or nonfinite value")
            converted[key] = arr
        if np.any(np.abs(converted["actions"]) > 1):
            raise ValueError("Replay actions must be normalized to [-1, 1]")
        for key, value in converted.items():
            self.data[key][self.position] = value
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device="cpu"):
        if batch_size < 1 or self.size < batch_size:
            raise ValueError("Replay must contain at least batch_size transitions")
        indices = self.rng.integers(self.size, size=batch_size)
        return {key: torch.as_tensor(value[indices], device=device)
                for key, value in self.data.items()}


def mlp(input_dim, hidden_sizes, output_dim):
    layers = []
    for width in hidden_sizes:
        layers.extend((nn.Linear(input_dim, width), nn.ReLU()))
        input_dim = width
    return nn.Sequential(*layers, nn.Linear(input_dim, output_dim))


class SquashedGaussianActor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = mlp(config.observation_dim, config.hidden_sizes, 2 * config.action_dim)

    def forward(self, observations, *, deterministic=False, generator=None):
        mean, log_std = self.net(observations).chunk(2, dim=-1)
        log_std = log_std.clamp(-20, 2)
        if deterministic:
            return mean.tanh(), None
        noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device,
                            generator=generator)
        latent = mean + log_std.exp() * noise  # reparameterized: gradients reach actor
        action = latent.tanh()
        gaussian_log_prob = -0.5 * (noise.square() + 2 * log_std + math.log(2 * math.pi))
        # Stable log(1 - tanh(z)^2), including saturated actions.
        log_jacobian = 2 * (math.log(2) - latent - F.softplus(-2 * latent))
        return action, (gaussian_log_prob - log_jacobian).sum(-1, keepdim=True)


class Critic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = mlp(config.observation_dim + config.action_dim, config.hidden_sizes, 1)

    def forward(self, observations, actions):
        return self.net(torch.cat((observations, actions), dim=-1))


def soft_bellman_target(rewards, ends, next_q, next_log_prob, alpha, gamma,
                        reward_scale=1.0):
    """Detached target; terminal observations are stored but never reset states."""
    with torch.no_grad():
        return reward_scale * rewards + gamma * (1 - ends) * (next_q - alpha * next_log_prob)


@torch.no_grad()
def polyak_update(source, target, tau):
    for current, slow in zip(source.parameters(), target.parameters(), strict=True):
        slow.lerp_(current, tau)


def checked_optimizer_step(loss, optimizer, parameters):
    """Reject nonfinite losses/gradients before mutating optimizer parameters."""
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite SAC loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [p.grad for p in parameters if p.grad is not None]
    norm = torch.stack([g.detach().norm() for g in gradients]).norm()
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite SAC gradient norm")
    optimizer.step()
    return norm.item()


class SACAgent(nn.Module):
    """Fixed observation transform, bounded actor, twin critics and target critics.

    Optimizers use normalized action densities, excluding the physical affine
    release transform. Targets/temperature therefore share normalized units.
    CPU implementation deliberately uses full float32 for this learning exercise.
    """
    def __init__(self, config=SACConfig(), *, observation_mean=None,
                 observation_scale=None, action_low=(0.0,), action_high=(491.61,), seed=0):
        super().__init__()
        self.config = config
        def vector(value, default, dimension):
            arr = np.asarray(default if value is None else value, dtype=np.float32)
            if arr.shape != (dimension,) or not np.isfinite(arr).all():
                raise ValueError("Invalid normalization or action bound vector")
            return torch.tensor(arr)
        self.register_buffer("observation_mean", vector(observation_mean,
                             np.zeros(config.observation_dim), config.observation_dim))
        self.register_buffer("observation_scale", vector(observation_scale,
                             np.ones(config.observation_dim), config.observation_dim))
        self.register_buffer("action_low", vector(action_low, None, config.action_dim))
        self.register_buffer("action_high", vector(action_high, None, config.action_dim))
        if (self.observation_scale <= 0).any() or (self.action_high <= self.action_low).any():
            raise ValueError("Positive scales and ordered action bounds required")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.actor = SquashedGaussianActor(config)
            self.q1, self.q2 = Critic(config), Critic(config)
        self.target_q1, self.target_q2 = deepcopy(self.q1), deepcopy(self.q2)
        self.target_q1.requires_grad_(False)
        self.target_q2.requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(config.initial_alpha)))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=config.learning_rate)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.learning_rate)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.updates = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def normalize(self, observation):
        return (observation - self.observation_mean) / self.observation_scale

    def physical_action(self, normalized):
        return self.action_low + (normalized + 1) * 0.5 * (self.action_high - self.action_low)

    @torch.no_grad()
    def act(self, observation, *, deterministic=False):
        obs = torch.as_tensor(np.asarray(observation), dtype=torch.float32)
        if obs.shape != (self.config.observation_dim,) or not torch.isfinite(obs).all():
            raise ValueError("Expected one finite raw observation")
        normalized, _ = self.actor(self.normalize(obs.unsqueeze(0)),
                                   deterministic=deterministic, generator=self.generator)
        return normalized[0].numpy().copy()

    def update(self, batch):
        """One critic step, one actor step, one temperature step, target averaging."""
        cfg = self.config
        obs = self.normalize(batch["observations"])
        following = self.normalize(batch["next_observations"])
        with torch.no_grad():
            next_action, next_logp = self.actor(following, generator=self.generator)
            next_q = torch.minimum(self.target_q1(following, next_action),
                                   self.target_q2(following, next_action))
            target = soft_bellman_target(batch["rewards"], batch["ends"], next_q,
                                         next_logp, self.alpha, cfg.gamma, cfg.reward_scale)
        q1, q2 = self.q1(obs, batch["actions"]), self.q2(obs, batch["actions"])
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        critic_grad = checked_optimizer_step(critic_loss, self.critic_optimizer,
                                            list(self.q1.parameters()) + list(self.q2.parameters()))
        self.critic_optimizer.zero_grad(set_to_none=True)

        # Freeze critic weights, but keep dQ/da so actor gradients pass through Q.
        self.q1.requires_grad_(False)
        self.q2.requires_grad_(False)
        try:
            action, logp = self.actor(obs, generator=self.generator)
            policy_q = torch.minimum(self.q1(obs, action), self.q2(obs, action))
            actor_loss = (self.alpha.detach() * logp - policy_q).mean()
            actor_grad = checked_optimizer_step(actor_loss, self.actor_optimizer,
                                               self.actor.parameters())
        finally:
            self.q1.requires_grad_(True)
            self.q2.requires_grad_(True)

        target_entropy = (-cfg.action_dim if cfg.target_entropy is None else cfg.target_entropy)
        # Standard log-temperature surrogate: entropy error cannot update the actor.
        alpha_loss = -(self.log_alpha * (logp.detach() + target_entropy)).mean()
        alpha_grad = checked_optimizer_step(alpha_loss, self.alpha_optimizer, [self.log_alpha])
        polyak_update(self.q1, self.target_q1, cfg.tau)
        polyak_update(self.q2, self.target_q2, cfg.tau)
        self.updates += 1
        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(),
                "alpha_loss": alpha_loss.item(), "alpha": self.alpha.item(),
                "entropy": -logp.detach().mean().item(),
                "q_disagreement": (q1 - q2).detach().abs().mean().item(),
                "target_mean": target.mean().item(), "target_min": target.min().item(),
                "target_max": target.max().item(), "critic_gradient_norm": critic_grad,
                "actor_gradient_norm": actor_grad, "alpha_gradient_norm": alpha_grad}

    def save(self, path):
        """Learner checkpoint, not a full environment/replay resume snapshot."""
        torch.save(dict(format_version=1, config=asdict(self.config), model=self.state_dict(),
                        optimizers=[o.state_dict() for o in (self.actor_optimizer,
                                    self.critic_optimizer, self.alpha_optimizer)],
                        generator=self.generator.get_state(), updates=self.updates), path)

    @classmethod
    def load(cls, path):
        """Only load trusted local checkpoints (optimizer payload uses pickle)."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["format_version"] != 1:
            raise ValueError("Unsupported SAC checkpoint")
        model = state["model"]
        agent = cls(SACConfig(**state["config"]),
                    observation_mean=model["observation_mean"].numpy(),
                    observation_scale=model["observation_scale"].numpy(),
                    action_low=model["action_low"].numpy(), action_high=model["action_high"].numpy())
        agent.load_state_dict(model)
        for optimizer, saved in zip((agent.actor_optimizer, agent.critic_optimizer,
                                     agent.alpha_optimizer), state["optimizers"], strict=True):
            optimizer.load_state_dict(saved)
        agent.generator.set_state(state["generator"])
        agent.updates = state["updates"]
        return agent
