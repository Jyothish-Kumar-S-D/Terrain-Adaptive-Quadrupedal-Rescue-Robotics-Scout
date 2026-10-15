"""Networks for PPO training with a CNN depth embedding and DDQN head.

Data flow (matches the training design):

    depth camera feed (H x W metric depth)
        -> CNN encoder                      -> depth embedding
    vector observation (63-D gait/path params, bound-scaled)
    concat(depth embedding, vector obs)     -> shared MLP trunk
    trunk features fan out to three heads:
        * PPO policy head : 26 x K categorical logits (one K-bin categorical
          per gait-cycle parameter -- the PPO stochastic policy)
        * PPO value head  : V(s) for GAE
        * dueling Double-Q: per-parameter dueling streams Q_i(s, a_i)
          (value stream V_i + advantage stream A_i, merged per parameter)

Each of the 26 gait-cycle parameters is discretized into K bins between its
action-space bounds. During PPO rollouts a parameter's bin is sampled from
the policy and the bin centre is sent to env.step(); at evaluation time the
Double-DQN argmax over the Q heads decides the bin (and therefore the
continuous action) sent to step(). The Q heads are trained with a
Double-DQN TD loss (online net selects the next bin, target net evaluates).
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from environment.action import (
    NOMINAL_ACTION_PARAMS,
    normalize_action,
)

# Observation keys that form the vector observation, in fixed order.
# Shapes sum to the 63-D non-depth part of the observation space.
VECTOR_KEYS = (
    "joint_positions",      # 12
    "joint_velocities",     # 12
    "trunk_orientation",    # 3
    "base_linear_velocity", # 3
    "foot_contacts",        # 4
    "previous_action",      # 26
    "path_offset",          # 1
    "path_distance",        # 1
    "velocity_tangent_angle",  # 1
)

# Per-key bound used to scale each block of the vector observation into
# roughly [-1, 1]. Derived from the declared observation-space bounds so the
# scaling is stateless (nothing to checkpoint, identical at eval time).
VECTOR_SCALE = (
    4.30,   # joint positions: URDF limits +/- 0.10 slack
    42.0,   # joint velocities: +/- 2 * max_motor_velocity
    math.pi + 0.05,   # roll/pitch/yaw
    20.0,   # base linear velocity
    1.0,    # foot contacts (already 0/1)
    1.0,    # previous action (already in [-1, 1])
    20.0,   # path offset
    20.0,   # path distance
    math.pi + 0.05,   # velocity-tangent angle
)

NUM_ACTION_PARAMS = len(NOMINAL_ACTION_PARAMS)  # 26 gait-cycle parameters
DEPTH_SCALE = 10.0  # metric depth buffer spans [0, 10] m


def build_action_bins(num_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-parameter bin edges (26, K+1) and centres (26, K).

    Bins span the env's *normalized* action space [-1, 1]^26 -- what
    ``env.step`` expects (it denormalizes to physical gait parameters
    internally). Each parameter therefore gets the same K bin edges,
    and every bin centre is a valid action value by construction.
    """
    if num_bins < 2:
        raise ValueError("num_bins must be >= 2")
    edges = np.linspace(-1.0, 1.0, num_bins + 1,
                        dtype=np.float32)              # (K+1,)
    edges = np.broadcast_to(edges, (NUM_ACTION_PARAMS, num_bins + 1)).copy()
    centers = (0.5 * (edges[:, :-1] + edges[:, 1:])).astype(np.float32)
    return edges, centers


def nominal_bin_indices(num_bins: int) -> np.ndarray:
    """Bin index of the nominal crawl-gait parameter for each action dim."""
    _, centers = build_action_bins(num_bins)
    nominal = normalize_action(NOMINAL_ACTION_PARAMS)
    return np.abs(centers - nominal[:, None]).argmin(axis=1).astype(np.int64)


def vectorize_observation(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """Stack the dict observation's vector keys into one scaled 63-D row."""
    parts = [
        np.asarray(obs[key], dtype=np.float32).reshape(-1) / scale
        for key, scale in zip(VECTOR_KEYS, VECTOR_SCALE)
    ]
    return np.clip(np.concatenate(parts), -10.0, 10.0).astype(np.float32)


def depth_to_input(depth: np.ndarray) -> np.ndarray:
    """Metric depth frame -> network input in [0, 1]."""
    return np.clip(np.asarray(depth, dtype=np.float32) / DEPTH_SCALE,
                   0.0, 1.0)


class DepthCNN(nn.Module):
    """Small CNN over the (H, W) metric depth frame -> depth embedding."""

    def __init__(self, depth_shape: Tuple[int, int], emb_dim: int = 64):
        super().__init__()
        height, width = depth_shape
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        # Compute conv output size for the configured depth resolution.
        h, w = float(height), float(width)
        for kernel, stride in ((8, 4), (4, 2), (3, 1)):
            h = math.floor((h - kernel) / stride) + 1
            w = math.floor((w - kernel) / stride) + 1
        if h < 1 or w < 1:
            raise ValueError(
                f"depth resolution {depth_shape} too small for the CNN stack"
            )
        self.proj = nn.Linear(64 * int(h) * int(w), emb_dim)
        self.emb_dim = emb_dim

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        # depth: (B, H, W) in [0, 1]
        x = depth.unsqueeze(1)
        return torch.relu(self.proj(self.conv(x)))


class DuelingParamQHead(nn.Module):
    """Dueling decomposition applied per gait parameter.

    Q_i(s, a_i) = V_i(s) + A_i(s, a_i) - mean_{a_i} A_i(s, a_i)
    """

    def __init__(self, hidden_dim: int, num_params: int, num_bins: int):
        super().__init__()
        self.num_params = num_params
        self.num_bins = num_bins
        self.value_stream = nn.Linear(hidden_dim, num_params)
        self.advantage_stream = nn.Linear(hidden_dim, num_params * num_bins)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch = features.shape[0]
        v = self.value_stream(features).unsqueeze(-1)          # (B, P, 1)
        a = self.advantage_stream(features).view(
            batch, self.num_params, self.num_bins)             # (B, P, K)
        return v + a - a.mean(dim=-1, keepdim=True)            # dueling merge


class PPODDQNPolicy(nn.Module):
    """Shared trunk with PPO policy/value heads and a dueling Double-Q head."""

    def __init__(
        self,
        depth_shape: Tuple[int, int],
        vec_dim: int = 63,
        num_bins: int = 9,
        emb_dim: int = 64,
        hidden_dim: int = 256,
        init_to_nominal: bool = True,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.depth_encoder = DepthCNN(depth_shape, emb_dim=emb_dim)
        self.trunk = nn.Sequential(
            nn.Linear(emb_dim + vec_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
        )
        self.policy_head = nn.Linear(
            hidden_dim, NUM_ACTION_PARAMS * num_bins)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.q_head = DuelingParamQHead(
            hidden_dim, NUM_ACTION_PARAMS, num_bins)

        if init_to_nominal:
            # Bias the initial categorical policy toward the known-good
            # nominal crawl gait (soft peak: +1.0 logit, not deterministic)
            # so early rollouts are not pure random gait parameters.
            bias = torch.zeros(NUM_ACTION_PARAMS * num_bins)
            idx = torch.from_numpy(nominal_bin_indices(num_bins))
            bias.view(NUM_ACTION_PARAMS, num_bins).scatter_(
                1, idx.unsqueeze(1), 1.0)
            with torch.no_grad():
                self.policy_head.bias.copy_(bias)

    # -- forward pieces ------------------------------------------------
    def features(self, depth: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        emb = self.depth_encoder(depth)
        return self.trunk(torch.cat([emb, vec], dim=-1))

    def policy(self, features: torch.Tensor) -> Categorical:
        logits = self.policy_head(features).view(
            -1, NUM_ACTION_PARAMS, self.num_bins)
        return Categorical(logits=logits)

    def value(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_head(features).squeeze(-1)

    def q_values(self, features: torch.Tensor) -> torch.Tensor:
        """Per-parameter Q values, shape (B, 26, K)."""
        return self.q_head(features)

    # -- convenience wrappers -----------------------------------------
    @staticmethod
    def to_tensors(
        depth: np.ndarray, vec: np.ndarray, device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert already-scaled network inputs (depth in [0, 1] via
        :func:`depth_to_input`, vector obs scaled via
        :func:`vectorize_observation`) to device tensors."""
        depth_t = torch.as_tensor(depth, dtype=torch.float32, device=device)
        vec_t = torch.as_tensor(vec, dtype=torch.float32, device=device)
        return depth_t, vec_t


def greedy_ddqn_action(
    net: PPODDQNPolicy,
    depth: np.ndarray,
    vec: np.ndarray,
    bin_centers: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Double-DQN argmax action selection (evaluation).

    Returns (continuous action (26,), chosen bin indices (26,)).
    """
    net.eval()
    with torch.no_grad():
        depth_t, vec_t = net.to_tensors(depth, vec, device)
        features = net.features(depth_t, vec_t)
        q = net.q_values(features)                      # (1, 26, K)
        bins = q.argmax(dim=-1).squeeze(0).cpu().numpy()
    action = bin_centers[
        np.arange(NUM_ACTION_PARAMS), bins].astype(np.float32)
    return action, bins
