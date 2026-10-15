#!/usr/bin/env python
"""Full Stable-Baselines3 PPO training script (headless) for the quadruped
path-following environment with the custom CNN depth-frame embedding +
DDQN network.

No GUI is used anywhere: the environment runs PyBullet in DIRECT mode
(``render_mode=None``).

Network / algorithm layout
--------------------------
* SB3 ``PPO`` is the training algorithm (clipped surrogate, GAE).
* ``DepthEmbeddingExtractor`` (a custom ``BaseFeaturesExtractor``) processes
  the observation dict:
      depth frame (48x64 metric) -> CNN -> depth embedding
      63-D vector obs (joints, orientation, velocity, contacts, previous
      action, path errors) -> bound-scaled
      concat -> shared MLP trunk -> features (``--features-dim``).
* ``CNNDDQNPolicy`` (a custom ``ActorCriticPolicy``) puts the SB3 actor and
  critic heads on those shared features, plus a **dueling per-parameter
  Double-Q head** (``DuelingParamQHead``). The action space is seen by SB3
  as ``MultiDiscrete([num_bins] * 26)``: one categorical over K bins per
  gait-cycle parameter.
* ``DiscretizeActionWrapper`` converts the sampled/argmax bin indices to
  the continuous 26-D normalized action that ``env.step`` receives, and
  records every transition into ``DDQNReplay``.
* ``DDQNAuxCallback._on_rollout_end`` trains the Q head with a Double-DQN
  TD loss each epoch (online net selects the next bin, soft-updated target
  net evaluates it; the shared CNN trunk is frozen during the auxiliary
  update, so PPO owns the representation).
* After training the latest checkpoint is reloaded from disk and the
  greedy Double-DQN argmax policy is evaluated (it decides the action
  sent to step).

Checkpoints
-----------
Every ``--save-every`` epochs (default 5) into ``--checkpoint-dir``
(default ``training/checkpoints_sb3``):

    ppo_cnn_ddqn_epoch_005.zip     # full SB3 model (policy, optimizer, ...)
    ddqn_head_epoch_005.pt         # Q head + target + aux optimizer + bins

Run from the project root inside the DL_env conda environment:

    conda activate DL_env
    python training/train_sb3_ppo.py                          # 80 epochs
    python training/train_sb3_ppo.py --epochs 80 --lr 3e-4 \
        --num-envs 4 --steps-per-epoch 512 --save-every 5
    python training/train_sb3_ppo.py --epochs 2 --steps-per-epoch 64  # smoke
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

import gymnasium as gym
from gymnasium import spaces

# Navigates up two levels (from script.py -> training/ -> project/) and
# adds it to Python's search path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.policies import ActorCriticPolicy  # noqa: E402
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from environment import QuadrupedPathTerrainEnv  # noqa: E402
from environment.config import RobotConfig  # noqa: E402
from training.models import (  # noqa: E402
    DEPTH_SCALE,
    NUM_ACTION_PARAMS,
    VECTOR_KEYS,
    VECTOR_SCALE,
    DepthCNN,
    DuelingParamQHead,
    build_action_bins,
    depth_to_input,
)
try:
    from training.lr_schedules import ExponentialDecaySchedule  # noqa: E402
except Exception:  # pragma: no cover
    ExponentialDecaySchedule = None  # type: ignore



# ----------------------------------------------------------------------
# Observation helpers
# ----------------------------------------------------------------------
def raw_vector(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate the raw (unscaled) vector-obs keys in VECTOR_KEYS order."""
    return np.concatenate([
        np.asarray(obs[key], dtype=np.float32).reshape(-1)
        for key in VECTOR_KEYS
    ]).astype(np.float32)


def vector_shapes(observation_space: spaces.Dict) -> List[int]:
    """Per-key flat sizes of the vector observation (sum = 63)."""
    return [int(np.prod(observation_space[key].shape)) for key in VECTOR_KEYS]


# ----------------------------------------------------------------------
# DDQN replay (records transitions the wrapper sees)
# ----------------------------------------------------------------------
class DDQNReplay:
    """Fixed-capacity transition store for the Double-DQN auxiliary loss.

    Depth frames are quantized to uint8 in [0, 1] to keep memory flat
    (8000 transitions ~= 100 MB); vectors stay float32.
    """

    def __init__(self, capacity: int, seed: int = 0):
        self.data: deque = deque(maxlen=int(capacity))
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.data)

    def add(self, depth01, vec_raw, action_bins, reward, terminated,
            next_depth01, next_vec_raw):
        self.data.append((
            np.round(np.asarray(depth01, np.float32) * 255.0).astype(np.uint8),
            np.asarray(vec_raw, np.float32),
            np.asarray(action_bins, np.int64),
            np.float32(reward),
            np.bool_(terminated),
            np.round(np.asarray(next_depth01, np.float32) * 255.0).astype(np.uint8),
            np.asarray(next_vec_raw, np.float32),
        ))

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = self.rng.integers(0, len(self.data), size=batch_size)
        cols = list(zip(*(self.data[int(i)] for i in idx)))
        return {
            "depth": np.stack(cols[0]),
            "vec": np.stack(cols[1]),
            "actions": np.stack(cols[2]),
            "rewards": np.asarray(cols[3], np.float32),
            "terminated": np.asarray(cols[4], np.float32),
            "next_depth": np.stack(cols[5]),
            "next_vec": np.stack(cols[6]),
        }


# ----------------------------------------------------------------------
# Custom network pieces
# ----------------------------------------------------------------------
class DepthEmbeddingExtractor(BaseFeaturesExtractor):
    """CNN over the depth frame + scaled vector obs -> shared features."""

    def __init__(self, observation_space: spaces.Dict,
                 features_dim: int = 256, emb_dim: int = 64):
        super().__init__(observation_space, features_dim=features_dim)
        if "depth" not in observation_space.spaces:
            raise ValueError("observation space must contain a 'depth' key")
        depth_shape = observation_space["depth"].shape
        vec_dim = int(sum(
            np.prod(observation_space[key].shape) for key in VECTOR_KEYS))
        self.depth_cnn = DepthCNN(depth_shape, emb_dim=emb_dim)
        self.trunk = nn.Sequential(
            nn.Linear(emb_dim + vec_dim, features_dim),
            nn.ReLU(inplace=True),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(features_dim),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # metric depth -> [0, 1] (stateless, matches training.models)
        depth01 = torch.clamp(observations["depth"] / DEPTH_SCALE, 0.0, 1.0)
        emb = self.depth_cnn(depth01)
        vec = torch.cat([
            observations[key] / scale
            for key, scale in zip(VECTOR_KEYS, VECTOR_SCALE)
        ], dim=-1)
        vec = torch.clamp(vec, -10.0, 10.0)
        return self.trunk(torch.cat([emb, vec], dim=-1))


class CNNDDQNPolicy(ActorCriticPolicy):
    """SB3 actor-critic policy with a dueling Double-Q head attached.

    The Q head is not part of the PPO loss; it is trained by the
    auxiliary Double-DQN loss in :class:`DDQNAuxCallback` and decides
    the evaluation action (per-parameter argmax -> bin -> continuous).
    """

    def __init__(self, observation_space, action_space, lr_schedule,
                 *args, num_bins: int = 9, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule,
                         *args, **kwargs)
        self.num_bins = int(num_bins)
        self.ddqn_head = DuelingParamQHead(
            self.features_extractor.features_dim,
            NUM_ACTION_PARAMS,
            self.num_bins,
        )


# ----------------------------------------------------------------------
# Env wrapper: MultiDiscrete bins -> continuous action, transition record
# ----------------------------------------------------------------------
class DiscretizeActionWrapper(gym.Wrapper):
    """Expose a MultiDiscrete(K)^26 action space and record transitions.

    ``action`` is one bin index per gait parameter; the wrapper converts
    it to the bin centre in the env's normalized [-1, 1]^26 space (what
    ``QuadrupedPathTerrainEnv.step`` expects) and stores
    (s, bins, r, s', terminated) in the DDQN replay.
    """

    def __init__(self, env: gym.Env, bin_centers: np.ndarray,
                 replay: DDQNReplay):
        super().__init__(env)
        self.bin_centers = np.asarray(bin_centers, dtype=np.float32)
        self.replay = replay
        self.action_space = spaces.MultiDiscrete(
            np.full(self.bin_centers.shape[0], self.bin_centers.shape[1],
                    dtype=np.int64))
        self._last_obs: Optional[Dict[str, np.ndarray]] = None

    def _to_continuous(self, action) -> np.ndarray:
        idx = np.asarray(action, dtype=np.int64).reshape(-1)
        return self.bin_centers[np.arange(idx.size), idx].astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        continuous = self._to_continuous(action)
        obs, reward, terminated, truncated, info = self.env.step(continuous)
        if self._last_obs is not None and self.env.use_depth:
            self.replay.add(
                depth01=depth_to_input(self._last_obs["depth"]),
                vec_raw=raw_vector(self._last_obs),
                action_bins=np.asarray(action, dtype=np.int64),
                reward=float(reward),
                terminated=bool(terminated),
                next_depth01=depth_to_input(obs["depth"]),
                next_vec_raw=raw_vector(obs),
            )
        self._last_obs = obs
        return obs, reward, terminated, truncated, info


# ----------------------------------------------------------------------
# Auxiliary Double-DQN training
# ----------------------------------------------------------------------
def batch_to_obs_dict(depth_u8: np.ndarray, vec_raw: np.ndarray,
                      shapes: List[int], device: torch.device
                      ) -> Dict[str, torch.Tensor]:
    """Replay batch -> raw observation dict for the shared extractor.

    Depth is dequantized back to metric meters and the raw vector is
    split per key, exactly matching what the extractor sees from SB3.
    """
    depth = torch.as_tensor(depth_u8, dtype=torch.float32, device=device)
    depth = depth / 255.0 * DEPTH_SCALE
    vec = torch.as_tensor(vec_raw, dtype=torch.float32, device=device)
    parts = torch.split(vec, shapes, dim=-1)
    return {"depth": depth,
            **{key: part for key, part in zip(VECTOR_KEYS, parts)}}


def train_ddqn_head(policy: CNNDDQNPolicy, target_head: DuelingParamQHead,
                    optimizer: torch.optim.Optimizer, replay: DDQNReplay,
                    shapes: List[int], args, device: torch.device
                    ) -> Dict[str, float]:
    """k gradient steps of the Double-DQN TD loss on the Q head only.

    The shared CNN trunk runs under ``no_grad`` so PPO owns the
    representation; selection uses the online head, evaluation the
    target head (the "double" in Double DQN).
    """
    if len(replay) < max(args.td_batch_size, 2):
        return {}
    losses, q_means = [], []
    for _ in range(args.td_steps):
        batch = replay.sample(args.td_batch_size)
        obs_dict = batch_to_obs_dict(batch["depth"], batch["vec"],
                                     shapes, device)
        next_dict = batch_to_obs_dict(batch["next_depth"], batch["next_vec"],
                                      shapes, device)
        with torch.no_grad():  # freeze the shared trunk for the aux update
            feats = policy.features_extractor(obs_dict)
            feats_next = policy.features_extractor(next_dict)

        q = policy.ddqn_head(feats)                       # (B, 26, K)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long,
                                  device=device)
        q_sa = q.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            a_star = policy.ddqn_head(feats_next).argmax(-1)   # online selects
            q_next = target_head(feats_next) \
                .gather(-1, a_star.unsqueeze(-1)).squeeze(-1)  # target evaluates
            bootstrap = q_next.mean(dim=-1)               # factored Q(s')
            rewards = torch.as_tensor(batch["rewards"], device=device)
            terminated = torch.as_tensor(batch["terminated"], device=device)
            td_target = (rewards + args.gamma
                         * (1.0 - terminated) * bootstrap)

        td_loss = nn.functional.mse_loss(
            q_sa, td_target.unsqueeze(1).expand_as(q_sa))

        optimizer.zero_grad(set_to_none=True)
        td_loss.backward()
        optimizer.step()

        # soft target update (Polyak averaging)
        with torch.no_grad():
            for tp, hp in zip(target_head.parameters(),
                              policy.ddqn_head.parameters()):
                tp.mul_(1.0 - args.target_tau).add_(hp, alpha=args.target_tau)

        losses.append(float(td_loss.detach()))
        q_means.append(float(q_sa.detach().abs().mean()))

    return {"td_loss": float(np.mean(losses)),
            "q_abs_mean": float(np.mean(q_means)),
            "replay_size": float(len(replay))}


class DDQNAuxCallback(BaseCallback):
    """Trains the DDQN head once per rollout (i.e. once per epoch)."""

    def __init__(self, replay: DDQNReplay, target_head: DuelingParamQHead,
                 aux_optimizer: torch.optim.Optimizer, shapes: List[int],
                 args, device: torch.device):
        super().__init__(verbose=0)
        self.replay = replay
        self.target_head = target_head
        self.aux_optimizer = aux_optimizer
        self.shapes = shapes
        self.args = args
        self.device = device
        self.td_info: Dict[str, float] = {}
        self._epoch_rewards: List[float] = []
        self._epoch_lengths: List[int] = []
        self.last_episode_stats: Dict[str, float] = {}

    def _on_step(self) -> bool:
        # Collect per-episode stats from Monitor's info["episode"] so each
        # epoch reports exact (non-cumulative) reward/length means.
        for info in self.locals.get("infos", ()):
            episode = info.get("episode") if isinstance(info, dict) else None
            if episode is not None:
                self._epoch_rewards.append(float(episode["r"]))
                self._epoch_lengths.append(int(episode["l"]))
        return True

    def _on_rollout_end(self) -> None:
        self.td_info = train_ddqn_head(
            self.model.policy, self.target_head, self.aux_optimizer,
            self.replay, self.shapes, self.args, self.device)
        if self._epoch_rewards:
            self.last_episode_stats = {
                "ep_rew": float(np.mean(self._epoch_rewards)),
                "ep_len": float(np.mean(self._epoch_lengths)),
                "ep_count": float(len(self._epoch_rewards)),
            }
            self._epoch_rewards.clear()
            self._epoch_lengths.clear()
        else:
            self.last_episode_stats = {}


# ----------------------------------------------------------------------
# Greedy Double-DQN evaluation (Q head decides the action for step)
# ----------------------------------------------------------------------
def greedy_ddqn_eval(model: PPO, bin_centers: np.ndarray, steps: int,
                     seed: int, device: torch.device,
                     max_episode_steps: int) -> float:
    """Run ``steps`` env steps with the Double-Q argmax action."""
    env = QuadrupedPathTerrainEnv(
        robot_config=RobotConfig(max_episode_steps=max_episode_steps))
    obs, _info = env.reset(seed=seed)
    total, episodes, falls = 0.0, 0, 0
    for _ in range(steps):
        obs_dict = {
            "depth": torch.as_tensor(
                np.asarray(obs["depth"], dtype=np.float32)[None],
                device=device),
        }
        for key in VECTOR_KEYS:
            obs_dict[key] = torch.as_tensor(
                np.asarray(obs[key], dtype=np.float32)[None], device=device)
        with torch.no_grad():
            feats = model.policy.features_extractor(obs_dict)
            bins = model.policy.ddqn_head(feats) \
                .argmax(dim=-1).squeeze(0).cpu().numpy()
        action = bin_centers[np.arange(NUM_ACTION_PARAMS), bins] \
            .astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        if terminated or truncated:
            episodes += 1
            falls += int(bool(info.get("fallen", False)))
            obs, _info = env.reset()
    env.close()
    print(f"[greedy DDQN eval] {steps} steps, reward={total:.3f}, "
          f"episodes={episodes}, falls={falls}", flush=True)
    return total


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Headless SB3 PPO + CNN depth embedding + DDQN training")
    # training schedule
    ap.add_argument("--epochs", type=int, default=80,
                    help="number of PPO training epochs (rollout + update)")
    ap.add_argument("--steps-per-epoch", type=int, default=256,
                    help="total environment steps collected per epoch")
    ap.add_argument("--num-envs", type=int, default=2)
    ap.add_argument("--max-episode-steps", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=10,
                    help="checkpoint every N epochs")
    ap.add_argument("--checkpoint-dir", default="training/checkpoints_sb3")
    # PPO hyperparameters (SB3 PPO arguments)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-init", type=float, default=2e-4)
    ap.add_argument("--lr-final", type=float, default=5e-6)
    ap.add_argument("--lr-decay-rate", type=float, default=5.0)
    ap.add_argument("--lr-schedule", choices=["constant", "exp"], default="exp",
                    help="learning rate schedule")
    ap.add_argument("--ppo-update-epochs", type=int, default=10,
                    help="gradient passes over each rollout (SB3 n_epochs)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-range", type=float, default=0.15)
    ap.add_argument("--ent-coef", type=float, default=0.005)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--target-kl", type=float, default=0.01,
                    help="0 disables the KL early-stop")
    ap.add_argument("--no-normalize-advantage", action="store_true",
                    help="disable advantage normalization")
    # network
    ap.add_argument("--features-dim", type=int, default=256)
    ap.add_argument("--emb-dim", type=int, default=64,
                    help="CNN depth embedding dimension")
    ap.add_argument("--mlp-size", type=int, default=64,
                    help="actor/critic MLP width")
    ap.add_argument("--num-bins", type=int, default=9,
                    help="discretization bins per gait parameter")
    # DDQN auxiliary training
    ap.add_argument("--ddqn-lr", type=float, default=3e-4)
    ap.add_argument("--td-batch-size", type=int, default=64)
    ap.add_argument("--td-steps", type=int, default=30,
                    help="auxiliary DDQN gradient steps per epoch")
    ap.add_argument("--td-replay-size", type=int, default=4000)
    ap.add_argument("--target-tau", type=float, default=0.01,
                    help="soft target-network update rate")
    # misc
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto",
                    help="'cpu', 'cuda', or 'auto'")
    ap.add_argument("--eval-steps", type=int, default=20,
                    help="greedy DDQN eval steps after training (0 = skip)")
    ap.add_argument("--require-stable", action="store_true",
                    help="exit non-zero if non-finite metrics are seen")
    return ap.parse_args(argv)


# ----------------------------------------------------------------------
# Checkpointing
# ----------------------------------------------------------------------
def save_checkpoint(model: PPO, callback: DDQNAuxCallback, args,
                    epoch_tag: str, ckpt_dir: Path,
                    bin_edges: np.ndarray, bin_centers: np.ndarray) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(ckpt_dir / f"ppo_cnn_ddqn_{epoch_tag}"))
    torch.save({
        "epoch_tag": epoch_tag,
        "ddqn_head_state": model.policy.ddqn_head.state_dict(),
        "target_head_state": callback.target_head.state_dict(),
        "aux_optimizer_state": callback.aux_optimizer.state_dict(),
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "args": vars(args),
        "td_info": callback.td_info,
    }, ckpt_dir / f"ddqn_head_{epoch_tag}.pt")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.steps_per_epoch % args.num_envs != 0:
        raise SystemExit("--steps-per-epoch must be divisible by --num-envs")
    n_steps = args.steps_per_epoch // args.num_envs
    if n_steps < 1:
        raise SystemExit("--steps-per-epoch too small")
    if args.batch_size > args.steps_per_epoch:
        raise SystemExit("--batch-size must be <= --steps-per-epoch")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    bin_edges, bin_centers = build_action_bins(args.num_bins)
    replay = DDQNReplay(args.td_replay_size, seed=args.seed)

    # --- headless vectorized environments ----------------------------
    def make_env_fn(rank: int):
        def _init():
            env = QuadrupedPathTerrainEnv(
                robot_config=RobotConfig(
                    max_episode_steps=args.max_episode_steps),
                render_mode= None,          # never a GUI: DIRECT sim only
            )
            # Monitor must be outermost so its "episode" info reaches SB3
            # (rollout/ep_rew_mean, ep_len_mean)
            return Monitor(DiscretizeActionWrapper(env, bin_centers, replay))
        return _init

    vec_env = DummyVecEnv([make_env_fn(i) for i in range(args.num_envs)])
    obs_space = vec_env.observation_space
    shapes = vector_shapes(obs_space)
    assert int(sum(shapes)) == 63, "unexpected vector observation size"

    policy_kwargs = dict(
        features_extractor_class=DepthEmbeddingExtractor,
        features_extractor_kwargs=dict(
            features_dim=args.features_dim, emb_dim=args.emb_dim),
        net_arch=dict(pi=[args.mlp_size, args.mlp_size],
                      vf=[args.mlp_size, args.mlp_size]),
        activation_fn=nn.ReLU,
        num_bins=args.num_bins,
    )

    lr = args.lr
    if args.lr_schedule == "exp" and ExponentialDecaySchedule is not None:
        lr = ExponentialDecaySchedule(args.lr_init, args.lr_final, args.lr_decay_rate)
    elif args.lr_schedule == "exp":
        lr = args.lr_init  # fallback
    else:
        lr = args.lr

    model = PPO(
        CNNDDQNPolicy,
        vec_env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=args.batch_size,
        n_epochs=args.ppo_update_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        normalize_advantage=not args.no_normalize_advantage,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=(args.target_kl if args.target_kl > 0 else None),
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
        device=device,
        tensorboard_log=None,
    )

    # --- DDQN auxiliary machinery ------------------------------------
    target_head = copy.deepcopy(model.policy.ddqn_head).to(device)
    for p in target_head.parameters():
        p.requires_grad_(False)
    aux_optimizer = torch.optim.Adam(model.policy.ddqn_head.parameters(),
                                     lr=args.ddqn_lr)
    callback = DDQNAuxCallback(replay, target_head, aux_optimizer, shapes,
                               args, device)

    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"render_mode=human  device={device.type}  "
          f"policy_params={n_params:,}  bins/param={args.num_bins}  "
          f"envs={args.num_envs}  n_steps={n_steps}  "
          f"steps/epoch={args.steps_per_epoch}  epochs={args.epochs}  "
          f"save_every={args.save_every}", flush=True)

    ckpt_dir = Path(args.checkpoint_dir)
    save_tags: List[str] = []
    unstable_reasons: List[str] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.learn(total_timesteps=args.steps_per_epoch,
                    reset_num_timesteps=False,
                    callback=callback,
                    log_interval=1,
                    progress_bar=False)
        elapsed = time.perf_counter() - t0

        values = dict(model.logger.name_to_value)   # verbose=0 -> never cleared

        def _finite(key: str) -> bool:
            val = values.get(key)
            if val is None:
                return True
            try:  # SB3 logs numpy scalars (np.float32 etc), not only floats
                return math.isfinite(float(val))
            except (TypeError, ValueError):
                return False

        pg = values.get("train/policy_gradient_loss")
        vf = values.get("train/value_loss")
        ent = values.get("train/entropy_loss")
        kl = values.get("train/approx_kl")
        clipf = values.get("train/clip_fraction")
        # exact per-epoch episode stats from the callback (Monitor info)
        ep_stats = callback.last_episode_stats
        ep_rew = ep_stats.get("ep_rew", values.get("rollout/ep_rew_mean"))
        ep_len = ep_stats.get("ep_len", values.get("rollout/ep_len_mean"))
        ep_count = ep_stats.get("ep_count")
        td = callback.td_info.get("td_loss")

        reasons = [k for k in ("train/policy_gradient_loss",
                               "train/value_loss",
                               "train/entropy_loss",
                               "train/approx_kl") if not _finite(k)]
        if td is not None and not math.isfinite(td):
            reasons.append("td_loss")
        if ep_rew is not None and not math.isfinite(float(ep_rew)):
            reasons.append("ep_rew_mean")
        unstable_reasons.extend(f"epoch {epoch}: {r}" for r in reasons)

        def fmt(val, spec=".4f"):
            return "n/a" if val is None else format(float(val), spec)

        print(f"[epoch {epoch}/{args.epochs}] {elapsed:6.1f}s  "
              f"ep_rew={fmt(ep_rew, '.3f')} "
              f"ep_len={fmt(ep_len, '.0f')} "
              f"episodes={('n/a' if ep_count is None else str(int(ep_count)))}  "
              f"pg={fmt(pg)} v={fmt(vf)} "
              f"H={'n/a' if ent is None else format(-float(ent), '.2f')} "
              f"kl={fmt(kl)} clip={fmt(clipf)} "
              f"td={fmt(td)} replay={len(replay)}"
              f"{'  UNSTABLE:' + ','.join(reasons) if reasons else ''}",
              flush=True)

        if epoch % args.save_every == 0:
            tag = f"epoch_{epoch:03d}"
            save_checkpoint(model, callback, args, tag, ckpt_dir,
                            bin_edges, bin_centers)
            save_checkpoint(model, callback, args, "latest", ckpt_dir,
                            bin_edges, bin_centers)
            save_tags.append(tag)
            print(f"  -> checkpoint saved: "
                  f"{ckpt_dir / f'ppo_cnn_ddqn_{tag}.zip'} + "
                  f"{ckpt_dir / f'ddqn_head_{tag}.pt'}", flush=True)

    # guarantee a latest checkpoint even if epochs % save_every != 0
    if not save_tags or save_tags[-1] != "latest":
        save_checkpoint(model, callback, args, "latest", ckpt_dir,
                        bin_edges, bin_centers)

    # --- reload from disk and evaluate the greedy DDQN policy --------
    if args.eval_steps > 0:
        zip_path = ckpt_dir / "ppo_cnn_ddqn_latest.zip"
        pt_path = ckpt_dir / "ddqn_head_latest.pt"
        loaded = PPO.load(str(zip_path), device=device)
        ckpt = torch.load(pt_path, map_location=device, weights_only=False)
        loaded.policy.ddqn_head.load_state_dict(ckpt["ddqn_head_state"])
        print(f"[reload] {zip_path.name} + {pt_path.name} loaded "
              f"(epoch tag: {ckpt['epoch_tag']})", flush=True)
        greedy_ddqn_eval(loaded, bin_centers, args.eval_steps,
                         seed=args.seed + 42, device=device,
                         max_episode_steps=args.max_episode_steps)

    vec_env.close()

    if unstable_reasons:
        print("STABILITY: FAIL -> " + "; ".join(unstable_reasons[:8]),
              flush=True)
        if args.require_stable:
            return 2
    else:
        print("STABILITY: PASS (all epoch metrics finite)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
