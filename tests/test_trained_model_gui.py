#!/usr/bin/env python
"""test_trained_model_gui.py -- Roll out the latest trained PPO+DDQN checkpoint
from ``training/checkpoints`` inside the quadruped path-following environment
using the PyBullet *GUI* physics client (``p.GUI``).

The latest checkpoint is the one written by ``training/train_ppo.py`` every
run as ``ppo_ddqn_latest.pt`` (a plain ``torch.save`` dict containing the
``PPODDQNPolicy`` state, the saved action-bin centres, the training ``args``
and the recorded metrics).

What this script does
---------------------
1. Locates the latest checkpoint in ``training/checkpoints``
   (prefers ``ppo_ddqn_latest.pt``; falls back to the newest
   ``ppo_ddqn_epoch*.pt``).
2. Rebuilds the environment with ``render_mode="human"`` so the env connects
   to ``pybullet.connect(p.GUI)`` (see ``environment/gym_env.py``
   ``_ensure_physics_client``).
3. Rebuilds the ``PPODDQNPolicy`` from the checkpoint's ``args`` (depth shape
   and vector dim are read from the live env observation space, exactly as
   ``training/train_ppo.py`` does), loads the saved ``model_state``.
4. Runs the greedy Double-DQN argmax policy (the evaluation policy used by
   training) for the requested number of steps, resetting on terminal/
   truncated episodes and printing live progress.

Run from the project root inside the DL_env conda environment:

    conda activate DL_env
    python tests/test_trained_model_gui.py
    python tests/test_trained_model_gui.py --steps 300 --seed 1
    python tests/test_trained_model_gui.py --direct            # headless smoke test
    python tests/test_trained_model_gui.py --checkpoint \
        training/checkpoints/ppo_ddqn_epoch1.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Navigate up one level (tests/ -> project/) and add it to the search path.
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from environment import QuadrupedPathTerrainEnv  # noqa: E402
from environment.config import RobotConfig  # noqa: E402
from training.models import (  # noqa: E402
    NUM_ACTION_PARAMS,
    PPODDQNPolicy,
    depth_to_input,
    greedy_ddqn_action,
    vectorize_observation,
    VECTOR_KEYS,
)

DEFAULT_CHECKPOINT_DIR = Path("training/checkpoints")


# --------------------------------------------------------------------------- #
# Checkpoint discovery + model loading
# --------------------------------------------------------------------------- #
def find_latest_checkpoint(ckpt_dir: Path) -> Path:
    """Return the latest trained checkpoint under ``ckpt_dir``.

    Prefers the canonical ``ppo_ddqn_latest.pt`` that every training run
    writes after its final epoch. If it is absent, falls back to the
    chronologically newest ``ppo_ddqn_epoch*.pt`` file.
    """
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {ckpt_dir}")

    latest = ckpt_dir / "ppo_ddqn_latest.pt"
    if latest.is_file():
        return latest

    candidates = sorted(
        ckpt_dir.glob("ppo_ddqn_epoch*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"no 'ppo_ddqn_*.pt' checkpoint found in {ckpt_dir}"
        )
    return candidates[0]


def load_checkpoint(ckpt_path: Path, device: torch.device) -> dict:
    """Load a checkpoint dict from disk.

    Checkpoints written by ``training/train_ppo.py`` are plain
    ``torch.save`` dicts holding ``model_state``, ``bin_centers``,
    ``bin_edges``, ``args`` and ``metrics``.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = checkpoint["args"]
    bin_centers = np.asarray(checkpoint["bin_centers"], dtype=np.float32)

    num_bins = int(args.get("num_bins", bin_centers.shape[1]))
    emb_dim = int(args.get("emb_dim", 64))
    hidden_dim = int(args.get("hidden_dim", 256))

    print(
        f"[load] {ckpt_path.name}: epoch={checkpoint.get('epoch', '?')} "
        f"num_bins={num_bins} emb_dim={emb_dim} hidden_dim={hidden_dim}",
        flush=True,
    )
    if "metrics" in checkpoint:
        print(f"[load] training metrics: {checkpoint['metrics']}", flush=True)

    return checkpoint


def build_policy_from_checkpoint(
    checkpoint: dict, device: torch.device, depth_shape, vec_dim: int
) -> tuple[PPODDQNPolicy, np.ndarray, dict]:
    """Rebuild the policy with the env-derived input dims, then load weights."""
    args = checkpoint["args"]
    bin_centers = np.asarray(checkpoint["bin_centers"], dtype=np.float32)

    model = PPODDQNPolicy(
        depth_shape=depth_shape,
        vec_dim=vec_dim,
        num_bins=int(args.get("num_bins", bin_centers.shape[1])),
        emb_dim=int(args.get("emb_dim", 64)),
        hidden_dim=int(args.get("hidden_dim", 256)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, bin_centers, args


# --------------------------------------------------------------------------- #
# Environment construction
# --------------------------------------------------------------------------- #
def make_env(render_mode: str, max_episode_steps: int) -> QuadrupedPathTerrainEnv:
    """Build the gym env.

    ``render_mode='human'`` makes ``QuadrupedPathTerrainEnv`` connect to the
    PyBullet GUI physics client (``p.connect(p.GUI)``), giving a live 3D
    window with the random path drawn in green and the robot walking it.
    ``render_mode='direct'`` runs headless via ``p.DIRECT``.
    """
    return QuadrupedPathTerrainEnv(
        robot_config=RobotConfig(
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        ),
        dry_run=False,
    )


# --------------------------------------------------------------------------- #
# Rollout
# --------------------------------------------------------------------------- #
@torch.no_grad()
def rollout(
    env: QuadrupedPathTerrainEnv,
    model: PPODDQNPolicy,
    bin_centers: np.ndarray,
    device: torch.device,
    num_steps: int,
    seed: int,
) -> None:
    """Run the greedy Double-DQN policy in the environment."""
    obs, info = env.reset(seed=seed)

    total_reward = 0.0
    steps_done = 0
    episodes = 0
    falls = 0

    try:
        while steps_done < num_steps:
            vec = vectorize_observation(obs)[None, :]
            depth = depth_to_input(obs["depth"])[None, ...]

            action, _bins = greedy_ddqn_action(
                model, depth, vec, bin_centers, device
            )

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            steps_done += 1

            if steps_done % 10 == 0 or terminated or truncated:
                print(
                    f"step {steps_done:4d}/{num_steps}  reward={reward:+.3f} "
                    f"running_total={total_reward:+.3f}  "
                    f"roll={float(info.get('trunk_roll', 0.0)):+.2f} "
                    f"pitch={float(info.get('trunk_pitch', 0.0)):+.2f}  "
                    f"path_off={float(info.get('path_offset', 0.0)):+.3f} m  "
                    f"path_dist={float(info.get('path_distance', 0.0)):.3f} m  "
                    f"{'FALLEN' if info.get('fallen') else ''}"
                    f"{'REACHED_END' if info.get('reached_end') else ''}",
                    flush=True,
                )

            if terminated or truncated:
                if info.get("fallen"):
                    falls += 1
                episodes += 1
                obs, info = env.reset()

    except KeyboardInterrupt:
        print("\n[stopped] interrupted by user", flush=True)
    finally:
        env.close()

    print(
        f"\n[done] steps={steps_done}  episodes={episodes}  falls={falls}  "
        f"total_reward={total_reward:.3f}  "
        f"mean_reward={total_reward / max(steps_done, 1):.3f}",
        flush=True,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Roll out the latest trained PPO+DDQN checkpoint with p.GUI."
    )
    ap.add_argument(
        "--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR),
        help="directory containing ppo_ddqn_*.pt checkpoints",
    )
    ap.add_argument(
        "--checkpoint", default=None,
        help="explicit checkpoint file (overrides --checkpoint-dir discovery)",
    )
    ap.add_argument("--steps", type=int, default=200,
                    help="number of environment steps to roll out")
    ap.add_argument("--max-episode-steps", type=int, default=100,
                    help="max gait-cycle steps per episode before truncation")
    ap.add_argument("--seed", type=int, default=0, help="episode RNG seed")
    ap.add_argument(
        "--device", default="auto",
        help="'cpu', 'cuda', or 'auto'",
    )
    ap.add_argument(
        "--direct", action="store_true",
        help="run headless with p.DIRECT instead of the p.GUI window",
    )
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[config] device={device}", flush=True)

    # 1. Locate the latest checkpoint.
    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = find_latest_checkpoint(Path(args.checkpoint_dir))
    print(f"[config] checkpoint: {ckpt_path}", flush=True)

    # 2. Build the env first so we can read the live observation dims.
    render_mode = "direct" if args.direct else "human"
    env = make_env(render_mode=render_mode, max_episode_steps=args.max_episode_steps)

    depth_shape = tuple(env.observation_space["depth"].shape)
    vec_dim = int(sum(
        int(np.prod(env.observation_space[k].shape)) for k in VECTOR_KEYS
    ))
    print(f"[config] depth_shape={depth_shape}  vec_dim={vec_dim}  "
          f"render_mode={render_mode}  "
          f"physics={'GUI (p.GUI)' if render_mode == 'human' else 'DIRECT'}",
          flush=True)

    # 3. Build + load the trained policy.
    checkpoint = load_checkpoint(ckpt_path, device)
    model, bin_centers, ckpt_args = build_policy_from_checkpoint(
        checkpoint, device, depth_shape, vec_dim
    )
    assert bin_centers.shape == (NUM_ACTION_PARAMS, int(ckpt_args["num_bins"])), (
        f"bin_centers shape {bin_centers.shape} != "
        f"({NUM_ACTION_PARAMS}, {ckpt_args['num_bins']})"
    )

    # 4. Roll the greedy Double-DQN policy out in the env (GUI by default).
    print("[run] starting rollout ...", flush=True)
    rollout(env, model, bin_centers, device, num_steps=args.steps, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
