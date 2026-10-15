"""Episode setup and signals for path-following locomotion training."""

import gymnasium as gym

from .task import PathTrainingTask, PathTaskConfig, EpisodePath
from .gym_env import QuadrupedPathTerrainEnv

# Register environment with Gymnasium
try:
    gym.register(
        id="QuadrupedPathTerrain-v0",
        entry_point="path_training.gym_env:QuadrupedPathTerrainEnv",
        max_episode_steps=3000,
    )
except Exception:
    pass

__all__ = [
    "PathTrainingTask",
    "PathTaskConfig",
    "EpisodePath",
    "QuadrupedPathTerrainEnv",
]
