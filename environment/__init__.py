"""Refactored quadruped path-following environment package.

Modules:
  - config    : dataclass configuration objects (path, terrain, action).
  - task      : random continuous Catmull-Rom path episode + path metrics.
  - terrain   : dense packed-pyramid terrain generator.
  - ik        : analytic IK + per-cycle crawl gait generator for the A1.
  - gym_env   : the Gymnasium environment that wires everything together.
"""

from .config import (
    ActionConfig,
    PathTaskConfig,
    RandomTerrainConfig,
    RobotConfig,
)
from .task import EpisodePath, PathTrainingTask, _catmull_rom
from .terrain import create_random_terrain, terrain_layout
from .gym_env import QuadrupedPathTerrainEnv

__all__ = [
    "ActionConfig",
    "PathTaskConfig",
    "RandomTerrainConfig",
    "RobotConfig",
    "EpisodePath",
    "PathTrainingTask",
    "create_random_terrain",
    "terrain_layout",
    "QuadrupedPathTerrainEnv",
]