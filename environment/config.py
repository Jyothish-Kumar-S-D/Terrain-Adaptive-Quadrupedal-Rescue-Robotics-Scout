"""Configuration dataclasses for the refactored environment.

Centralises all tunable knobs so the gym_env module stays thin and the
training script can import a single place to build env kwargs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple


@dataclass
class RandomTerrainConfig:
    """Knobs for the random packed-pyramid terrain.

    Pyramids are scattered on a jittered grid over a square patch
    centred on the origin. With spacing <= pyramid size the bases
    overlap, giving a tightly packed, rugged obstacle field for
    gait training. Every placement is drawn from a seeded RNG, so
    a given config always produces the same layout.

    extent         side length [m] of the square patch
    spacing        grid pitch [m]; keep it <= size for packing
    size           range of pyramid base side lengths [m]
    height         range of pyramid heights [m]
    density        probability that a grid cell gets a pyramid
    jitter         max random offset of a pyramid centre [m]
    clear_radius   disc around the origin kept flat (spawn area)
    base_z         pyramid base height above the plane [m]
    palette        list of rgba colours; None -> DEFAULT_PALETTE
    """

    seed: int = 7
    extent: float = 3.0
    spacing: float = 0.26
    size: Tuple[float, float] = (0.16, 0.34)
    height: Tuple[float, float] = (0.02, 0.08)
    density: float = 0.95
    jitter: float = 0.05
    clear_radius: float = 0.45
    base_z: float = 0.0
    palette: Optional[Sequence] = None


@dataclass
class PathTaskConfig:
    """Per-episode path & reward configuration."""

    path_length: float = 2.5
    path_width: float = 1.0
    waypoint_count: int = 6
    path_samples: int = 300
    path_tolerance: float = 0.20
    target_speed: float = 0.20
    depth_width: int = 64
    depth_height: int = 48
    # Reward weights
    path_weight: float = 1.0
    velocity_weight: float = 0.5
    stability_weight: float = 0.5
    progress_weight: float = 0.25
    terrain: Optional[RandomTerrainConfig] = None

    def __post_init__(self):
        if self.terrain is None:
            self.terrain = RandomTerrainConfig()
        if self.waypoint_count < 3 or self.path_samples < 2:
            raise ValueError("a path needs at least 3 waypoints and 2 samples")
        if self.path_length <= 0 or self.path_width <= 0:
            raise ValueError("path dimensions must be positive")
        if self.depth_width < 1 or self.depth_height < 1:
            raise ValueError("depth output dimensions must be positive")


@dataclass
class ActionConfig:
    """26-D continuous gait-cycle action space bounds and nominal params."""

    # phase offset for each leg (FR, FL, RR, RL) in [0.0, 1.0)
    phase_offset: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    # step duration / swing fraction for each leg in [0.10, 0.35]
    swing_frac: Tuple[float, float, float, float] = (0.10, 0.10, 0.10, 0.10)
    # total cycle time duration [s] in [0.5, 1.8]
    cycle_time: float = 0.5
    # step height for each leg [m] in [0.04, 0.16]
    step_height: Tuple[float, float, float, float] = (0.04, 0.04, 0.04, 0.04)
    # forward-backward step length for each leg [m] in [-0.06, 0.08]
    length_x: Tuple[float, float, float, float] = (-0.06, -0.06, -0.06, -0.06)
    # sideways step length for each leg [m] in [-0.04, 0.04]
    length_y: Tuple[float, float, float, float] = (-0.04, -0.04, -0.04, -0.04)
    # base height / stance depth for each leg [m] in [-0.23, -0.16]
    stance_z: Tuple[float, float, float, float] = (-0.23, -0.23, -0.23, -0.23)
    # lateral stance width offset [m] in [0.0, 0.04]
    stance_width: float = 0.0


@dataclass
class RobotConfig:
    """URDF / simulation / observation knobs."""

    urdf_path: Optional[str] = None
    sim_time_step: float = 1.0 / 240.0
    action_repeat: int = 4
    max_episode_steps: int = 100
    depth_width: int = 64
    depth_height: int = 48
    depth_capture_interval: int = 5
    use_depth: bool = True
    flatten_obs: bool = False
    render_mode: Optional[str] = None
    render_path_in_gui: bool = True
    action_scale: float = 1.0
    randomize_friction: bool = True
    max_motor_force: float = 33.5
    max_motor_velocity: float = 21.0