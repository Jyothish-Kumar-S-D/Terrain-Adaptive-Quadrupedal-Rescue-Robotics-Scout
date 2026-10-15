"""Gymnasium Environment for Quadruped Path-Following over Dense Pyramid Terrain.

Requirements addressed:
1. Random continuous path initialized per episode (Catmull-Rom spline).
2. Densely packed random pyramids of different sizes/heights on top of base plane.
3. Multi-objective reward:
   - Robot position inside intended path (lateral tolerance / distance)
   - Velocity aligned along path tangent (directional cosine and target speed match)
   - Robot stability from trunk link orientation (roll / pitch stability)
4. Comprehensive observation space:
   - Joint states (positions and velocities for 12 actuated leg joints)
   - Downscaled depth camera frame (compact resolution for RL policies)
   - Trunk orientation (roll, pitch, yaw)
   - Offset from intended path & distance
   - Angle between instantaneous velocity and path tangent
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    p = None
    pybullet_data = None

try:
    import cv2
except ImportError:
    cv2 = None

from surrounding_building.terrain_building import (
    RandomTerrainConfig,
    create_random_terrain,
)
from path_training.task import EpisodePath, PathTaskConfig, _catmull_rom
from Inverse_Kinematics.IK_A1 import (
    LEG_NAMES,
    SIDE,
    CrawlGait,
    CycleCommand,
    LegStep,
    foot_to_joints,
)


# Actuated joints in canonical quadruped order: FR, FL, RR, RL (hip, thigh, calf)
JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

# Foot (lower leg tip) link names for contact detection
FOOT_LINK_NAMES = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
CALF_LINK_NAMES = ["FR_calf", "FL_calf", "RR_calf", "RL_calf"]

# Joint limits from URDF (hip, thigh, calf)
JOINT_LOWER_LIMITS = np.array([
    -0.80285, -1.0472, -2.69653,
    -0.80285, -1.0472, -2.69653,
    -0.80285, -1.0472, -2.69653,
    -0.80285, -1.0472, -2.69653,
], dtype=np.float32)

JOINT_UPPER_LIMITS = np.array([
    0.80285, 4.18879, -0.91630,
    0.80285, 4.18879, -0.91630,
    0.80285, 4.18879, -0.91630,
    0.80285, 4.18879, -0.91630,
], dtype=np.float32)

# Nominal standing joint angles computed from IK (stance_z = -0.22m, cg_offset_x = -0.085m)
NOMINAL_JOINT_ANGLES = np.array([
    0.0, 0.712626, -1.028711,
    0.0, 0.712626, -1.028711,
    0.0, 0.712626, -1.028711,
    0.0, 0.712626, -1.028711,
], dtype=np.float32)

MAX_MOTOR_FORCE = 33.5
MAX_MOTOR_VELOCITY = 21.0

# ----------------------------------------------------------------------
# 18-Dimensional Task-Space Corrective Action Space definition
# (Gangapurwala et al., ICRA 2021, "Real-Time Trajectory Adaptation for Quadrupedal
# Locomotion using Deep Reinforcement Learning"):
# - delta_P_base (3 params: x, y, z base translation) in meters
# - delta_O_base (3 params: roll, pitch, yaw base rotation) in radians
# - delta_P_feet (12 params: x, y, z for FR, FL, RR, RL feet) in meters
# ----------------------------------------------------------------------
CORRECTIVE_ACTION_NAMES = [
    # 0..2: additive base position deviation (x, y, z) in meters
    "delta_base_x", "delta_base_y", "delta_base_z",
    # 3..5: additive base orientation deviation (roll, pitch, yaw) in radians
    "delta_base_roll", "delta_base_pitch", "delta_base_yaw",
    # 6..8: additive foot position deviation FR (x, y, z) in meters
    "delta_foot_FR_x", "delta_foot_FR_y", "delta_foot_FR_z",
    # 9..11: additive foot position deviation FL (x, y, z) in meters
    "delta_foot_FL_x", "delta_foot_FL_y", "delta_foot_FL_z",
    # 12..14: additive foot position deviation RR (x, y, z) in meters
    "delta_foot_RR_x", "delta_foot_RR_y", "delta_foot_RR_z",
    # 15..17: additive foot position deviation RL (x, y, z) in meters
    "delta_foot_RL_x", "delta_foot_RL_y", "delta_foot_RL_z",
]

# Physical maximum deviation scales corresponding to normalized actions in [-1.0, 1.0]
CORRECTIVE_ACTION_SCALES = np.array([
    0.04, 0.03, 0.04,   # delta base position (m)
    0.15, 0.15, 0.15,   # delta base orientation (rad, ~8.6 deg)
    0.05, 0.03, 0.06,   # FR foot position (m)
    0.05, 0.03, 0.06,   # FL foot position (m)
    0.05, 0.03, 0.06,   # RR foot position (m)
    0.05, 0.03, 0.06,   # RL foot position (m)
], dtype=np.float32)

# Hip joint offsets relative to base CoM from URDF (xyz="±0.1805 ±0.047 0")
A1_HIP_OFFSETS: Dict[str, np.ndarray] = {
    "FR": np.array([0.1805, -0.047, 0.0], dtype=np.float32),
    "FL": np.array([0.1805,  0.047, 0.0], dtype=np.float32),
    "RR": np.array([-0.1805, -0.047, 0.0], dtype=np.float32),
    "RL": np.array([-0.1805,  0.047, 0.0], dtype=np.float32),
}


def get_nominal_corrective_action() -> np.ndarray:
    """Return nominal zero-deviation corrective action vector in [-1.0, 1.0]^18."""
    return np.zeros(len(CORRECTIVE_ACTION_NAMES), dtype=np.float32)


# ----------------------------------------------------------------------
# 26-Dimensional Action Space definition for high-level gait cycle generation:
# - phase offset for each leg (4 parameters: FR, FL, RR, RL) in [0.0, 1.0)
# - step duration / swing fraction for each leg (4 parameters: FR, FL, RR, RL) in [0.10, 0.35]
# - total cycle time duration (1 parameter) in [0.5, 1.8] s
# - step height for each leg (4 parameters: FR, FL, RR, RL) in [0.04, 0.16] m
# - forward-backward step length length_x for each leg (4 parameters) in [-0.06, 0.08] m
# - sideways step length length_y for each leg (4 parameters) in [-0.04, 0.04] m
# - base height / stance depth stance_z for each leg (4 parameters) in [-0.23, -0.16] m
# - lateral stance width foot spread (1 parameter) in [0.0, 0.04] m
# ----------------------------------------------------------------------
ACTION_PARAM_NAMES = [
    # 0..3: phase offset for each leg (FR, FL, RR, RL)
    "phase_offset_FR", "phase_offset_FL", "phase_offset_RR", "phase_offset_RL",
    # 4..7: step duration / swing fraction for each leg (FR, FL, RR, RL)
    "swing_frac_FR", "swing_frac_FL", "swing_frac_RR", "swing_frac_RL",
    # 8: total cycle time duration [s]
    "cycle_time",
    # 9..12: step height for each leg (FR, FL, RR, RL) [m]
    "step_height_FR", "step_height_FL", "step_height_RR", "step_height_RL",
    # 13..16: forward-backward step length for each leg (FR, FL, RR, RL) [m]
    "length_x_FR", "length_x_FL", "length_x_RR", "length_x_RL",
    # 17..20: sideways step length for each leg (FR, FL, RR, RL) [m]
    "length_y_FR", "length_y_FL", "length_y_RR", "length_y_RL",
    # 21..24: base height / stance depth for each leg (FR, FL, RR, RL) [m]
    "stance_z_FR", "stance_z_FL", "stance_z_RR", "stance_z_RL",
    # 25: lateral stance width offset [m]
    "stance_width",
]

ACTION_LOW = np.array([
    0.0, 0.0, 0.0, 0.0,           # phase offsets [0.0, 1.0]
    0.10, 0.10, 0.10, 0.10,       # swing fractions [0.10, 0.35]
    0.5,                          # cycle time [0.5, 1.8] s
    0.04, 0.04, 0.04, 0.04,       # step heights [0.04, 0.16] m
    -0.06, -0.06, -0.06, -0.06,   # length_x [-0.06, 0.08] m
    -0.04, -0.04, -0.04, -0.04,   # length_y [-0.04, 0.04] m
    -0.23, -0.23, -0.23, -0.23,   # stance_z [-0.23, -0.16] m
    0.0,                          # stance_width [0.0, 0.04] m
], dtype=np.float32)

ACTION_HIGH = np.array([
    1.0, 1.0, 1.0, 1.0,           # phase offsets [0.0, 1.0]
    0.35, 0.35, 0.35, 0.35,       # swing fractions [0.10, 0.35]
    1.8,                          # cycle time [0.5, 1.8] s
    0.16, 0.16, 0.16, 0.16,       # step heights [0.04, 0.16] m
    0.08, 0.08, 0.08, 0.08,       # length_x [-0.06, 0.08] m
    0.04, 0.04, 0.04, 0.04,       # length_y [-0.04, 0.04] m
    -0.16, -0.16, -0.16, -0.16,   # stance_z [-0.23, -0.16] m
    0.04,                         # stance_width [0.0, 0.04] m
], dtype=np.float32)

NOMINAL_ACTION_PARAMS = np.array([
    0.25, 0.75, 0.50, 0.00,       # phase offsets (canonical crawl: RL 0.0, FL 0.75, RR 0.50, FR 0.25)
    0.20, 0.20, 0.20, 0.20,       # swing fractions
    1.0,                          # cycle time [s]
    0.12, 0.12, 0.12, 0.12,       # step heights [m]
    0.06, 0.06, 0.06, 0.06,       # forward length_x [m]
    0.0, 0.0, 0.0, 0.0,           # lateral length_y [m]
    -0.21, -0.21, -0.21, -0.21,   # base stance_z [m]
    0.02,                         # lateral stance width [m]
], dtype=np.float32)


def normalize_action(params: np.ndarray) -> np.ndarray:
    """Map physical gait cycle parameters to normalized action in [-1.0, 1.0]."""
    p_arr = np.asarray(params, dtype=np.float32)
    norm = 2.0 * (p_arr - ACTION_LOW) / (ACTION_HIGH - ACTION_LOW) - 1.0
    return np.clip(norm, -1.0, 1.0).astype(np.float32)


def denormalize_action(action: np.ndarray) -> np.ndarray:
    """Map normalized action [-1.0, 1.0] to physical gait cycle parameters."""
    a_arr = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    return (ACTION_LOW + 0.5 * (a_arr + 1.0) * (ACTION_HIGH - ACTION_LOW)).astype(np.float32)


def get_nominal_cycle_action() -> np.ndarray:
    """Return nominal normalized action vector in [-1.0, 1.0]."""
    return normalize_action(NOMINAL_ACTION_PARAMS)


def action_to_cycle_command(action: np.ndarray) -> CycleCommand:
    """Decode normalized action [-1.0, 1.0]^26 into an executable CycleCommand."""
    params = denormalize_action(action)
    phase_offset = {
        "FR": float(params[0]),
        "FL": float(params[1]),
        "RR": float(params[2]),
        "RL": float(params[3]),
    }
    swing_frac = {
        "FR": float(params[4]),
        "FL": float(params[5]),
        "RR": float(params[6]),
        "RL": float(params[7]),
    }
    cycle_time = float(params[8])
    step_height = {
        "FR": float(params[9]),
        "FL": float(params[10]),
        "RR": float(params[11]),
        "RL": float(params[12]),
    }
    length_x = {
        "FR": float(params[13]),
        "FL": float(params[14]),
        "RR": float(params[15]),
        "RL": float(params[16]),
    }
    length_y = {
        "FR": float(params[17]),
        "FL": float(params[18]),
        "RR": float(params[19]),
        "RL": float(params[20]),
    }
    stance_z = {
        "FR": float(params[21]),
        "FL": float(params[22]),
        "RR": float(params[23]),
        "RL": float(params[24]),
    }
    stance_width = float(params[25])

    legs = {}
    for n in LEG_NAMES:
        legs[n] = LegStep(
            length_x=length_x[n],
            length_y=length_y[n],
            height=step_height[n],
            stance_z=stance_z[n],
            offset_y=SIDE[n] * stance_width,
            swing_frac=swing_frac[n],
        )

    mean_stance_z = float(np.mean(list(stance_z.values())))
    mean_swing_frac = float(np.mean(list(swing_frac.values())))

    return CycleCommand(
        legs=legs,
        cycle_time=cycle_time,
        stance_z=mean_stance_z,
        swing_frac=mean_swing_frac,
        stance_width=stance_width,
        phase_offset=phase_offset,
    )


class QuadrupedPathTerrainEnv(gym.Env):
    """Gymnasium environment for quadruped robot navigating rugged pyramid terrain

    along a randomized continuous path.
    Each action specifies gait parameters for one full walking cycle.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "direct"],
        "render_fps": 60,
    }

    def __init__(
        self,
        urdf_path: Optional[str] = None,
        render_mode: Optional[str] = None,
        task_config: Optional[PathTaskConfig] = None,
        terrain_config: Optional[RandomTerrainConfig] = None,
        action_scale: float = 1.0,
        action_mode: str = "corrective",
        control_step_dt: float = 0.05,
        depth_width: int = 64,
        depth_height: int = 48,
        use_depth: bool = True,
        flatten_obs: bool = False,
        sim_time_step: float = 1.0 / 240.0,
        action_repeat: int = 4,
        max_episode_steps: int = 1200,
        render_path_in_gui: bool = True,
        depth_capture_interval: int = 5,
        randomize_friction: bool = True,
        dry_run: Optional[bool] = None,
        seed: Optional[int] = None,
    ):
        if render_mode == "human" and dry_run is not True and p is None:
            raise ImportError(
                "render_mode='human' requires PyBullet, but 'pybullet' is not installed in the currently running Python environment.\n"
                "Please run with the conda environment where PyBullet is installed:\n"
                "  conda activate a1_pybullet\n"
                "or run using:\n"
                "  C:\\Users\\mahes\\.conda\\envs\\a1_pybullet\\python.exe tests/test_path_terrain_env.py --policy nominal --max-steps 50"
            )

        self.dry_run = (p is None) if dry_run is None else bool(dry_run)
        self.render_mode = render_mode
        self.render_path_in_gui = render_path_in_gui
        self.action_scale = float(action_scale)
        self.action_mode = str(action_mode).lower()
        if self.action_mode not in ("corrective", "gait_cycle"):
            raise ValueError(f"Unknown action_mode: '{action_mode}'. Supported: 'corrective', 'gait_cycle'.")
        self.control_step_dt = float(control_step_dt)
        self.depth_width = int(depth_width)
        self.depth_height = int(depth_height)
        self.use_depth = bool(use_depth)
        self.flatten_obs = bool(flatten_obs)
        self.sim_time_step = float(sim_time_step)
        self.action_repeat = int(action_repeat)
        self.max_episode_steps = int(max_episode_steps)
        self.depth_capture_interval = max(1, int(depth_capture_interval))
        self.randomize_friction = bool(randomize_friction)

        # Locate URDF
        if urdf_path is None:
            # Default to repo root urdf/a1_unitree.urdf
            repo_root = Path(__file__).resolve().parent.parent
            default_urdf = repo_root / "urdf" / "a1_unitree.urdf"
            self.urdf_path = str(default_urdf)
        else:
            self.urdf_path = str(Path(urdf_path).resolve())

        # Configs
        self.task_config = task_config or PathTaskConfig(
            depth_width=self.depth_width,
            depth_height=self.depth_height,
        )
        if terrain_config is not None:
            self.task_config.terrain = terrain_config
        elif self.task_config.terrain is None:
            self.task_config.terrain = RandomTerrainConfig()

        self.terrain_config = self.task_config.terrain

        # Action space configuration
        if self.action_mode == "corrective":
            # 18D continuous task-space residual deviations [-1.0, 1.0] (Gangapurwala et al., ICRA 2021)
            self.action_param_names = list(CORRECTIVE_ACTION_NAMES)
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(len(CORRECTIVE_ACTION_NAMES),),
                dtype=np.float32,
            )
        else:
            # 26D continuous gait cycle parameter space [-1.0, 1.0]
            self.action_param_names = list(ACTION_PARAM_NAMES)
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(len(ACTION_PARAM_NAMES),),
                dtype=np.float32,
            )

        # Observation spaces (incorporating Paper Section IV-B and Table I features)
        dict_obs_spaces = {
            "joint_positions": spaces.Box(
                low=JOINT_LOWER_LIMITS - 0.10,
                high=JOINT_UPPER_LIMITS + 0.10,
                dtype=np.float32,
            ),
            "joint_velocities": spaces.Box(
                low=-MAX_MOTOR_VELOCITY * 2.0,
                high=MAX_MOTOR_VELOCITY * 2.0,
                shape=(12,),
                dtype=np.float32,
            ),
            "trunk_orientation": spaces.Box(
                low=-np.pi - 0.05,
                high=np.pi + 0.05,
                shape=(3,),
                dtype=np.float32,
            ),  # [roll, pitch, yaw]
            "base_linear_velocity": spaces.Box(
                low=-20.0,
                high=20.0,
                shape=(3,),
                dtype=np.float32,
            ),  # trunk linear velocity [m/s]
            "base_angular_velocity": spaces.Box(
                low=-20.0,
                high=20.0,
                shape=(3,),
                dtype=np.float32,
            ),  # trunk angular velocity [rad/s]
            "foot_contacts": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(4,),
                dtype=np.float32,
            ),  # binary contact flags: FR, FL, RR, RL
            "foot_velocities": spaces.Box(
                low=0.0,
                high=20.0,
                shape=(4,),
                dtype=np.float32,
            ),  # linear velocity magnitudes of the 4 feet (for slip penalty)
            "previous_action": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=self.action_space.shape,
                dtype=np.float32,
            ),  # last applied action for temporal continuity
            "path_offset": spaces.Box(
                low=-20.0,
                high=20.0,
                shape=(1,),
                dtype=np.float32,
            ),  # signed lateral error
            "path_distance": spaces.Box(
                low=0.0,
                high=20.0,
                shape=(1,),
                dtype=np.float32,
            ),  # distance to nearest path point
            "velocity_tangent_angle": spaces.Box(
                low=-np.pi - 0.05,
                high=np.pi + 0.05,
                shape=(1,),
                dtype=np.float32,
            ),  # angle between velocity & path tangent
            "short_horizon_goal": spaces.Box(
                low=-20.0,
                high=20.0,
                shape=(3,),
                dtype=np.float32,
            ),  # relative displacement vector to lookahead waypoint in base frame
            "height_map": spaces.Box(
                low=-2.0,
                high=2.0,
                shape=(36,),
                dtype=np.float32,
            ),  # 6x6 local terrain elevation grid centered at base (Paper IV-B.2)
            "foot_terrain_clearances": spaces.Box(
                low=-2.0,
                high=2.0,
                shape=(4,),
                dtype=np.float32,
            ),  # terrain clearance below each foot (Paper Table I)
        }

        if self.use_depth:
            # Downscaled depth frame: metric depth buffer in meters (0 to 10m)
            dict_obs_spaces["depth"] = spaces.Box(
                low=0.0,
                high=10.0,
                shape=(self.depth_height, self.depth_width),
                dtype=np.float32,
            )

        self._dict_observation_space = spaces.Dict(dict_obs_spaces)

        if self.flatten_obs:
            flat_dim = sum(int(np.prod(space.shape)) for space in dict_obs_spaces.values())
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(flat_dim,),
                dtype=np.float32,
            )
        else:
            self.observation_space = self._dict_observation_space

        # Internal simulation state
        self.physics_client_id: Optional[int] = None
        self.robot_id: Optional[int] = None
        self.plane_id: Optional[int] = None
        self.terrain_ids: List[int] = []
        self.debug_line_ids: List[int] = []

        self.joint_indices: List[int] = []
        self.foot_link_indices: List[int] = []  # FR, FL, RR, RL foot links
        self.calf_link_indices: List[int] = []
        self.trunk_link_index: int = -1
        self.camera_link_index: int = -1

        self.episode_path: Optional[EpisodePath] = None
        self.episode_step_count: int = 0
        self.episode_index: int = 0
        self._last_progress_index: int = 0

        # Gait generator instance
        self.gait: Optional[CrawlGait] = None

        # Depth camera projection matrix
        self._camera_proj_matrix = None

        # Temporal context: previous action, velocities, and cached depth frame
        self._previous_action = (
            get_nominal_corrective_action()
            if self.action_mode == "corrective"
            else get_nominal_cycle_action()
        )
        self._previous_joint_positions = NOMINAL_JOINT_ANGLES.copy()
        self._previous_joint_velocities = np.zeros(12, dtype=np.float32)
        self._previous_foot_velocities = np.zeros(4, dtype=np.float32)
        self._cached_depth_frame: Optional[np.ndarray] = None
        self._depth_step_counter: int = 0

        if seed is not None:
            self.reset(seed=seed)

    def _ensure_physics_client(self):
        """Connect to PyBullet if not already connected."""
        if self.dry_run:
            return
        if p is None:
            raise ImportError(
                "pybullet is required to run the simulation. "
                "Please install pybullet (`pip install pybullet`)."
            )
        if self.physics_client_id is not None:
            return
        try:
            if p.getConnectionInfo().get("isConnected", 0) != 0:
                self.physics_client_id = 0
                return
        except Exception:
            pass

        if self.render_mode == "human":
            self.physics_client_id = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        else:
            self.physics_client_id = p.connect(p.DIRECT)

        if pybullet_data is not None:
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.sim_time_step)

    def _init_depth_camera_matrix(self):
        """Pre-compute camera projection matrix for downscaled depth frame."""
        if p is None:
            return
        fov = 90.0
        aspect = float(self.depth_width) / float(self.depth_height)
        near = 0.05
        far = 10.0
        self._camera_near = near
        self._camera_far = far
        self._camera_proj_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=aspect,
            nearVal=near,
            farVal=far,
        )

    def _setup_robot_links_and_joints(self):
        """Map joints and links from the loaded URDF."""
        if p is None or self.robot_id is None:
            return

        num_joints = p.getNumJoints(self.robot_id)
        name_to_idx = {}
        link_name_to_idx = {}
        for j in range(num_joints):
            info = p.getJointInfo(self.robot_id, j)
            j_name = info[1].decode("utf-8")
            l_name = info[12].decode("utf-8")
            name_to_idx[j_name] = j
            link_name_to_idx[l_name] = j
            if l_name == "trunk":
                self.trunk_link_index = j
            if j_name == "cam_L_joint" or l_name == "cam_L":
                self.camera_link_index = j

        # Order joints strictly as JOINT_NAMES
        self.joint_indices = []
        for name in JOINT_NAMES:
            if name in name_to_idx:
                self.joint_indices.append(name_to_idx[name])
            else:
                raise ValueError(f"Required joint '{name}' not found in URDF.")

        # Map foot links for contact detection (FR, FL, RR, RL)
        self.foot_link_indices = []
        self.calf_link_indices = []
        for foot_name in FOOT_LINK_NAMES:
            if foot_name in link_name_to_idx:
                self.foot_link_indices.append(link_name_to_idx[foot_name])
        for calf_name in CALF_LINK_NAMES:
            if calf_name in link_name_to_idx:
                self.calf_link_indices.append(link_name_to_idx[calf_name])

        # Set foot and calf contact dynamics
        all_foot_links = set(self.foot_link_indices + self.calf_link_indices)
        for idx in all_foot_links:
            p.changeDynamics(
                self.robot_id, idx,
                lateralFriction=1.2,
                spinningFriction=0.01,
                rollingFriction=0.01,
            )

        # Set default motor control
        for j in range(num_joints):
            p.setJointMotorControl2(
                self.robot_id, j, p.VELOCITY_CONTROL, force=0
            )

    def _generate_continuous_path(self, rng: np.random.Generator) -> EpisodePath:
        """Requirement 1: A random continuous path initialized for each episode."""
        cfg = self.task_config
        xs = np.linspace(0.0, cfg.path_length, cfg.waypoint_count)
        ys = rng.uniform(-cfg.path_width / 2.0, cfg.path_width / 2.0, cfg.waypoint_count)
        # Fix spawn at origin and forward heading
        ys[0] = 0.0
        ys[1] = ys[0] + rng.uniform(-0.05, 0.05)
        ys[-1] = float(rng.uniform(-cfg.path_width / 3.0, cfg.path_width / 3.0))

        control = np.column_stack((xs, ys))
        return _catmull_rom(control, cfg.path_samples)

    def _load_dense_terrain(self, seed: int):
        """Requirement 2: Random pyramids of different sizes, densely packed on top of plane."""
        if self.dry_run or p is None:
            from surrounding_building.terrain_building import terrain_layout
            cfg = self.terrain_config
            terrain_cfg = RandomTerrainConfig(
                seed=seed,
                extent=cfg.extent,
                spacing=cfg.spacing,
                size=cfg.size,
                height=cfg.height,
                density=cfg.density,
                jitter=cfg.jitter,
                clear_radius=cfg.clear_radius,
                base_z=cfg.base_z,
                palette=cfg.palette,
            )
            layout = list(terrain_layout(terrain_cfg))
            self.terrain_ids = list(range(len(layout)))
            return

        # Clean up existing terrain pyramids
        for pid in self.terrain_ids:
            try:
                p.removeBody(pid)
            except Exception:
                pass
        self.terrain_ids = []

        cfg = self.terrain_config
        terrain_cfg = RandomTerrainConfig(
            seed=seed,
            extent=cfg.extent,
            spacing=cfg.spacing,
            size=cfg.size,
            height=cfg.height,
            density=cfg.density,
            jitter=cfg.jitter,
            clear_radius=cfg.clear_radius,
            base_z=cfg.base_z,
            palette=cfg.palette,
        )
        self.terrain_ids = create_random_terrain(terrain_cfg)

    def _draw_path_debug_lines(self):
        """Visualize continuous path in PyBullet GUI."""
        if self.dry_run or p is None or self.render_mode != "human" or not self.render_path_in_gui:
            return

        # Remove previous lines
        for lid in self.debug_line_ids:
            try:
                p.removeUserDebugItem(lid)
            except Exception:
                pass
        self.debug_line_ids = []

        if self.episode_path is None:
            return

        pts = self.episode_path.points
        for i in range(len(pts) - 1):
            p1 = [float(pts[i][0]), float(pts[i][1]), 0.03]
            p2 = [float(pts[i + 1][0]), float(pts[i + 1][1]), 0.03]
            lid = p.addUserDebugLine(p1, p2, [0.1, 0.9, 0.2], lineWidth=2.5)
            self.debug_line_ids.append(lid)

    def _reset_robot_pose(self):
        """Spawn and stabilize robot at start of path in nominal standing posture."""
        if self.dry_run:
            self._sim_pos = np.array([0.0, 0.0, 0.28], dtype=np.float32)
            self._sim_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            init_yaw = 0.0
            if self.episode_path is not None and len(self.episode_path.tangents) > 0:
                init_tangent = self.episode_path.tangents[0]
                init_yaw = float(math.atan2(init_tangent[1], init_tangent[0]))
            self._sim_rpy = np.array([0.0, 0.0, init_yaw], dtype=np.float32)
            self._sim_joints = NOMINAL_JOINT_ANGLES.copy()
            self._sim_joint_vels = np.zeros(12, dtype=np.float32)
            return

        if p is None or self.robot_id is None:
            return

        # Initial heading along the start of the path
        if self.episode_path is not None and len(self.episode_path.tangents) > 0:
            init_tangent = self.episode_path.tangents[0]
            init_yaw = math.atan2(init_tangent[1], init_tangent[0])
        else:
            init_yaw = 0.0

        init_quat = p.getQuaternionFromEuler([0.0, 0.0, init_yaw])
        init_pos = [0.0, 0.0, 0.28]  # nominal stand height

        p.resetBasePositionAndOrientation(self.robot_id, init_pos, init_quat)
        p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])

        # Reset joint positions to nominal standing posture
        for idx, target_q in zip(self.joint_indices, NOMINAL_JOINT_ANGLES):
            p.resetJointState(self.robot_id, idx, targetValue=float(target_q), targetVelocity=0.0)
            p.setJointMotorControl2(
                self.robot_id,
                idx,
                p.POSITION_CONTROL,
                targetPosition=float(target_q),
                force=MAX_MOTOR_FORCE,
            )

        # Allow physics to settle slightly
        for _ in range(15):
            p.stepSimulation()

    def _capture_downscaled_depth(self) -> np.ndarray:
        """Requirement 4: Capture and downscale depth frame."""
        if p is None or self.robot_id is None or self.camera_link_index < 0:
            return np.zeros((self.depth_height, self.depth_width), dtype=np.float32)

        link_state = p.getLinkState(
            self.robot_id, self.camera_link_index, computeForwardKinematics=True
        )
        camera_pos = np.array(link_state[4])
        camera_orn = link_state[5]

        rotation = np.array(p.getMatrixFromQuaternion(camera_orn)).reshape(3, 3)
        forward = rotation @ np.array([1.0, 0.0, 0.0])
        up = rotation @ np.array([0.0, 0.0, 1.0])
        target = camera_pos + forward

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target,
            cameraUpVector=up,
        )

        image = p.getCameraImage(
            width=self.depth_width,
            height=self.depth_height,
            viewMatrix=view_matrix,
            projectionMatrix=self._camera_proj_matrix,
            renderer=p.ER_TINY_RENDERER,
        )

        depth_buf = np.asarray(image[3], dtype=np.float32).reshape(
            self.depth_height, self.depth_width
        )

        # Convert buffer to metric depth
        metric_depth = (
            self._camera_near * self._camera_far
            / (self._camera_far - depth_buf * (self._camera_far - self._camera_near) + 1e-6)
        )
        return np.clip(metric_depth, 0.0, self._camera_far).astype(np.float32)

    def _get_trunk_orientation(self) -> Tuple[float, float, float]:
        """Get orientation (roll, pitch, yaw) of the trunk link."""
        if p is None or self.robot_id is None:
            return (0.0, 0.0, 0.0)

        if self.trunk_link_index >= 0:
            link_state = p.getLinkState(self.robot_id, self.trunk_link_index)
            quat = link_state[5]
        else:
            _, quat = p.getBasePositionAndOrientation(self.robot_id)

        roll, pitch, yaw = p.getEulerFromQuaternion(quat)
        return float(roll), float(pitch), float(yaw)

    def _get_foot_contacts(self) -> np.ndarray:
        """Detect ground contact for each foot (FR, FL, RR, RL)."""
        contacts = np.zeros(4, dtype=np.float32)
        if p is None or self.robot_id is None or not self.foot_link_indices:
            return contacts
        for i, foot_idx in enumerate(self.foot_link_indices):
            pts = p.getContactPoints(bodyA=self.robot_id, linkIndexA=foot_idx)
            if not pts and i < len(self.calf_link_indices):
                pts = p.getContactPoints(bodyA=self.robot_id, linkIndexA=self.calf_link_indices[i])
            if pts:
                contacts[i] = 1.0
        return contacts

    def _get_base_angular_velocity(self) -> np.ndarray:
        """Get base angular velocity [wx, wy, wz] in rad/s (Paper Section IV-A/IV-B)."""
        if self.dry_run or p is None or self.robot_id is None:
            return np.zeros(3, dtype=np.float32)
        try:
            _, ang_vel = p.getBaseVelocity(self.robot_id)
            return np.array(ang_vel, dtype=np.float32)
        except Exception:
            return np.zeros(3, dtype=np.float32)

    def _get_foot_velocities(self) -> np.ndarray:
        """Get foot linear velocity magnitudes for FR, FL, RR, RL (Paper Table I Foot Slip)."""
        if self.dry_run or p is None or self.robot_id is None:
            return np.zeros(4, dtype=np.float32)
        vels = np.zeros(4, dtype=np.float32)
        for i, foot_idx in enumerate(self.foot_link_indices):
            try:
                state = p.getLinkState(self.robot_id, foot_idx, computeLinkVelocity=1)
                lin_vel = state[6]
                vels[i] = float(np.linalg.norm(lin_vel))
            except Exception:
                vels[i] = 0.0
        return vels

    def _get_local_height_map(self, base_pos: np.ndarray, yaw: float) -> np.ndarray:
        """Sample 6x6 local terrain elevation grid centered at robot base (Paper Section IV-B.2).
        
        The grid covers an area of 1.0 x 1.0 m aligned with the robot's heading and lateral axes.
        Elevations are relative to base z, clipped between [-2.0, 2.0] m.
        """
        if self.dry_run or p is None:
            return np.zeros(36, dtype=np.float32)

        coords = np.linspace(-0.5, 0.5, 6)
        gx, gy = np.meshgrid(coords, coords)
        local_x = gx.ravel()
        local_y = gy.ravel()

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        world_x = base_pos[0] + cos_yaw * local_x - sin_yaw * local_y
        world_y = base_pos[1] + sin_yaw * local_x + cos_yaw * local_y
        z_base = float(base_pos[2])

        ray_froms = [[float(wx), float(wy), z_base + 0.8] for wx, wy in zip(world_x, world_y)]
        ray_tos = [[float(wx), float(wy), z_base - 0.8] for wx, wy in zip(world_x, world_y)]

        try:
            results = p.rayTestBatch(ray_froms, ray_tos)
            elevations = np.zeros(36, dtype=np.float32)
            for i, res in enumerate(results):
                if res[0] >= 0:
                    elevations[i] = float(res[3][2] - z_base)
                else:
                    elevations[i] = -z_base
            return np.clip(elevations, -2.0, 2.0).astype(np.float32)
        except Exception:
            return np.zeros(36, dtype=np.float32)

    def _get_foot_terrain_clearances(self) -> np.ndarray:
        """Measure vertical clearance between each foot and terrain ground surface (Paper Table I Foot Clearance)."""
        if self.dry_run or p is None or self.robot_id is None or not self.foot_link_indices:
            return np.zeros(4, dtype=np.float32)

        clearances = np.zeros(4, dtype=np.float32)
        ray_froms = []
        ray_tos = []
        foot_zs = []

        for foot_idx in self.foot_link_indices:
            try:
                state = p.getLinkState(self.robot_id, foot_idx)
                f_pos = state[0]
                foot_zs.append(float(f_pos[2]))
                ray_froms.append([float(f_pos[0]), float(f_pos[1]), float(f_pos[2]) + 0.3])
                ray_tos.append([float(f_pos[0]), float(f_pos[1]), float(f_pos[2]) - 0.5])
            except Exception:
                foot_zs.append(0.0)
                ray_froms.append([0.0, 0.0, 0.3])
                ray_tos.append([0.0, 0.0, -0.5])

        try:
            results = p.rayTestBatch(ray_froms, ray_tos)
            for i, res in enumerate(results):
                if res[0] >= 0:
                    clearances[i] = float(foot_zs[i] - res[3][2])
                else:
                    clearances[i] = float(foot_zs[i])
            return np.clip(clearances, -1.0, 1.0).astype(np.float32)
        except Exception:
            return np.zeros(4, dtype=np.float32)

    def _get_short_horizon_goal(
        self, base_pos: np.ndarray, yaw: float, horizon_dist: float = 0.5
    ) -> np.ndarray:
        """Displacement vector to lookahead path waypoint in base frame (Paper Section IV-B.1 C^{traj}_{t+H})."""
        if self.episode_path is None or len(self.episode_path.points) == 0:
            return np.zeros(3, dtype=np.float32)

        nearest_idx, _, _, _ = self.episode_path.nearest(base_pos[:2])
        pts = self.episode_path.points
        total_len = len(pts)
        step_ahead = max(1, int(round(horizon_dist / max(self.task_config.path_length / max(total_len, 1), 1e-4))))
        lookahead_idx = min(nearest_idx + step_ahead, total_len - 1)

        goal_world = np.array([pts[lookahead_idx][0], pts[lookahead_idx][1], base_pos[2]], dtype=np.float32)
        delta_world = goal_world - base_pos

        cos_yaw = math.cos(-yaw)
        sin_yaw = math.sin(-yaw)
        dx_base = cos_yaw * delta_world[0] - sin_yaw * delta_world[1]
        dy_base = sin_yaw * delta_world[0] + cos_yaw * delta_world[1]
        dz_base = delta_world[2]

        return np.array([dx_base, dy_base, dz_base], dtype=np.float32)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Compute full observation space (Paper Section IV-B & Table I)."""
        if self.dry_run:
            joint_positions = self._sim_joints.copy()
            joint_velocities = self._sim_joint_vels.copy()
            pos = self._sim_pos
            linear_velocity = self._sim_vel
            base_ang_vel = np.zeros(3, dtype=np.float32)
            trunk_roll, trunk_pitch, trunk_yaw = float(self._sim_rpy[0]), float(self._sim_rpy[1]), float(self._sim_rpy[2])
            nearest_idx, distance, lateral_offset, tangent = self.episode_path.nearest(pos[:2])

            vel_xy = linear_velocity[:2].astype(np.float64)
            speed = float(np.linalg.norm(vel_xy))
            if speed > 1e-5:
                dot = float(np.clip(np.dot(vel_xy / speed, tangent), -1.0, 1.0))
                cross = float(tangent[0] * vel_xy[1] - tangent[1] * vel_xy[0])
                velocity_angle = float(math.atan2(cross, dot))
            else:
                velocity_angle = 0.0

            foot_contacts = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            foot_velocities = np.zeros(4, dtype=np.float32)
            foot_clearances = np.zeros(4, dtype=np.float32)
            short_horizon_goal = self._get_short_horizon_goal(pos, trunk_yaw)
            height_map = np.zeros(36, dtype=np.float32)
            depth = np.zeros((self.depth_height, self.depth_width), dtype=np.float32) if self.use_depth else None
        elif p is None or self.robot_id is None:
            joint_positions = NOMINAL_JOINT_ANGLES.copy()
            joint_velocities = np.zeros(12, dtype=np.float32)
            trunk_roll, trunk_pitch, trunk_yaw = 0.0, 0.0, 0.0
            lateral_offset = 0.0
            distance = 0.0
            velocity_angle = 0.0
            foot_contacts = np.zeros(4, dtype=np.float32)
            foot_velocities = np.zeros(4, dtype=np.float32)
            foot_clearances = np.zeros(4, dtype=np.float32)
            short_horizon_goal = np.zeros(3, dtype=np.float32)
            height_map = np.zeros(36, dtype=np.float32)
            depth = np.zeros((self.depth_height, self.depth_width), dtype=np.float32) if self.use_depth else None
            nearest_idx = 0
            linear_velocity = np.zeros(3, dtype=np.float32)
            base_ang_vel = np.zeros(3, dtype=np.float32)
        else:
            joint_states = p.getJointStates(self.robot_id, self.joint_indices)
            joint_positions = np.array([s[0] for s in joint_states], dtype=np.float32)
            joint_velocities = np.array([s[1] for s in joint_states], dtype=np.float32)

            pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
            linear_velocity = np.array(lin_vel, dtype=np.float32)
            base_ang_vel = np.array(ang_vel, dtype=np.float32)

            trunk_roll, trunk_pitch, trunk_yaw = self._get_trunk_orientation()
            foot_contacts = self._get_foot_contacts()
            foot_velocities = self._get_foot_velocities()
            foot_clearances = self._get_foot_terrain_clearances()
            height_map = self._get_local_height_map(np.array(pos, dtype=np.float32), trunk_yaw)
            short_horizon_goal = self._get_short_horizon_goal(np.array(pos, dtype=np.float32), trunk_yaw)

            # Path metrics
            nearest_idx, distance, lateral_offset, tangent = self.episode_path.nearest(pos[:2])

            vel_xy = linear_velocity[:2].astype(np.float64)
            speed = float(np.linalg.norm(vel_xy))
            if speed > 1e-5:
                dot = float(np.clip(np.dot(vel_xy / speed, tangent), -1.0, 1.0))
                cross = float(tangent[0] * vel_xy[1] - tangent[1] * vel_xy[0])
                velocity_angle = float(math.atan2(cross, dot))
            else:
                velocity_angle = 0.0

            # Depth: only re-render every N steps, reuse cached frame otherwise
            if self.use_depth:
                self._depth_step_counter += 1
                if self._cached_depth_frame is None or self._depth_step_counter >= self.depth_capture_interval:
                    self._cached_depth_frame = self._capture_downscaled_depth()
                    self._depth_step_counter = 0
                depth = self._cached_depth_frame.copy()
            else:
                depth = None

        obs_dict = {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "trunk_orientation": np.array([trunk_roll, trunk_pitch, trunk_yaw], dtype=np.float32),
            "base_linear_velocity": np.array(linear_velocity, dtype=np.float32).copy(),
            "base_angular_velocity": np.array(base_ang_vel, dtype=np.float32).copy(),
            "foot_contacts": foot_contacts,
            "foot_velocities": foot_velocities,
            "previous_action": self._previous_action.copy(),
            "path_offset": np.array([lateral_offset], dtype=np.float32),
            "path_distance": np.array([distance], dtype=np.float32),
            "velocity_tangent_angle": np.array([velocity_angle], dtype=np.float32),
            "short_horizon_goal": short_horizon_goal,
            "height_map": height_map,
            "foot_terrain_clearances": foot_clearances,
        }
        if self.use_depth:
            obs_dict["depth"] = depth

        self._current_info_metrics = {
            "path_index": nearest_idx,
            "path_distance": distance,
            "path_offset": lateral_offset,
            "velocity_tangent_angle": velocity_angle,
            "trunk_roll": trunk_roll,
            "trunk_pitch": trunk_pitch,
            "trunk_yaw": trunk_yaw,
            "base_linear_velocity": np.array(linear_velocity, dtype=np.float32).copy(),
            "base_angular_velocity": np.array(base_ang_vel, dtype=np.float32).copy(),
            "foot_clearances": foot_clearances.copy(),
            "foot_velocities": foot_velocities.copy(),
        }
        return obs_dict

    def _format_obs(self, obs_dict: Dict[str, np.ndarray]) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """Convert obs dict to flat array if flatten_obs is True in canonical dict order."""
        if not self.flatten_obs:
            return obs_dict

        flat_parts = []
        for key in self._dict_observation_space.spaces.keys():
            if key in obs_dict and obs_dict[key] is not None:
                flat_parts.append(obs_dict[key].flatten())
        return np.concatenate(flat_parts).astype(np.float32)

    def get_nominal_action(self) -> np.ndarray:
        """Return the 26-dim normalized nominal gait cycle action."""
        return get_nominal_cycle_action()

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Map normalized action [-1, 1]^26 to physical gait cycle parameters."""
        return denormalize_action(action)

    def normalize_action(self, params: np.ndarray) -> np.ndarray:
        """Map physical gait cycle parameters to normalized action [-1, 1]^26."""
        return normalize_action(params)

    def action_to_cycle_command(self, action: np.ndarray) -> CycleCommand:
        """Decode normalized action [-1, 1]^26 into an executable CycleCommand."""
        return action_to_cycle_command(action)

    def _compute_cycle_reward(
        self,
        action: np.ndarray,
        obs_dict: Dict[str, np.ndarray],
        cycle_progress: float,
        terminated: bool,
        dt_cycle: float,
    ) -> Tuple[float, Dict[str, float]]:
        """Multi-objective reward for an executed gait cycle.

        Components:
        1. Forward velocity along path tangent (integrated over cycle duration)
        2. Path corridor tracking (lateral distance penalty)
        3. Trunk stability (orientation penalty)
        4. Cycle forward progress (waypoint advancement bonus)
        5. Action smoothness (penalize changes in gait parameters between cycles)
        6. Energy penalty (discourage unnecessarily extreme parameters)
        7. Stall penalty (explicit penalty for stationary robot)
        8. Fall penalty (-10.0 if robot fell during the cycle)
        """
        cfg = self.task_config

        lin_vel = self._current_info_metrics["base_linear_velocity"][:2]
        speed = float(np.linalg.norm(lin_vel))
        angle = float(obs_dict["velocity_tangent_angle"][0])
        target_speed = max(cfg.target_speed, 1e-4)

        forward_speed = speed * math.cos(angle) if speed > 1e-5 else 0.0
        reward_forward = max(0.0, forward_speed / target_speed) * math.exp(
            -0.5 * ((forward_speed - target_speed) / target_speed) ** 2
        )

        dist = float(obs_dict["path_distance"][0])
        path_tol = max(cfg.path_tolerance, 1e-4)
        reward_path = math.exp(-0.5 * (dist / path_tol) ** 2)

        roll = float(obs_dict["trunk_orientation"][0])
        pitch = float(obs_dict["trunk_orientation"][1])
        reward_stability = math.exp(-0.5 * ((roll / 0.45) ** 2 + (pitch / 0.45) ** 2))

        action_delta = action - self._previous_action
        action_rate_penalty = float(np.mean(np.square(action_delta)))
        energy_penalty = float(np.mean(np.square(action)))

        stall_penalty = -0.3 if forward_speed < 0.01 else 0.0
        fall_penalty = -10.0 if terminated else 0.0

        rate_reward = (
            1.5 * reward_forward
            + cfg.path_weight * reward_path
            + cfg.stability_weight * reward_stability
            + stall_penalty
        )

        total_reward = (
            rate_reward * dt_cycle
            + 4.0 * cycle_progress
            - 0.05 * action_rate_penalty
            - 0.02 * energy_penalty
            + fall_penalty
        )

        components = {
            "reward_forward": float(reward_forward),
            "reward_velocity": float(reward_forward),
            "reward_path": float(reward_path),
            "reward_stability": float(reward_stability),
            "reward_progress": float(cycle_progress),
            "action_rate_penalty": float(action_rate_penalty),
            "energy_penalty": float(energy_penalty),
            "stall_penalty": float(stall_penalty),
            "fall_penalty": float(fall_penalty),
            "total_reward": float(total_reward),
        }
        return float(total_reward), components

    def _compute_reward(
        self,
        action: np.ndarray,
        obs_dict: Dict[str, np.ndarray],
        terminated: bool,
    ) -> Tuple[float, Dict[str, float]]:
        """Legacy substep reward computation."""
        return self._compute_cycle_reward(
            action=action,
            obs_dict=obs_dict,
            cycle_progress=0.0,
            terminated=terminated,
            dt_cycle=self.sim_time_step * self.action_repeat,
        )

    def _compute_paper_reward(
        self,
        action: np.ndarray,
        obs_dict: Dict[str, np.ndarray],
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        foot_velocities: np.ndarray,
        progress: float,
        terminated: bool,
        dt_step: float,
    ) -> Tuple[float, Dict[str, float]]:
        """Multi-objective trajectory tracking and stability reward (Gangapurwala et al., ICRA 2021, Table I)."""
        cfg = self.task_config

        # 1. Trajectory Tracking error ||P_base^m - P_base^traj||^2
        dist_to_path = float(obs_dict["path_distance"][0])
        path_tol = max(cfg.path_tolerance, 1e-4)
        reward_tracking = math.exp(-0.5 * (dist_to_path / path_tol) ** 2)

        # 2. Short Horizon Goal progress: lookahead progress
        reward_goal = float(progress) * 4.0

        # 3. Base Orientation stability: ||O_base^{x,y,z} - [0,0,O_base^z]||^2 = roll^2 + pitch^2
        roll = float(obs_dict["trunk_orientation"][0])
        pitch = float(obs_dict["trunk_orientation"][1])
        orientation_error = roll * roll + pitch * pitch
        reward_orientation = math.exp(-0.5 * orientation_error / (0.25 ** 2))

        # 4. Foot Slip penalty: ||v_foot_world||^2 for feet with contact > 0
        contacts = obs_dict["foot_contacts"]
        foot_slip_penalty = float(np.sum(contacts * np.square(foot_velocities)))

        # 5. Foot Clearance: (fh - f_des)^2 * ||v_foot||^2 during swing (contacts == 0)
        clearances = obs_dict["foot_terrain_clearances"]
        des_clearance = 0.08
        swing_mask = 1.0 - contacts
        foot_clearance_penalty = float(
            np.sum(swing_mask * np.square(clearances - des_clearance) * np.square(foot_velocities))
        )

        # 6. Smoothness: ||J_t - J_{t-1}||^2
        joint_delta = joint_positions - self._previous_joint_positions
        smoothness_penalty = float(np.sum(np.square(joint_delta)))

        # 7. Joint Velocity penalty: ||J_dot_t||^2
        joint_vel_penalty = float(np.sum(np.square(joint_velocities)))

        # 8. Joint Acceleration penalty: ||J_ddot_t||^2
        joint_acc = (joint_velocities - self._previous_joint_velocities) / max(dt_step, 1e-4)
        joint_acc_penalty = float(np.mean(np.square(joint_acc)))

        # 9. Action rate penalty & energy
        action_delta = action - self._previous_action
        action_rate_penalty = float(np.mean(np.square(action_delta)))
        energy_penalty = float(np.mean(np.square(action)))

        # 10. Stability margin bonus / survival
        stability_bonus = 0.5 if (not terminated and abs(roll) < 0.35 and abs(pitch) < 0.35) else 0.0
        fall_penalty = -10.0 if terminated else 0.0

        step_reward = (
            2.0 * reward_tracking
            + reward_goal
            + 1.5 * reward_orientation
            + stability_bonus
            - 0.5 * foot_slip_penalty
            - 0.3 * foot_clearance_penalty
            - 0.1 * smoothness_penalty
            - 0.01 * joint_vel_penalty
            - 0.001 * joint_acc_penalty
            - 0.05 * action_rate_penalty
            - 0.02 * energy_penalty
            + fall_penalty
        )

        components = {
            "reward_tracking": float(reward_tracking),
            "reward_goal": float(reward_goal),
            "reward_orientation": float(reward_orientation),
            "stability_bonus": float(stability_bonus),
            "foot_slip_penalty": float(foot_slip_penalty),
            "foot_clearance_penalty": float(foot_clearance_penalty),
            "smoothness_penalty": float(smoothness_penalty),
            "joint_vel_penalty": float(joint_vel_penalty),
            "joint_acc_penalty": float(joint_acc_penalty),
            "action_rate_penalty": float(action_rate_penalty),
            "energy_penalty": float(energy_penalty),
            "fall_penalty": float(fall_penalty),
            "total_reward": float(step_reward),
        }
        return float(step_reward), components

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Union[Dict[str, np.ndarray], np.ndarray], Dict[str, Any]]:
        """Reset environment for a new episode."""
        super().reset(seed=seed)

        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)
            rng = np.random.default_rng(seed)
        else:
            rng = self.np_random
        self.episode_step_count = 0
        self._last_progress_index = 0
        self._previous_action = (
            get_nominal_corrective_action()
            if self.action_mode == "corrective"
            else get_nominal_cycle_action()
        )
        self._previous_joint_positions = NOMINAL_JOINT_ANGLES.copy()
        self._previous_joint_velocities = np.zeros(12, dtype=np.float32)
        self._previous_foot_velocities = np.zeros(4, dtype=np.float32)
        self._cached_depth_frame = None
        self._depth_step_counter = 0

        # Ensure PyBullet simulation is initialized
        self._ensure_physics_client()
        self._init_depth_camera_matrix()

        # Load plane if not loaded
        if self.plane_id is None and p is not None:
            self.plane_id = p.loadURDF("plane.urdf")

        # Load robot if not loaded
        if self.robot_id is None and p is not None:
            if not os.path.exists(self.urdf_path):
                raise FileNotFoundError(f"Robot URDF not found at: {self.urdf_path}")
            self.robot_id = p.loadURDF(
                self.urdf_path,
                [0.0, 0.0, 0.35],
                useFixedBase=False,
            )
            self._setup_robot_links_and_joints()

        # Per-episode friction randomization for domain robustness
        if self.randomize_friction and p is not None:
            lateral_friction = float(rng.uniform(0.8, 1.4))
            spinning_friction = float(rng.uniform(0.005, 0.02))
            if self.plane_id is not None:
                p.changeDynamics(self.plane_id, -1,
                                 lateralFriction=lateral_friction,
                                 spinningFriction=spinning_friction)
            if self.robot_id is not None:
                all_links = set(self.foot_link_indices + self.calf_link_indices)
                for idx in all_links:
                    p.changeDynamics(self.robot_id, idx,
                                     lateralFriction=lateral_friction,
                                     spinningFriction=spinning_friction)

        # Requirement 1: Initialize random continuous path
        self.episode_path = self._generate_continuous_path(rng)
        self._draw_path_debug_lines()

        # Requirement 2: Densely packed random pyramids of different sizes
        terrain_seed = int(rng.integers(0, 2**31 - 1))
        self._load_dense_terrain(terrain_seed)

        # Reset robot base and joints to nominal standing configuration
        self._reset_robot_pose()

        # Initialize CrawlGait for reference trajectory generation
        nominal_cycle_cmd = action_to_cycle_command(get_nominal_cycle_action())
        self.gait = CrawlGait(nominal_cycle_cmd, blend_time=0.0)
        self.gait.cycle_phase = 0.0

        self.episode_index += 1

        obs_dict = self._get_obs()
        obs = self._format_obs(obs_dict)
        info = {
            "episode_index": int(self.episode_index),
            "action_mode": str(self.action_mode),
            "path_length": float(self.task_config.path_length),
            "num_pyramids": int(len(self.terrain_ids)),
            "path_index": int(self._current_info_metrics["path_index"]),
            "path_distance": float(self._current_info_metrics["path_distance"]),
            "path_offset": float(self._current_info_metrics["path_offset"]),
            "velocity_tangent_angle": float(self._current_info_metrics["velocity_tangent_angle"]),
            "trunk_roll": float(self._current_info_metrics["trunk_roll"]),
            "trunk_pitch": float(self._current_info_metrics["trunk_pitch"]),
            "trunk_yaw": float(self._current_info_metrics["trunk_yaw"]),
            "base_linear_velocity": np.array(self._current_info_metrics["base_linear_velocity"], copy=True),
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[Union[Dict[str, np.ndarray], np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Apply action, execute trajectory, return transition tuple."""
        self.episode_step_count += 1
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Action shape mismatch: expected {self.action_space.shape}, got {action.shape}"
            )

        start_progress_idx = int(self._current_info_metrics.get("path_index", 0))
        dt_ctrl = self.sim_time_step * self.action_repeat

        fallen = False
        far_from_path = False
        reached_end = False
        terminated = False
        truncated = False
        last_substep = 0

        if self.action_mode == "corrective":
            # ------------------------------------------------------------------
            # Task-Space Residual Correction (Gangapurwala et al., ICRA 2021)
            # Additive deviations over nominal crawl reference trajectory setpoints
            # ------------------------------------------------------------------
            dt_step = self.control_step_dt
            n_substeps = max(1, int(round(dt_step / dt_ctrl)))
            dt_sub = dt_step / float(n_substeps)

            # Scaled additive task-space deviations
            deviations = action * CORRECTIVE_ACTION_SCALES
            delta_P_base = deviations[0:3]
            delta_O_base = deviations[3:6]
            delta_P_feet = {
                "FR": deviations[6:9],
                "FL": deviations[9:12],
                "RR": deviations[12:15],
                "RL": deviations[15:18],
            }

            # Exact path-following reference trajectory steering (Section III-A)
            # Combines Pure Pursuit lookahead, Stanley cross-track error correction,
            # omnidirectional lateral crabbing, and turn-adaptive forward speed scaling
            if self.episode_path is not None and len(self.episode_path.points) > 0:
                pos = self._sim_pos if self.dry_run else (
                    p.getBasePositionAndOrientation(self.robot_id)[0] if (p and self.robot_id) else [0.0, 0.0, 0.28]
                )
                nearest_idx, distance, lat_offset, tangent = self.episode_path.nearest(pos[:2])
                pts = self.episode_path.points
                total_pts = len(pts)

                # Lookahead waypoint on spline (~0.12m ahead: tightly follows curves)
                lookahead_idx = min(nearest_idx + 14, total_pts - 1)
                target_pt = pts[lookahead_idx]
                target_vec = target_pt - pos[:2]
                target_dist = float(np.linalg.norm(target_vec))

                # Ensure target vector always points along path travel direction (never backwards)
                if target_dist > 1e-4 and float(np.dot(target_vec, tangent)) > 0.0:
                    target_yaw = math.atan2(target_vec[1], target_vec[0])
                else:
                    target_yaw = math.atan2(tangent[1], tangent[0])

                trunk_yaw = float(self._sim_rpy[2]) if self.dry_run else self._get_trunk_orientation()[2]
                yaw_err = math.atan2(math.sin(target_yaw - trunk_yaw), math.cos(target_yaw - trunk_yaw))

                # Stanley cross-track error steering term:
                # lat_offset > 0 (robot right of path) -> steer left (+turn)
                # lat_offset < 0 (robot left of path) -> steer right (-turn)
                cross_yaw = math.atan2(3.0 * lat_offset, 0.30)
                turn_cmd = float(np.clip(1.5 * yaw_err + 1.2 * cross_yaw, -0.70, 0.70))

                # Dynamic forward speed scaling: decelerate on sharp turns to maintain tight turning radius
                turn_severity = abs(yaw_err)
                speed_scale = max(0.55, math.cos(min(turn_severity, math.pi / 2)))
                length_x = float(0.082 * speed_scale)

                # Omnidirectional quadruped lateral crabbing: directly eliminates lateral offset
                lat_stride = float(np.clip(0.35 * lat_offset, -0.024, 0.024))

                self.gait.cmd = CycleCommand.differential(
                    length_x=length_x,
                    length_y=lat_stride,
                    turn=turn_cmd,
                    height=0.14,
                    stance_z=-0.22,
                    stance_width=0.035,
                    swing_frac=0.18,
                    cycle_time=0.76,
                )

            for substep in range(n_substeps):
                last_substep = substep

                # Obstacle-aware swing start: read actual foot positions in hip frames
                actual_foot_pos = None
                trunk_roll = 0.0
                trunk_pitch = 0.0
                if not self.dry_run and p is not None and self.robot_id is not None and len(self.foot_link_indices) == 4:
                    try:
                        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot_id)
                        inv_b_pos, inv_b_quat = p.invertTransform(base_pos, base_quat)
                        actual_foot_pos = {}
                        for leg, foot_idx in zip(("FR", "FL", "RR", "RL"), self.foot_link_indices):
                            f_state = p.getLinkState(self.robot_id, foot_idx)
                            f_trunk, _ = p.multiplyTransforms(inv_b_pos, inv_b_quat, f_state[0], [0, 0, 0, 1])
                            r_hip = A1_HIP_OFFSETS[leg]
                            actual_foot_pos[leg] = (f_trunk[0] - r_hip[0], f_trunk[1] - r_hip[1], f_trunk[2] - r_hip[2])
                        trunk_roll, trunk_pitch, _ = self._get_trunk_orientation()
                    except Exception:
                        actual_foot_pos = None

                # Update nominal reference crawl gait with terrain-aware swing liftoff
                self.gait.update(dt_sub, actual_foot_pos=actual_foot_pos)

                # Active attitude leveling feedback (keeps trunk horizontal on uneven pyramid facets)
                roll_level = float(np.clip(0.12 * trunk_roll, -0.025, 0.025))
                pitch_level = float(np.clip(0.10 * trunk_pitch, -0.025, 0.025))

                # Compute corrected hip-relative foot targets and map via IK
                targets_ordered = []
                for leg in ("FR", "FL", "RR", "RL"):
                    p_ref = np.array(self.gait.foot_targets[leg], dtype=np.float32)
                    r_hip = A1_HIP_OFFSETS[leg]

                    # Rotational displacement: delta_O_base x r_hip
                    delta_rot = np.array([
                        -delta_O_base[2] * r_hip[1],
                         delta_O_base[2] * r_hip[0],
                         delta_O_base[0] * r_hip[1] - delta_O_base[1] * r_hip[0],
                    ], dtype=np.float32)

                    # Dynamic attitude leveling offset per leg
                    side_sign = float(SIDE[leg])  # +1 for FL/RL, -1 for FR/RR
                    fore_sign = 1.0 if leg.startswith("F") else -1.0  # +1 for FR/FL, -1 for RR/RL
                    z_level = side_sign * roll_level + fore_sign * pitch_level

                    # Corrected foot position relative to hip
                    p_corr = p_ref + delta_P_feet[leg] - delta_P_base - delta_rot
                    p_corr[2] += z_level

                    # Inverse Kinematics
                    q_leg = foot_to_joints(leg, float(p_corr[0]), float(p_corr[1]), float(p_corr[2]))
                    targets_ordered.extend(q_leg)

                target_positions = np.clip(targets_ordered, JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS)

                if self.dry_run:
                    self._sim_joints = target_positions.astype(np.float32).copy()
                    if self.episode_path is not None:
                        _, _, _, tangent = self.episode_path.nearest(self._sim_pos[:2])
                        motion_speed = 0.06 / 1.0  # nominal crawl speed ~0.06 m/s
                        self._sim_pos[:2] += tangent * (motion_speed * dt_ctrl)
                        self._sim_vel[:2] = tangent * motion_speed
                elif p is not None and self.robot_id is not None:
                    for j_idx, target_q in zip(self.joint_indices, target_positions):
                        p.setJointMotorControl2(
                            self.robot_id,
                            j_idx,
                            p.POSITION_CONTROL,
                            targetPosition=float(target_q),
                            force=MAX_MOTOR_FORCE,
                        )
                    for _ in range(self.action_repeat):
                        p.stepSimulation()
                    if self.render_mode == "human":
                        import time
                        time.sleep(dt_ctrl)

                # Substep termination check
                if self.dry_run:
                    base_z = float(self._sim_pos[2])
                    roll, pitch = float(self._sim_rpy[0]), float(self._sim_rpy[1])
                    nearest_idx, dist_to_path, _, _ = self.episode_path.nearest(self._sim_pos[:2])
                elif p is not None and self.robot_id is not None:
                    try:
                        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
                        base_z = float(pos[2])
                        roll, pitch, _ = self._get_trunk_orientation()
                        nearest_idx, dist_to_path, _, _ = self.episode_path.nearest(pos[:2])
                    except Exception:
                        terminated = True
                        break
                else:
                    base_z = 0.25
                    roll, pitch = 0.0, 0.0
                    dist_to_path = 0.0
                    nearest_idx = 0

                fallen = bool(base_z < 0.08 or abs(roll) > 1.15 or abs(pitch) > 1.15)
                far_from_path = bool(dist_to_path > 1.80)
                reached_end = bool(
                    nearest_idx >= len(self.episode_path.points) - 2
                    or (pos is not None and float(np.linalg.norm(pos[:2] - self.episode_path.points[-1][:2])) < 0.10)
                )

                if fallen or far_from_path:
                    terminated = True
                    break
                if reached_end:
                    truncated = True
                    break

                # Camera track in GUI mode
                if self.render_mode == "human" and p is not None and self.robot_id is not None and substep % 4 == 0:
                    pos, _ = p.getBasePositionAndOrientation(self.robot_id)
                    p.resetDebugVisualizerCamera(
                        cameraDistance=1.5,
                        cameraYaw=30.0,
                        cameraPitch=-30.0,
                        cameraTargetPosition=pos,
                    )

            obs_dict = self._get_obs()
            obs = self._format_obs(obs_dict)

            if self.episode_step_count >= self.max_episode_steps:
                truncated = True

            end_progress_idx = int(self._current_info_metrics["path_index"])
            total_pts = max(len(self.episode_path.points) - 1, 1)
            step_progress = max(0, end_progress_idx - start_progress_idx) / total_pts
            self._last_progress_index = end_progress_idx

            reward, reward_components = self._compute_paper_reward(
                action=action,
                obs_dict=obs_dict,
                joint_positions=obs_dict["joint_positions"],
                joint_velocities=obs_dict["joint_velocities"],
                foot_velocities=obs_dict["foot_velocities"],
                progress=step_progress,
                terminated=terminated,
                dt_step=dt_step,
            )

        else:
            # ------------------------------------------------------------------
            # High-level Gait Cycle parameterization mode
            # ------------------------------------------------------------------
            cmd = action_to_cycle_command(action)
            self.gait.cmd = cmd

            n_substeps = max(1, int(round(cmd.cycle_time / dt_ctrl)))
            dt_gait = cmd.cycle_time / float(n_substeps)

            for substep in range(n_substeps):
                last_substep = substep
                joint_targets = self.gait.update(dt_gait)
                targets_ordered = []
                for leg in ("FR", "FL", "RR", "RL"):
                    targets_ordered.extend(joint_targets[leg])
                target_positions = np.clip(targets_ordered, JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS)

                if self.dry_run:
                    self._sim_joints = target_positions.astype(np.float32).copy()
                    if self.episode_path is not None:
                        _, _, _, tangent = self.episode_path.nearest(self._sim_pos[:2])
                        mean_length_x = float(np.mean([cmd.legs[n].length_x for n in LEG_NAMES]))
                        motion_speed = mean_length_x / cmd.cycle_time if cmd.cycle_time > 0 else 0.0
                        self._sim_pos[:2] += tangent * (motion_speed * dt_ctrl)
                        self._sim_vel[:2] = tangent * motion_speed
                elif p is not None and self.robot_id is not None:
                    for j_idx, target_q in zip(self.joint_indices, target_positions):
                        p.setJointMotorControl2(
                            self.robot_id,
                            j_idx,
                            p.POSITION_CONTROL,
                            targetPosition=float(target_q),
                            force=MAX_MOTOR_FORCE,
                        )
                    for _ in range(self.action_repeat):
                        p.stepSimulation()
                    if self.render_mode == "human":
                        import time
                        time.sleep(dt_ctrl)

                # Substep termination check
                if self.dry_run:
                    base_z = float(self._sim_pos[2])
                    roll, pitch = float(self._sim_rpy[0]), float(self._sim_rpy[1])
                    nearest_idx, dist_to_path, _, _ = self.episode_path.nearest(self._sim_pos[:2])
                elif p is not None and self.robot_id is not None:
                    try:
                        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
                        base_z = float(pos[2])
                        roll, pitch, _ = self._get_trunk_orientation()
                        nearest_idx, dist_to_path, _, _ = self.episode_path.nearest(pos[:2])
                    except Exception:
                        terminated = True
                        break
                else:
                    base_z = 0.25
                    roll, pitch = 0.0, 0.0
                    dist_to_path = 0.0
                    nearest_idx = 0

                fallen = bool(base_z < 0.08 or abs(roll) > 1.15 or abs(pitch) > 1.15)
                far_from_path = bool(dist_to_path > 1.80)
                reached_end = bool(nearest_idx >= len(self.episode_path.points) - 5)

                if fallen or far_from_path:
                    terminated = True
                    break
                if reached_end:
                    truncated = True
                    break

                if self.render_mode == "human" and p is not None and self.robot_id is not None and substep % 4 == 0:
                    pos, _ = p.getBasePositionAndOrientation(self.robot_id)
                    p.resetDebugVisualizerCamera(
                        cameraDistance=1.5,
                        cameraYaw=30.0,
                        cameraPitch=-30.0,
                        cameraTargetPosition=pos,
                    )

            obs_dict = self._get_obs()
            obs = self._format_obs(obs_dict)

            if self.episode_step_count >= self.max_episode_steps:
                truncated = True

            end_progress_idx = int(self._current_info_metrics["path_index"])
            total_pts = max(len(self.episode_path.points) - 1, 1)
            cycle_progress = max(0, end_progress_idx - start_progress_idx) / total_pts
            self._last_progress_index = end_progress_idx

            reward, reward_components = self._compute_cycle_reward(
                action=action,
                obs_dict=obs_dict,
                cycle_progress=cycle_progress,
                terminated=terminated,
                dt_cycle=cmd.cycle_time,
            )

        # Update previous temporal states
        self._previous_action = action.copy()
        self._previous_joint_positions = obs_dict["joint_positions"].copy()
        self._previous_joint_velocities = obs_dict["joint_velocities"].copy()
        self._previous_foot_velocities = obs_dict["foot_velocities"].copy()

        info = {
            "step_count": int(self.episode_step_count),
            "action_mode": str(self.action_mode),
            "fallen": bool(fallen),
            "far_from_path": bool(far_from_path),
            "reached_end": bool(reached_end),
            "substeps_executed": int(last_substep + 1),
            "reward_components": dict(reward_components),
            "path_index": int(self._current_info_metrics["path_index"]),
            "path_distance": float(self._current_info_metrics["path_distance"]),
            "path_offset": float(self._current_info_metrics["path_offset"]),
            "velocity_tangent_angle": float(self._current_info_metrics["velocity_tangent_angle"]),
            "trunk_roll": float(self._current_info_metrics["trunk_roll"]),
            "trunk_pitch": float(self._current_info_metrics["trunk_pitch"]),
            "trunk_yaw": float(self._current_info_metrics["trunk_yaw"]),
            "base_linear_velocity": np.array(self._current_info_metrics["base_linear_velocity"], copy=True),
        }

        if self.render_mode == "human" and p is not None and self.robot_id is not None:
            pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=30.0,
                cameraPitch=-30.0,
                cameraTargetPosition=pos,
            )

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Render frame."""
        if self.render_mode == "rgb_array":
            if p is None or self.robot_id is None:
                return np.zeros((480, 640, 3), dtype=np.uint8)
            pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            view_mat = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=pos,
                distance=1.6,
                yaw=45.0,
                pitch=-30.0,
                roll=0.0,
                upAxisIndex=2,
            )
            proj_mat = p.computeProjectionMatrixFOV(
                fov=60.0, aspect=640.0 / 480.0, nearVal=0.1, farVal=100.0
            )
            img = p.getCameraImage(
                width=640,
                height=480,
                viewMatrix=view_mat,
                projectionMatrix=proj_mat,
                renderer=p.ER_TINY_RENDERER,
            )
            rgb = np.asarray(img[2], dtype=np.uint8).reshape((480, 640, 4))
            return rgb[:, :, :3]
        return None

    def close(self):
        """Disconnect PyBullet simulation and clean up resources."""
        if p is not None and self.physics_client_id is not None:
            try:
                p.disconnect(self.physics_client_id)
            except Exception:
                pass
            self.physics_client_id = None
            self.robot_id = None
            self.plane_id = None
            self.terrain_ids = []
            self.debug_line_ids = []
