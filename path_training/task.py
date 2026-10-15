"""Random path episodes, compact observations, and path-following rewards.

PyBullet is imported lazily by terrain setup / observation methods so path
sampling and reward calculations can also be used by non-simulation code.
"""

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from surrounding_building.terrain_building import (
    RandomTerrainConfig, create_random_terrain,
    create_random_terrain_along_path,
)


@dataclass
class PathTaskConfig:
    path_length: float = 2.5
    path_width: float = 1.0
    waypoint_count: int = 6
    path_samples: int = 300
    path_tolerance: float = 0.20
    target_speed: float = 0.20
    depth_width: int = 64
    depth_height: int = 48
    # Reward weights
    path_weight: float = 2.0
    path_deviation_weight: float = 1.0
    velocity_weight: float = 0.35
    stability_weight: float = 0.35
    progress_weight: float = 1.0
    waypoint_bonus: float = 2.0
    off_path_termination_distance: float = 1.25
    terrain: RandomTerrainConfig = None

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
class EpisodePath:
    """Sampled world-frame path and its per-sample unit tangent."""

    points: np.ndarray
    tangents: np.ndarray

    def nearest(self, position_xy):
        delta = self.points - np.asarray(position_xy, dtype=np.float64)[:2]
        index = int(np.argmin(np.einsum("ij,ij->i", delta, delta)))
        tangent = self.tangents[index]
        # Signed lateral error: positive means left of the direction of travel.
        offset = float(tangent[0] * delta[index, 1] - tangent[1] * delta[index, 0])
        return index, float(np.linalg.norm(delta[index])), offset, tangent.copy()


def _catmull_rom(control, samples, smooth_passes=1):
    """Sample a C1-continuous Catmull-Rom spline through 2D control points.

    Args:
        control: Control points array (n, 2)
        samples: Number of output samples
        smooth_passes: Number of smoothing passes on control points (default 1)
    """
    # Optional smoothing of control points for gentler curves.
    if smooth_passes > 0:
        control = control.copy()
        for _ in range(smooth_passes):
            smoothed = control.copy()
            for i in range(1, len(control) - 1):
                # Laplacian smoothing: average with neighbors.
                smoothed[i] = 0.5 * control[i] + 0.25 * (control[i-1] + control[i+1])
            control = smoothed

    points = []
    segments = len(control) - 1
    per_segment = max(2, int(math.ceil(samples / segments)))
    for i in range(segments):
        p0 = control[max(i - 1, 0)]
        p1, p2 = control[i], control[i + 1]
        p3 = control[min(i + 2, len(control) - 1)]
        for t in np.linspace(0.0, 1.0, per_segment, endpoint=False):
            t2, t3 = t * t, t * t * t
            point = 0.5 * ((2 * p1) + (-p0 + p2) * t
                           + (2*p0 - 5*p1 + 4*p2 - p3) * t2
                           + (-p0 + 3*p1 - 3*p2 + p3) * t3)
            points.append(point)
    points.append(control[-1])
    points = np.asarray(points, dtype=np.float64)
    deriv = np.gradient(points, axis=0)
    norms = np.linalg.norm(deriv, axis=1, keepdims=True)
    tangents = deriv / np.maximum(norms, 1e-9)
    return EpisodePath(points, tangents)


class PathTrainingTask:
    """Training episode state and reusable observation/reward calculations."""

    def __init__(self, config: Optional[PathTaskConfig] = None, seed=None):
        self.config = config or PathTaskConfig()
        self.seed = seed
        self.episode_index = 0
        self.path = None
        self.terrain_ids = []
        self._last_progress_index = None
        self._route_complete = False

    def reset_episode(self, seed=None, load_terrain=True):
        """Create a new random path and terrain realization for an episode.

        Load a plane in the environment before calling this with
        ``load_terrain=True``. A supplied seed makes the episode reproducible.
        """
        if seed is not None:
            self.seed = int(seed)
        elif self.seed is None:
            self.seed = int(np.random.SeedSequence().generate_state(1)[0])
        rng = np.random.default_rng(self.seed + self.episode_index)
        if self.terrain_ids:
            import pybullet as p
            for body_id in self.terrain_ids:
                p.removeBody(body_id)
            self.terrain_ids = []
        cfg = self.config

        # Randomize path length and waypoint count per episode.
        path_length = float(rng.uniform(cfg.path_length * 0.7, cfg.path_length * 1.3))
        waypoint_count = int(rng.integers(max(4, cfg.waypoint_count - 2), cfg.waypoint_count + 3))

        # Start at origin and finish forward, with randomized lateral bends.
        xs = np.linspace(0.0, path_length, waypoint_count)
        # Add random curvature: each waypoint gets lateral + longitudinal jitter.
        # Reduced lateral range for smoother paths (was full path_width/2).
        lateral_range = cfg.path_width * 0.25  # Reduced from 0.5 to 0.25
        lateral_jitter = rng.uniform(-lateral_range, lateral_range, waypoint_count)
        longitudinal_jitter = rng.uniform(-path_length * 0.02, path_length * 0.02, waypoint_count)  # Reduced from 0.05
        lateral_jitter[0] = 0.0
        longitudinal_jitter[0] = 0.0
        # End point has some lateral variation but must be forward.
        lateral_jitter[-1] = float(rng.uniform(-cfg.path_width * 0.15, cfg.path_width * 0.15))  # Reduced from 1/3
        longitudinal_jitter[-1] = float(rng.uniform(-path_length * 0.02, path_length * 0.02))  # Reduced from 0.03
        ys = lateral_jitter
        xs = xs + longitudinal_jitter
        control = np.column_stack((xs, ys))
        self.path = _catmull_rom(control, cfg.path_samples, smooth_passes=2)

        terrain_cfg = self.config.terrain
        # Derive an independent terrain seed for every episode.
        terrain_cfg = RandomTerrainConfig(
            seed=int(rng.integers(0, 2**31 - 1)), extent=terrain_cfg.extent,
            spacing=terrain_cfg.spacing, size=terrain_cfg.size,
            height=terrain_cfg.height, density=terrain_cfg.density,
            jitter=terrain_cfg.jitter, clear_radius=terrain_cfg.clear_radius,
            base_z=terrain_cfg.base_z, palette=terrain_cfg.palette,
        )
        # Create terrain covering the full path with margin.
        self.terrain_ids = (create_random_terrain_along_path(
            self.path, terrain_cfg, margin=2.0) if load_terrain else [])
        self._last_progress_index = 0
        self._route_complete = False
        self.episode_index += 1
        return self.path

    def observe(self, robot_id, depth_camera=None):
        """Return named robot state, path errors, and optional resized depth.

        Joint position and velocity vectors include all movable joints. Depth
        is resized with area averaging to keep the image input compact.
        """
        import pybullet as p

        if self.path is None:
            raise RuntimeError("reset_episode() must be called before observe()")
        movable = [j for j in range(p.getNumJoints(robot_id))
                   if p.getJointInfo(robot_id, j)[2] != p.JOINT_FIXED]
        states = p.getJointStates(robot_id, movable) if movable else []
        joint_position = np.asarray([s[0] for s in states], dtype=np.float32)
        joint_velocity = np.asarray([s[1] for s in states], dtype=np.float32)
        position, quaternion = p.getBasePositionAndOrientation(robot_id)
        linear_velocity, _ = p.getBaseVelocity(robot_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(quaternion)
        index, distance, lateral_offset, tangent = self.path.nearest(position[:2])
        velocity_xy = np.asarray(linear_velocity[:2], dtype=np.float64)
        speed = float(np.linalg.norm(velocity_xy))
        if speed > 1e-6:
            dot = float(np.clip(np.dot(velocity_xy / speed, tangent), -1.0, 1.0))
            velocity_angle = math.atan2(tangent[0] * velocity_xy[1]
                                        - tangent[1] * velocity_xy[0], dot)
        else:
            velocity_angle = 0.0

        depth = None
        if depth_camera is not None:
            raw = np.asarray(depth_camera.capture(), dtype=np.float32)
            try:
                import cv2
                depth = cv2.resize(raw, (self.config.depth_width,
                                         self.config.depth_height),
                                   interpolation=cv2.INTER_AREA)
            except ImportError:
                # Deterministic nearest-cell fallback when OpenCV is absent.
                rows = np.linspace(0, raw.shape[0] - 1,
                                   self.config.depth_height).astype(int)
                cols = np.linspace(0, raw.shape[1] - 1,
                                   self.config.depth_width).astype(int)
                depth = raw[np.ix_(rows, cols)]
        return {
            "joint_position": joint_position,
            "joint_velocity": joint_velocity,
            "trunk_orientation": np.asarray([roll, pitch, yaw], dtype=np.float32),
            "path_offset": np.float32(lateral_offset),
            "path_distance": np.float32(distance),
            "path_tangent": tangent.astype(np.float32),
            "path_target": self.path.points[index].astype(np.float32),
            "velocity_tangent_angle": np.float32(velocity_angle),
            "base_linear_velocity": np.asarray(linear_velocity, dtype=np.float32),
            "path_index": index,
            "depth": depth,
        }

    def reward(self, observation, *, terminated=False):
        """Reward adherence and forward progress on this episode's random path."""
        cfg = self.config
        distance = float(observation["path_distance"])
        path_term = math.exp(-0.5 * (distance / max(cfg.path_tolerance, 1e-6)) ** 2)
        # A continuous penalty makes leaving the route costly even when the
        # Gaussian path bonus has already become close to zero.
        deviation_penalty = min(distance / max(cfg.path_tolerance, 1e-6), 5.0)
        # Additional harsh penalty for being significantly off-path (beyond tolerance).
        off_path_penalty = 0.0
        if distance > cfg.path_tolerance * 2.0:
            off_path_penalty = (distance - cfg.path_tolerance * 2.0) * 2.0
        velocity = np.asarray(observation["base_linear_velocity"], dtype=float)[:2]
        speed = float(np.linalg.norm(velocity))
        angle = float(observation["velocity_tangent_angle"])
        # Reward forward velocity aligned with the local path tangent and near
        # the commanded speed. Reverse motion receives no positive alignment.
        velocity_term = max(0.0, math.cos(angle)) * math.exp(
            -abs(speed - cfg.target_speed) / max(cfg.target_speed, 1e-6))
        roll, pitch = np.asarray(observation["trunk_orientation"], dtype=float)[:2]
        stability_term = math.exp(-0.5 * ((roll / 0.35) ** 2 + (pitch / 0.35) ** 2))
        index = int(observation["path_index"])
        previous_index = self._last_progress_index
        progress = 0.0 if previous_index is None else max(
            0, index - previous_index) / max(len(self.path.points) - 1, 1)
        # Only grant progress reward while the robot remains near its assigned
        # route, so cutting across bends is not an attractive shortcut.
        progress *= path_term
        self._last_progress_index = max(previous_index or 0, index)
        reached_goal = (index >= len(self.path.points) - 2
                        and distance <= cfg.path_tolerance)
        goal_bonus = cfg.waypoint_bonus if reached_goal and not self._route_complete else 0.0
        self._route_complete = self._route_complete or reached_goal
        reward = (cfg.path_weight * path_term
                  - cfg.path_deviation_weight * deviation_penalty
                  - cfg.path_deviation_weight * off_path_penalty
                  + cfg.velocity_weight * velocity_term
                  + cfg.stability_weight * stability_term
                  + cfg.progress_weight * progress
                  + goal_bonus)
        off_path = distance > cfg.off_path_termination_distance
        return {
            "total": float(reward), "path": path_term,
            "path_deviation_penalty": deviation_penalty,
            "off_path_penalty": off_path_penalty,
            "velocity_alignment": velocity_term, "stability": stability_term,
            "progress": progress, "goal_bonus": goal_bonus,
            "off_path": off_path,
        }
