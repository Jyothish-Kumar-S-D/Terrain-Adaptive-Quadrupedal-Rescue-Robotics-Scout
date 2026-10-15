"""Task module: random continuous Catmull-Rom path episodes + path metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import PathTaskConfig, RandomTerrainConfig
from .terrain import create_random_terrain


@dataclass
class EpisodePath:
    """Sampled world-frame path and its per-sample unit tangent."""

    points: np.ndarray
    tangents: np.ndarray

    def nearest(self, position_xy):
        delta = self.points - np.asarray(position_xy, dtype=np.float64)[:2]
        index = int(np.argmin(np.einsum("ij,ij->i", delta, delta)))
        tangent = self.tangents[index]
        offset = float(tangent[0] * delta[index, 1] - tangent[1] * delta[index, 0])
        return index, float(np.linalg.norm(delta[index])), offset, tangent.copy()


def _catmull_rom(control, samples):
    """Sample a C1-continuous Catmull-Rom spline through 2D control points."""
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
                           + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                           + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
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

    def reset_episode(self, seed=None, load_terrain=True):
        """Create a new random path and terrain realization for an episode."""
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

        xs = np.linspace(0.0, cfg.path_length, cfg.waypoint_count)
        ys = rng.uniform(-cfg.path_width / 2, cfg.path_width / 2,
                         cfg.waypoint_count)
        ys[0] = 0.0
        ys[-1] = float(rng.uniform(-cfg.path_width / 3, cfg.path_width / 3))
        control = np.column_stack((xs, ys))
        self.path = _catmull_rom(control, cfg.path_samples)

        terrain_cfg = self.config.terrain
        terrain_cfg = RandomTerrainConfig(
            seed=int(rng.integers(0, 2**31 - 1)), extent=terrain_cfg.extent,
            spacing=terrain_cfg.spacing, size=terrain_cfg.size,
            height=terrain_cfg.height, density=terrain_cfg.density,
            jitter=terrain_cfg.jitter, clear_radius=terrain_cfg.clear_radius,
            base_z=terrain_cfg.base_z, palette=terrain_cfg.palette,
        )
        self.terrain_ids = create_random_terrain(terrain_cfg) if load_terrain else []
        self._last_progress_index = 0
        self.episode_index += 1
        return self.path