"""Dense packed-pyramid terrain generator.

PyBullet is imported lazily so the layout sampling can be used by
non-simulation code (e.g. for statistics or visualisation).
"""

from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import List, Optional, Sequence, Tuple

try:
    import pybullet as p
except ImportError:
    p = None

from .config import RandomTerrainConfig

# earthy palette sampled by the random generator (rgba)
DEFAULT_PALETTE = [
    (0.85, 0.35, 0.15, 1.0),
    (0.75, 0.45, 0.20, 1.0),
    (0.80, 0.30, 0.25, 1.0),
    (0.90, 0.40, 0.10, 1.0),
    (0.70, 0.35, 0.30, 1.0),
    (0.65, 0.40, 0.35, 1.0),
]


def _resolve(config: Optional[RandomTerrainConfig],
             overrides: Optional[dict] = None) -> RandomTerrainConfig:
    cfg = config or RandomTerrainConfig()
    return replace(cfg, **overrides) if overrides else cfg


def create_pyramid(position, size=1.5, height=0.5, color=None):
    """Create a static pyramid mesh body in PyBullet."""
    if p is None:
        raise ImportError("pybullet is required to create pyramid meshes in simulation")
    if color is None:
        color = [0.8, 0.3, 0.1, 1.0]
    half = size / 2.0
    vertices = [
        (-half, -half, 0.0),
        ( half, -half, 0.0),
        ( half,  half, 0.0),
        (-half,  half, 0.0),
        (0.0, 0.0, height),
    ]
    indices = [
        0, 1, 4,
        1, 2, 4,
        2, 3, 4,
        3, 0, 4,
        0, 3, 2,
        0, 2, 1,
    ]
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        vertices=vertices,
        indices=indices,
    )
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        vertices=vertices,
        indices=indices,
        rgbaColor=color,
    )
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
    )


def terrain_layout(config: Optional[RandomTerrainConfig] = None,
                   **overrides) -> List[Tuple[float, float, float, float, list]]:
    """Deterministically sample the terrain placements.

    Yields (x, y, size, height, color) for every pyramid of the
    config. Same config -> same layout, every call (the RNG is
    re-seeded internally), so stats can be computed without
    building any collision geometry.
    """
    cfg = _resolve(config, overrides)
    assert cfg.extent > 0.0 and cfg.spacing > 0.0, "extent/spacing must be > 0"
    assert 0.0 < cfg.density <= 1.0, "density must be in (0, 1]"
    assert cfg.size[0] <= cfg.size[1], "size-min must be <= size-max"
    assert cfg.height[0] <= cfg.height[1], "height-min must be <= height-max"

    rng = random.Random(cfg.seed)
    palette = list(cfg.palette) if cfg.palette else list(DEFAULT_PALETTE)

    pitch = max(cfg.spacing, 1e-3)
    cells = max(1, int(cfg.extent / pitch))
    start = -cfg.extent / 2.0 + pitch / 2.0

    layout = []
    for i in range(cells):
        for j in range(cells):
            x = start + i * pitch + rng.uniform(-cfg.jitter, cfg.jitter)
            y = start + j * pitch + rng.uniform(-cfg.jitter, cfg.jitter)
            if math.hypot(x, y) < cfg.clear_radius:
                continue  # keep the spawn disc flat
            if rng.random() > cfg.density:
                continue
            size = rng.uniform(*cfg.size)
            height = rng.uniform(*cfg.height)
            color = list(palette[rng.randrange(len(palette))])
            layout.append((x, y, size, height, color))
    return layout


def create_random_terrain(config: Optional[RandomTerrainConfig] = None,
                           **overrides) -> List[int]:
    """Build the random packed-pyramid terrain on top of whatever
    plane is already loaded in the PyBullet env.

    Accepts a RandomTerrainConfig (or None for defaults) plus
    keyword overrides for any of its fields, e.g.
    create_random_terrain(seed=3, extent=4.0, height=(0.02, 0.1)).

    Returns the list of pyramid body ids. Every pyramid is a
    static mesh body with BOTH collision and visual geometry, so
    feet collide with (and climb over) them.
    """
    cfg = _resolve(config, overrides)
    layout = terrain_layout(cfg)
    return [
        create_pyramid((x, y, cfg.base_z), size=size,
                       height=height, color=color)
        for x, y, size, height, color in layout
    ]