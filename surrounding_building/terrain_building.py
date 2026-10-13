from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple
try:
    import pybullet as p
except ImportError:
    p = None


def create_pyramid(position, size=1.5, height=0.5, color=[0.8, 0.3, 0.1, 1.0]):
    if p is None:
        raise ImportError("pybullet is required to create pyramid meshes in simulation")
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
        indices=indices
    )

    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        vertices=vertices,
        indices=indices,
        rgbaColor=color
    )

    pyramid_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position
    )

    return pyramid_id


def create_surroundings():
    """Creates 10 pyramids in the surroundings of the robot."""
    configs = [
        # (position, size, height, color)
        (( 3.0,  0.0, 0.01), 1.5, 0.50, [0.85, 0.35, 0.15, 1.0]),  # Directly in front
        (( 2.5,  2.2, 0.01), 1.2, 0.40, [0.75, 0.45, 0.20, 1.0]),  # Front-left
        (( 2.8, -2.0, 0.01), 1.3, 0.45, [0.80, 0.30, 0.25, 1.0]),  # Front-right
        (( 5.0,  1.2, 0.01), 1.8, 0.60, [0.90, 0.40, 0.10, 1.0]),  # Mid-distance left
        (( 5.5, -1.5, 0.01), 1.6, 0.55, [0.70, 0.35, 0.30, 1.0]),  # Mid-distance right
        (( 7.5,  0.5, 0.01), 2.0, 0.70, [0.85, 0.50, 0.20, 1.0]),  # Far-front center
        (( 7.0, -3.0, 0.01), 1.4, 0.50, [0.65, 0.40, 0.35, 1.0]),  # Far-right
        (( 6.8,  3.5, 0.01), 1.5, 0.50, [0.80, 0.45, 0.15, 1.0]),  # Far-left
        ((-2.0,  2.5, 0.01), 1.2, 0.40, [0.75, 0.30, 0.20, 1.0]),  # Rear-left
        ((-2.5, -2.2, 0.01), 1.3, 0.45, [0.70, 0.40, 0.25, 1.0]),  # Rear-right
    ]

    pyramid_ids = []
    for pos, size, height, color in configs:
        pid = create_pyramid(position=pos, size=size, height=height, color=color)
        pyramid_ids.append(pid)

    return pyramid_ids


def create_terrain():
    return create_surroundings()


# ----------------------------------------------------------------------
# Random configurable packed-pyramid terrain
# ----------------------------------------------------------------------
# earthy palette sampled by the random generator (rgba)
DEFAULT_PALETTE = [
    (0.85, 0.35, 0.15, 1.0),
    (0.75, 0.45, 0.20, 1.0),
    (0.80, 0.30, 0.25, 1.0),
    (0.90, 0.40, 0.10, 1.0),
    (0.70, 0.35, 0.30, 1.0),
    (0.65, 0.40, 0.35, 1.0),
]


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


def _resolve(config, overrides) -> RandomTerrainConfig:
    cfg = config or RandomTerrainConfig()
    return replace(cfg, **overrides) if overrides else cfg


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
            yield x, y, size, height, color


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
    return [
        create_pyramid((x, y, cfg.base_z), size=size,
                       height=height, color=color)
        for x, y, size, height, color in terrain_layout(cfg)
    ]
