"""
randomization_check.py -- verify randomization of target paths and
obstacles/terrain for the refactored environment package
(environment.gym_env.QuadrupedPathTerrainEnv).

Checks:
1. Path sampling : distinct seeds -> distinct Catmull-Rom paths; same seed
                   -> bit-identical path; spawn pinned at origin; lateral
                   bends stay inside path_width; tangents are unit length.
2. Terrain layout: distinct seeds -> distinct packed-pyramid layouts; same
                   seed -> identical layout; every pyramid centre stays
                   outside clear_radius; sizes/heights inside config range.
3. Env integration (pybullet DIRECT): successive resets produce different
                   paths and different loaded terrain bodies, episode
                   friction stays inside the randomization window.
4. path_training.PathTrainingTask path sampling (used by existing tests).

Run from the project root:
  python tests/randomization_check.py
"""

import math
import sys
from pathlib import Path

import numpy as np

# Navigates up two levels (from script.py -> tests/ -> project/) and
# adds it to Python's search path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from environment import QuadrupedPathTerrainEnv
from environment.config import RandomTerrainConfig
from environment.terrain import terrain_layout
from path_training import PathTaskConfig, PathTrainingTask


def _path_fingerprint(points: np.ndarray) -> bytes:
    return np.round(points, 6).tobytes()


def check_path_sampling():
    cfg = PathTaskConfig()
    env = QuadrupedPathTerrainEnv(dry_run=True, seed=123)
    env.reset(seed=123)
    path_a = env.episode_path.points.copy()
    path_a_t = env.episode_path.tangents.copy()

    # Same seed -> identical path
    env.reset(seed=123)
    path_same = env.episode_path.points.copy()
    assert np.array_equal(path_a, path_same), "same seed must reproduce the path"

    # Different seeds -> different paths
    fingerprints = {_path_fingerprint(path_a)}
    for seed in (1, 2, 3, 4, 5):
        env.reset(seed=seed)
        pts = env.episode_path.points
        fingerprints.add(_path_fingerprint(pts))
        assert len(pts) >= cfg.path_samples, "path sample count mismatch"
        # spawn pinned at origin, all lateral bends inside path_width
        assert np.allclose(pts[0], (0.0, 0.0)), "path must start at origin"
        assert np.max(np.abs(pts[:, 1])) <= cfg.path_width / 2.0 + 1e-9, \
            "lateral bend exceeded path_width"
        assert pts[-1, 0] == cfg.path_length, "path must end at path_length"
    assert len(fingerprints) >= 6, "paths did not vary across seeds"

    # tangents from the first draw are unit length
    norms = np.linalg.norm(path_a_t, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6), "tangents must be unit length"
    env.close()
    print(f"[OK] path sampling: {len(fingerprints)} distinct paths, "
          f"spawn pinned, bends within +/-{cfg.path_width / 2:.2f} m")


def check_terrain_layout():
    cfg_a = RandomTerrainConfig(seed=11)
    cfg_b = RandomTerrainConfig(seed=12)

    layout_a1 = terrain_layout(cfg_a)
    layout_a2 = terrain_layout(cfg_a)
    layout_b = terrain_layout(cfg_b)

    assert layout_a1 == layout_a2, "same seed must reproduce the layout"
    assert layout_a1 != layout_b, "different seeds must change the layout"
    assert len(layout_a1) > 0, "default terrain must place pyramids"

    for x, y, size, height, _color in layout_a1:
        assert math.hypot(x, y) >= cfg_a.clear_radius, \
            "pyramid inside the spawn clear disc"
        assert cfg_a.size[0] <= size <= cfg_a.size[1], "size out of range"
        assert cfg_a.height[0] <= height <= cfg_a.height[1], "height out of range"
    print(f"[OK] terrain layout: {len(layout_a1)} pyramids, "
          f"seed-stable and seed-sensitive, clear_radius respected")


def check_env_integration():
    import pybullet as p

    env = QuadrupedPathTerrainEnv()  # real pybullet, DIRECT mode
    path_fps, terrain_positions, frictions = set(), [], []

    for episode in range(4):
        obs, info = env.reset()
        assert env.observation_space.contains(obs), "obs outside observation space"
        path_fps.add(_path_fingerprint(env.episode_path.points))

        # loaded obstacle bodies for this episode
        positions = sorted(
            tuple(np.round(p.getBasePositionAndOrientation(bid)[0], 5))
            for bid in env.terrain_ids
        )
        terrain_positions.append(tuple(positions))
        assert len(env.terrain_ids) == info["num_pyramids"], \
            "info num_pyramids must match loaded terrain bodies"
        assert info["num_pyramids"] > 0, "no obstacles loaded"
        # obstacle bodies must not accumulate across episodes (ids are
        # recycled by PyBullet, so count, not identity, is the invariant)
        total_bodies = p.getNumBodies()
        assert total_bodies == 2 + info["num_pyramids"], \
            f"terrain bodies leaked across episodes ({total_bodies} bodies)"

        # per-episode friction randomization (plane base link)
        frictions.append(float(p.getDynamicsInfo(env.plane_id, -1)[1]))

    assert len(path_fps) == 4, "paths repeated across unseeded resets"
    assert len(set(terrain_positions)) >= 2, \
        "terrain layout did not change across resets"
    assert all(0.8 <= f <= 1.4 for f in frictions), \
        f"friction outside randomization window: {frictions}"
    assert len(set(frictions)) >= 2, "friction did not vary across episodes"

    env.close()
    print(f"[OK] env integration: 4 resets -> {len(path_fps)} paths, "
          f"{len(set(terrain_positions))} distinct terrains, "
          f"frictions={[round(f, 3) for f in frictions]}")


def check_legacy_task_paths():
    task = PathTrainingTask(PathTaskConfig())
    fps = set()
    for seed in range(5):
        task.reset_episode(seed=seed, load_terrain=False)
        fps.add(_path_fingerprint(task.path.points))
    assert len(fps) == 5, "PathTrainingTask paths did not vary with seed"
    print(f"[OK] path_training.PathTrainingTask: {len(fps)} distinct paths")


def main():
    check_path_sampling()
    check_terrain_layout()
    check_env_integration()
    check_legacy_task_paths()
    print("ALL RANDOMIZATION CHECKS PASSED")


if __name__ == "__main__":
    main()
