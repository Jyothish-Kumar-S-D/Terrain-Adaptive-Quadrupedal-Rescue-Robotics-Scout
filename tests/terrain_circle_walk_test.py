"""
terrain_circle_walk_test.py -- circle-walk the A1 over a random,
configurable packed-pyramid terrain.

Loads the collision plane from pybullet_data, then overlays a
jittered grid of random pyramids (built by
surrounding_building/terrain_building.create_random_terrain,
same mesh-with-collision approach as create_surroundings) to turn
the flat ground into a rugged training patch, then runs the
circle-walk loop from circle_wlk_test.py over it.

A disc of `clear_radius` around the origin is kept flat so the
robot can stand up safely before the gait starts.

Run from the project root:
  python tests/terrain_circle_walk_test.py --radius 0.6 --seed 7
  python tests/terrain_circle_walk_test.py --direct --laps 2
  python tests/terrain_circle_walk_test.py --extent 4.0 --spacing 0.22 \
      --size-min 0.14 --size-max 0.30 --height-min 0.02 --height-max 0.10

radius > 0: circle centre on the robot's left (counter-clockwise).
Needs circle_wlk_test.py and IK_module_test.py in the same folder.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)
tests_dir = str(Path(__file__).resolve().parent)
if tests_dir not in sys.path:
    sys.path.append(tests_dir)

import pybullet as p
import pybullet_data

from surrounding_building.terrain_building import (
    RandomTerrainConfig, terrain_layout,
)
from path_training import PathTaskConfig, PathTrainingTask

# same-folder test modules
from circle_wlk_test import circle_command, fit_circle
from IK_module_test import (
    DT, Z_DOWN, Z_STAND, CG_X,
    leg_joint_map, command_legs, crouch_targets, step, stand_up, yaw_of,
)
from Inverse_Kinematics.IK_A1 import LEG_NAMES, CrawlGait


def main():
    ap = argparse.ArgumentParser(
        description="circle-walk the A1 over a random packed-pyramid terrain")
    # circle-walk params (same meaning as in circle_wlk_test.py)
    ap.add_argument("--urdf", default="urdf/a1_unitree.urdf")
    ap.add_argument("--radius", type=float, default=0.6,
                    help="signed circle radius [m]; + = left/CCW, - = right/CW")
    ap.add_argument("--speed", type=float, default=0.09, help="body speed [m/s]")
    ap.add_argument("--cycle-time", type=float, default=1.0)
    ap.add_argument("--height", type=float, default=0.10,
                    help="swing foot clearance [m]")
    ap.add_argument("--laps", type=float, default=1.0)
    ap.add_argument("--direct", action="store_true", help="headless, no realtime")

    # random terrain params
    g = ap.add_argument_group("random terrain")
    g.add_argument("--seed", type=int, default=7,
                   help="RNG seed; same seed -> same terrain")
    g.add_argument("--extent", type=float, default=6.7,
                   help="side length [m] of the square terrain patch")
    g.add_argument("--spacing", type=float, default=0.24,
                   help="grid pitch [m]; <= pyramid size packs tightly")
    g.add_argument("--size-min", dest="size_min", type=float, default=0.16,
                   help="min pyramid base side [m]")
    g.add_argument("--size-max", dest="size_max", type=float, default=0.34,
                   help="max pyramid base side [m]")
    g.add_argument("--height-min", dest="height_min", type=float, default=0.02,
                   help="min pyramid height [m]")
    g.add_argument("--height-max", dest="height_max", type=float, default=0.08,
                   help="max pyramid height [m]")
    g.add_argument("--density", type=float, default=1.0,
                   help="fraction of grid cells that get a pyramid")
    g.add_argument("--jitter", type=float, default=0.05,
                   help="max random offset of a pyramid centre [m]")
    g.add_argument("--clear-radius", dest="clear_radius", type=float,
                   default=0.45, help="flat spawn disc around origin [m]")
    g.add_argument("--base-z", dest="base_z", type=float, default=0.0,
                   help="pyramid base height above the plane [m]")
    args = ap.parse_args()

    if abs(args.radius) < 0.25:
        print("warning: radius < ~0.25 m will hit stride limits; expect heavy scaling")
    if args.height_max > args.height:
        print(f"warning: tallest pyramid ({args.height_max:.2f} m) exceeds the "
              f"swing clearance ({args.height:.2f} m); feet will clip obstacles")

    cfg = RandomTerrainConfig(
        seed=args.seed, extent=args.extent, spacing=args.spacing,
        size=(args.size_min, args.size_max),
        height=(args.height_min, args.height_max),
        density=args.density, jitter=args.jitter,
        clear_radius=args.clear_radius, base_z=args.base_z,
    )
    task = PathTrainingTask(
        PathTaskConfig(terrain=cfg), seed=args.seed,
    )

    # terrain stats, computed from the layout without touching the sim
    layout = list(terrain_layout(cfg))
    heights = [h for _, _, _, h, _ in layout]
    base_area = sum(s * s * 0.5 for _, _, s, _, _ in layout)  # triangle faces
    patch_area = args.extent ** 2 - math.pi * cfg.clear_radius ** 2
    print(f"terrain: {len(layout)} pyramids over {args.extent:.1f} m patch "
          f"(seed={cfg.seed})")
    print(f"  heights {min(heights):.3f}-{max(heights):.3f} m, "
          f"mean {sum(heights) / len(heights):.3f} m, "
          f"base coverage ~{100 * base_area / patch_area:.0f}%")

    cmd, v_eff = circle_command(args.radius, args.speed,
                                cycle_time=args.cycle_time, height=args.height)
    omega = v_eff / args.radius
    print(f"radius={args.radius:+.2f} m  v={v_eff:.3f} m/s (requested {args.speed:.3f})  "
          f"yaw rate={math.degrees(omega):+.2f} deg/s")

    gui = not args.direct
    p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(DT)

    # collision plane first, rugged pyramids on top of it
    p.loadURDF("plane.urdf")
    episode_path = task.reset_episode(load_terrain=True)
    print(f"episode {task.episode_index}: loaded {len(task.terrain_ids)} collidable "
          f"pyramids; generated {len(episode_path.points)}-sample smooth path")

    robot = p.loadURDF(args.urdf, [0, 0, abs(Z_DOWN) + 0.02], useFixedBase=False)
    joints = leg_joint_map(robot)
    for j in range(p.getNumJoints(robot)):
        p.setJointMotorControl2(robot, j, p.VELOCITY_CONTROL, force=0)

    command_legs(robot, joints, crouch_targets(Z_DOWN), reset=True)
    for _ in range(int(0.3 / DT)):
        step(gui)
    stand_up(robot, joints, gui)
    print("stand-up done, starting circle walk on terrain")

    gait = CrawlGait(cmd, cycle_provider=lambda n: cmd)

    pos0 = p.getBasePositionAndOrientation(robot)[0]
    yaw0 = yaw_of(robot)
    c0, s0 = math.cos(yaw0), math.sin(yaw0)

    def to_start_frame(pos):
        dx, dy = pos[0] - pos0[0], pos[1] - pos0[1]
        return c0 * dx + s0 * dy, -s0 * dx + c0 * dy

    duration = args.laps * 2 * math.pi * abs(args.radius) / v_eff
    steps = int(duration / DT)
    print(f"running {args.laps:g} lap(s) = {duration:.0f} s sim time")

    traj, next_log, next_print, fallen = [], 0.0, 0.0, False
    try:
        for _ in range(steps):
            command_legs(robot, joints, gait.update(DT))
            step(gui)

            if gait.t >= next_log:
                next_log += 0.1
                traj.append(to_start_frame(p.getBasePositionAndOrientation(robot)[0]))
            if gait.t >= next_print:
                next_print += 2.0
                pos = p.getBasePositionAndOrientation(robot)[0]
                observation = task.observe(robot)
                reward = task.reward(observation, terminated=pos[2] < 0.12)
                x, y = to_start_frame(pos)
                yaw = math.degrees(yaw_of(robot) - yaw0)
                print(f"  t={gait.t:5.1f}s  x={x:+.3f} y={y:+.3f} z={pos[2]:.3f}  "
                      f"yaw={yaw:+7.1f} deg (expected {math.degrees(omega * gait.t):+7.1f}) "
                      f"path-offset={float(observation['path_offset']):+.3f} m "
                      f"reward={reward['total']:.3f}")
                if pos[2] < 0.12:
                    print("  !! base very low / fallen, stopping")
                    fallen = True
                    break
    except KeyboardInterrupt:
        pass

    yaw_total = math.degrees(yaw_of(robot) - yaw0)
    print(f"\nyaw change (unwrapped by eye, wraps at +-180): {yaw_total:+.1f} deg, "
          f"expected {math.degrees(omega * gait.t):+.1f} deg")
    print(f"terrain survived: {'NO - robot fell' if fallen else 'yes'}")
    if len(traj) >= 10:
        cx, cy, r = fit_circle(np.array(traj))
        print(f"circle fit: radius={r:.3f} m (commanded {abs(args.radius):.3f}), "
              f"centre=({cx:+.3f}, {cy:+.3f}) m in start frame "
              f"(commanded (0, {args.radius:+.3f}))")
    if gui:
        try:
            input("\n[PyBullet GUI] Simulation finished. Press [Enter] in this terminal to close the window...")
        except (KeyboardInterrupt, EOFError):
            pass
    p.disconnect()


if __name__ == "__main__":
    main()
