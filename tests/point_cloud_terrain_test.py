#!/usr/bin/env python3
"""
Point Cloud Generation Test - Built on terrain_circlewalk test.

This test runs the A1 robot in a circle walk over random terrain while
continuously generating and visualizing colored point clouds from the
robot's depth camera.

Run from the project root:
  python tests/point_cloud_terrain_test.py --radius 0.6 --seed 7
  python tests/point_cloud_terrain_test.py --direct --laps 2
  python tests/point_cloud_terrain_test.py --save-ply --output-dir pointclouds
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

# Add project root to path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from surrounding_building.terrain_building import (
    RandomTerrainConfig, terrain_layout,
)
from path_training import PathTaskConfig, PathTrainingTask
from depth_processing.PointCloudGenerator import (
    PointCloudGenerator,
    PointCloudVisualizer,
)

# Same-folder test modules
from circle_wlk_test import circle_command, fit_circle
from IK_module_test import (
    DT, Z_DOWN, Z_STAND, CG_X,
    leg_joint_map, command_legs, crouch_targets, step, stand_up, yaw_of,
)
from Inverse_Kinematics.IK_A1 import LEG_NAMES, CrawlGait


def main():
    ap = argparse.ArgumentParser(
        description="Circle-walk the A1 over random terrain with point cloud generation"
    )
    # Circle-walk params
    ap.add_argument("--urdf", default="urdf/a1_unitree.urdf")
    ap.add_argument("--radius", type=float, default=0.6,
                    help="Signed circle radius [m]; + = left/CCW, - = right/CW")
    ap.add_argument("--speed", type=float, default=0.09, help="Body speed [m/s]")
    ap.add_argument("--cycle-time", type=float, default=1.0)
    ap.add_argument("--height", type=float, default=0.10,
                    help="Swing foot clearance [m]")
    ap.add_argument("--laps", type=float, default=1.0)
    ap.add_argument("--direct", action="store_true", help="Headless, no realtime")

    # Random terrain params
    g = ap.add_argument_group("random terrain")
    g.add_argument("--seed", type=int, default=7, help="RNG seed")
    g.add_argument("--extent", type=float, default=6.7, help="Terrain patch side [m]")
    g.add_argument("--spacing", type=float, default=0.24, help="Grid pitch [m]")
    g.add_argument("--size-min", dest="size_min", type=float, default=0.16)
    g.add_argument("--size-max", dest="size_max", type=float, default=0.34)
    g.add_argument("--height-min", dest="height_min", type=float, default=0.02)
    g.add_argument("--height-max", dest="height_max", type=float, default=0.08)
    g.add_argument("--density", type=float, default=1.0)
    g.add_argument("--jitter", type=float, default=0.05)
    g.add_argument("--clear-radius", dest="clear_radius", type=float, default=0.45)
    g.add_argument("--base-z", dest="base_z", type=float, default=0.0)

    # Point cloud params
    pc = ap.add_argument_group("point cloud")
    pc.add_argument("--pc-downsample", type=int, default=4,
                    help="Point cloud downsample factor")
    pc.add_argument("--pc-max-points", type=int, default=30000,
                    help="Max points per frame")
    pc.add_argument("--pc-depth-max", type=float, default=5.0,
                    help="Max depth [m]")
    pc.add_argument("--pc-width", type=int, default=640, help="Camera width")
    pc.add_argument("--pc-height", type=int, default=640, help="Camera height")
    pc.add_argument("--pc-fov", type=float, default=90.0, help="Camera FOV")
    pc.add_argument("--save-ply", action="store_true",
                    help="Save point clouds as PLY files")
    pc.add_argument("--output-dir", default="pointclouds",
                    help="Output directory for PLY files")
    pc.add_argument("--save-interval", type=float, default=1.0,
                    help="Save interval in seconds")

    args = ap.parse_args()

    if abs(args.radius) < 0.25:
        print("warning: radius < ~0.25 m will hit stride limits; expect heavy scaling")
    if args.height_max > args.height:
        print(f"warning: tallest pyramid ({args.height_max:.2f} m) exceeds the "
              f"swing clearance ({args.height:.2f} m); feet will clip obstacles")

    # Setup terrain config
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

    # Terrain stats
    layout = list(terrain_layout(cfg))
    heights = [h for _, _, _, h, _ in layout]
    base_area = sum(s * s * 0.5 for _, _, s, _, _ in layout)
    patch_area = args.extent ** 2 - math.pi * cfg.clear_radius ** 2
    print(f"terrain: {len(layout)} pyramids over {args.extent:.1f} m patch "
          f"(seed={cfg.seed})")
    print(f"  heights {min(heights):.3f}-{max(heights):.3f} m, "
          f"mean {sum(heights) / len(heights):.3f} m, "
          f"base coverage ~{100 * base_area / patch_area:.0f}%")

    # Circle command
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

    # Load terrain
    p.loadURDF("plane.urdf")
    episode_path = task.reset_episode(load_terrain=True)
    print(f"episode {task.episode_index}: loaded {len(task.terrain_ids)} collidable "
          f"pyramids; generated {len(episode_path.points)}-sample smooth path")

    # Load robot
    robot = p.loadURDF(args.urdf, [0, 0, abs(Z_DOWN) + 0.02], useFixedBase=False)
    joints = leg_joint_map(robot)
    for j in range(p.getNumJoints(robot)):
        p.setJointMotorControl2(robot, j, p.VELOCITY_CONTROL, force=0)

    command_legs(robot, joints, crouch_targets(Z_DOWN), reset=True)
    for _ in range(int(0.3 / DT)):
        step(gui)
    stand_up(robot, joints, gui)
    print("stand-up done, starting circle walk on terrain with point cloud")

    # Create point cloud generator
    pc_gen = PointCloudGenerator(
        robot_id=robot,
        camera_name="cam_L_joint",
        width=args.pc_width,
        height=args.pc_height,
        fov=args.pc_fov,
        near=0.05,
        far=10.0,
        downsample=args.pc_downsample,
    )

    # Create visualizer if not headless
    visualizer = None
    if not args.direct:
        visualizer = PointCloudVisualizer(
            window_name="A1 Terrain Point Cloud",
            point_size=1.5,
        )

    # Setup PLY saving
    if args.save_ply:
        import open3d as o3d
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving PLY files to {output_dir}")

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
    next_save = 0.0
    frame_count = 0
    total_points = 0

    try:
        for i in range(steps):
            command_legs(robot, joints, gait.update(DT))
            step(gui)

            # Capture point cloud
            points, colors, depths = pc_gen.capture_point_cloud(
                world_frame=True,
                max_points=args.pc_max_points,
                depth_threshold=args.pc_depth_max,
            )

            if len(points) > 0:
                total_points += len(points)
                frame_count += 1

                if visualizer is not None:
                    visualizer.update(points, colors)

                if args.save_ply and gait.t >= next_save:
                    next_save += args.save_interval
                    import open3d as o3d
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
                    filename = output_dir / f"pointcloud_t{gait.t:.1f}_frame{frame_count:04d}.ply"
                    o3d.io.write_point_cloud(str(filename), pcd)
                    print(f"  Saved {filename} ({len(points)} points)")

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
                      f"reward={reward['total']:.3f}  pc_frames={frame_count}")
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
    print(f"Point cloud frames captured: {frame_count}")
    print(f"Total points generated: {total_points}")
    if frame_count > 0:
        print(f"Average points per frame: {total_points / frame_count:.0f}")

    if len(traj) >= 10:
        cx, cy, r = fit_circle(np.array(traj))
        print(f"circle fit: radius={r:.3f} m (commanded {abs(args.radius):.3f}), "
              f"centre=({cx:+.3f}, {cy:+.3f}) m in start frame "
              f"(commanded (0, {args.radius:+.3f}))")

    if visualizer is not None:
        visualizer.close()
    pc_gen.close()
    p.disconnect()


if __name__ == "__main__":
    main()