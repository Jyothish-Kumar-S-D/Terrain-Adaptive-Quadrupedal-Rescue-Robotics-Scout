#!/usr/bin/env python3
"""
Interactive point cloud visualization script for the A1 robot.

This script continuously captures depth and RGB from the robot's camera
and generates a colored 3D point cloud that updates in real-time.

Usage:
    python scripts/point_cloud_visualizer.py [--urdf URDF_PATH] [--downsample N]
"""

import argparse
import time
import sys
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

# Add project root to path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from depth_processing.PointCloudGenerator import (
    PointCloudGenerator,
    PointCloudVisualizer,
)


def main():
    ap = argparse.ArgumentParser(
        description="Interactive point cloud visualization for A1 robot"
    )
    ap.add_argument(
        "--urdf",
        default="urdf/a1_unitree.urdf",
        help="Path to A1 URDF file",
    )
    ap.add_argument(
        "--downsample",
        type=int,
        default=2,
        help="Downsample factor for point cloud (1 = full resolution)",
    )
    ap.add_argument(
        "--max-points",
        type=int,
        default=50000,
        help="Maximum points to display",
    )
    ap.add_argument(
        "--depth-max",
        type=float,
        default=5.0,
        help="Maximum depth to include (meters)",
    )
    ap.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera image width",
    )
    ap.add_argument(
        "--height",
        type=int,
        default=640,
        help="Camera image height",
    )
    ap.add_argument(
        "--fov",
        type=float,
        default=90.0,
        help="Camera field of view (degrees)",
    )
    ap.add_argument(
        "--direct",
        action="store_true",
        help="Run headless (no GUI)",
    )
    ap.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Run duration in seconds (0 = infinite)",
    )
    args = ap.parse_args()

    gui = not args.direct
    p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)

    # Load ground plane
    p.loadURDF("plane.urdf")

    # Add some objects for the camera to see
    p.loadURDF("cube.urdf", [1.5, 0, 0.25], globalScaling=0.5)
    p.loadURDF("sphere.urdf", [-1.0, 1.0, 0.2], globalScaling=0.3)
    p.loadURDF("cylinder.urdf", [0, -1.5, 0.3], globalScaling=0.4)

    # Load robot
    robot = p.loadURDF(
        args.urdf,
        [0, 0, 0.5],
        useFixedBase=False,
    )

    # Disable default velocity motors
    for j in range(p.getNumJoints(robot)):
        p.setJointMotorControl2(robot, j, p.VELOCITY_CONTROL, force=0)

    # Create point cloud generator
    pc_gen = PointCloudGenerator(
        robot_id=robot,
        camera_name="cam_L_joint",
        width=args.width,
        height=args.height,
        fov=args.fov,
        near=0.05,
        far=10.0,
        downsample=args.downsample,
    )

    # Create visualizer
    visualizer = PointCloudVisualizer(
        window_name="A1 Robot Point Cloud",
        point_size=2.0,
    )

    print("Starting point cloud visualization...")
    print("Press Ctrl+C to stop")

    start_time = time.time()
    frame_count = 0

    try:
        while True:
            p.stepSimulation()

            points, colors, depths = pc_gen.capture_point_cloud(
                world_frame=True,
                max_points=args.max_points,
                depth_threshold=args.depth_max,
            )

            if len(points) > 0:
                visualizer.update(points, colors)
                frame_count += 1

                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    info = pc_gen.get_camera_info()
                    print(f"Frame {frame_count}: {len(points)} points, "
                          f"{fps:.1f} FPS, "
                          f"Camera pos: [{info['position'][0]:.2f}, "
                          f"{info['position'][1]:.2f}, {info['position'][2]:.2f}]")

            if args.duration > 0 and (time.time() - start_time) > args.duration:
                break

            if gui:
                time.sleep(1.0 / 240.0)

    except KeyboardInterrupt:
        print("\nStopped by user")

    finally:
        elapsed = time.time() - start_time
        if elapsed > 0:
            print(f"\nTotal frames: {frame_count}, Average FPS: {frame_count/elapsed:.1f}")
        visualizer.close()
        pc_gen.close()
        p.disconnect()


if __name__ == "__main__":
    main()