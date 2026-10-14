"""Launch the A1 simulation with a live depth point-cloud UI.

Run from the project root with ``venv/bin/python main.py``.
"""

import math
import time

import numpy as np
import pybullet as p
import pybullet_data

from depth_processing import DepthCamera
from Inverse_Kinematics.IK_A1 import (
    LEG_NAMES,
    CycleCommand,
    CrawlGait,
    foot_to_joints,
)
from point_cloud import PointCloudAccumulator, PointCloudViewer, depth_to_point_cloud


DT = 1.0 / 120.0
MAX_FORCE = 33.5
Z_DOWN = -0.10
Z_STAND = -0.22
CG_X = -0.085

# Alternating waypoints steer the robot around the large blocks.
PATH = (
    (0.0, 0.0), (0.8, 0.0), (1.5, 1.0), (2.6, 1.0),
    (3.2, -1.0), (4.2, -1.0), (4.8, 1.0), (5.8, 1.0),
    (6.4, 0.0), (7.2, 0.0),
)


def add_arch(position, opening=1.05, height=1.5, depth=0.65, thickness=0.28,
             color=(0.82, 0.22, 0.12, 1)):
    """Create a three-piece arch as one colored, collidable PyBullet body."""
    pillar_half = [depth / 2, thickness / 2, (height - thickness) / 2]
    beam_half = [depth / 2, (opening + 2 * thickness) / 2, thickness / 2]
    pillar_y = opening / 2 + thickness / 2
    pillar_z = (height - thickness) / 2
    beam_z = height - thickness / 2
    shape_types = [p.GEOM_BOX, p.GEOM_BOX, p.GEOM_BOX]
    collision = p.createCollisionShapeArray(
        shapeTypes=shape_types,
        halfExtents=[pillar_half, pillar_half, beam_half],
        collisionFramePositions=[
            [0, -pillar_y, pillar_z],
            [0, pillar_y, pillar_z],
            [0, 0, beam_z],
        ],
    )
    visual = p.createVisualShapeArray(
        shapeTypes=shape_types,
        halfExtents=[pillar_half, pillar_half, beam_half],
        visualFramePositions=[
            [0, -pillar_y, pillar_z],
            [0, pillar_y, pillar_z],
            [0, 0, beam_z],
        ],
        rgbaColors=[color, color, color],
    )
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=[position[0], position[1], 0],
    )


def draw_path():
    for start, end in zip(PATH, PATH[1:]):
        p.addUserDebugLine(
            [start[0], start[1], 0.035],
            [end[0], end[1], 0.035],
            lineColorRGB=[0.1, 0.95, 0.25],
            lineWidth=5,
            lifeTime=0,
        )


def joint_map(robot_id):
    names = {
        p.getJointInfo(robot_id, joint)[1].decode("utf-8"): joint
        for joint in range(p.getNumJoints(robot_id))
    }
    return {
        leg: tuple(names[f"{leg}_{part}_joint"] for part in ("hip", "thigh", "calf"))
        for leg in LEG_NAMES
    }


def command_legs(robot_id, joints, targets):
    for leg, angles in targets.items():
        for joint, angle in zip(joints[leg], angles):
            p.setJointMotorControl2(
                robot_id,
                joint,
                p.POSITION_CONTROL,
                targetPosition=angle,
                force=MAX_FORCE,
            )


def stand_up(robot_id, joints):
    steps = int(1.0 / DT)
    for step_index in range(steps + 1):
        blend = 0.5 * (1 - math.cos(math.pi * step_index / steps))
        z = Z_DOWN + (Z_STAND - Z_DOWN) * blend
        targets = {
            leg: foot_to_joints(leg, CG_X, 0.0, z)
            for leg in LEG_NAMES
        }
        command_legs(robot_id, joints, targets)
        p.stepSimulation()
        time.sleep(DT)


def path_command(robot_id, path_index):
    """Choose a differential crawl command toward the next route waypoint."""
    position, orientation = p.getBasePositionAndOrientation(robot_id)
    if path_index == len(PATH) - 1 and math.dist(position[:2], PATH[-1]) < 0.65:
        return CycleCommand.uniform(
            length_x=0.0,
            height=0.02,
            stance_z=Z_STAND,
            cg_offset_x=CG_X,
        )
    yaw = p.getEulerFromQuaternion(orientation)[2]
    target = PATH[path_index]
    dx, dy = target[0] - position[0], target[1] - position[1]
    heading_error = math.atan2(dy, dx) - yaw
    heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
    # Use the full differential range at sharp bends so heading corrections
    # are visible in the crawl instead of letting the robot continue straight.
    turn = max(-1.0, min(1.0, heading_error / 0.55))
    # Keep the longest stride on straights. During a turn, cap the outside
    # legs' stride at the straight-line stride to avoid overreaching.
    stride = 0.16 / (1.0 + abs(turn))
    return CycleCommand.differential(
        length_x=stride,
        turn=turn,
        height=0.14,
        cycle_time=0.4,
        stance_z=Z_STAND,
        cg_offset_x=CG_X,
    )


def reached_or_passed(position, waypoint_index):
    """Avoid getting stuck when momentum carries the robot past a waypoint."""
    target = PATH[waypoint_index]
    if math.dist(position[:2], target) < 0.55:
        return True

    start = PATH[waypoint_index - 1]
    segment_x = target[0] - start[0]
    segment_y = target[1] - start[1]
    segment_length = math.hypot(segment_x, segment_y)
    if segment_length == 0:
        return False
    relative_x = position[0] - start[0]
    relative_y = position[1] - start[1]
    along = (relative_x * segment_x + relative_y * segment_y) / segment_length
    cross_track = abs(relative_x * segment_y - relative_y * segment_x) / segment_length
    return along >= segment_length and cross_track < 0.8


def main():
    client = p.connect(p.GUI)
    if client < 0:
        raise RuntimeError("Could not open the PyBullet GUI")

    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(DT)
        p.loadURDF("plane.urdf")
        # Collidable arches stand ahead along the route; each is one visual
        # object, so its complete point-cloud scan receives one color.
        add_arch((2.0, 0.0), color=(0.82, 0.22, 0.12, 1))
        add_arch((3.7, 0.7), color=(0.18, 0.42, 0.88, 1))
        add_arch((5.3, -0.7), color=(0.92, 0.58, 0.12, 1))
        draw_path()
        robot_id = p.loadURDF(
            "urdf/a1_unitree.urdf",
            basePosition=[0, 0, 0.12],
            useFixedBase=False,
        )

        joints = joint_map(robot_id)
        for joint in range(p.getNumJoints(robot_id)):
            p.setJointMotorControl2(robot_id, joint, p.VELOCITY_CONTROL, force=0)
        for leg in LEG_NAMES:
            crouch = foot_to_joints(leg, CG_X, 0, Z_DOWN)
            for joint, angle in zip(joints[leg], crouch):
                p.resetJointState(robot_id, joint, angle)
        for _ in range(int(0.2 / DT)):
            p.stepSimulation()
            time.sleep(DT)
        stand_up(robot_id, joints)

        waypoint = 1
        gait = CrawlGait(path_command(robot_id, waypoint))
        gait.cycle_provider = lambda _cycle: path_command(robot_id, waypoint)
        camera = DepthCamera(robot_id)
        cloud = PointCloudAccumulator()
        last_cloud_update = 0.0
        scan_started = time.monotonic()
        final_reached_at = None
        scan_timeout = 120.0
        while p.isConnected(client):
            # Advance the look-ahead waypoint once the robot reaches it.
            position = p.getBasePositionAndOrientation(robot_id)[0]
            while waypoint < len(PATH) - 1 and reached_or_passed(position, waypoint):
                waypoint += 1
                gait.set_next_cycle(path_command(robot_id, waypoint))
            at_final_waypoint = (
                waypoint == len(PATH) - 1
                and math.dist(position[:2], PATH[-1]) < 0.65
            )
            if at_final_waypoint:
                if final_reached_at is None:
                    final_reached_at = time.monotonic()
            else:
                final_reached_at = None

            command_legs(robot_id, joints, gait.update(DT))
            p.stepSimulation()
            now = time.monotonic()
            if now - last_cloud_update >= 0.1:
                depth, segmentation = camera.capture_frame()
                points, colors = depth_to_point_cloud(
                    depth, camera, stride=6, segmentation=segmentation,
                )
                cloud.add(points, colors)
                last_cloud_update = now
            if (
                (final_reached_at is not None and now - final_reached_at >= 2.0)
                or now - scan_started >= scan_timeout
            ):
                break
            time.sleep(DT)
        final_points = cloud.points()
        final_colors = cloud.colors()
        if len(final_points) > 2000:
            display_indices = np.linspace(
                0, len(final_points) - 1, 2000, dtype=int,
            )
            final_points = final_points[display_indices]
            final_colors = final_colors[display_indices]
        print(f"Scan complete: displaying {len(final_points)} points")
        viewer = PointCloudViewer()
        viewer.update(final_points, final_colors)
        viewer.show()
    except KeyboardInterrupt:
        pass
    finally:
        if p.isConnected(client):
            p.disconnect(client)


if __name__ == "__main__":
    main()
