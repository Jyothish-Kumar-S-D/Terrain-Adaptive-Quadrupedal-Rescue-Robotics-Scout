"""Launch the A1 simulation with a live depth point-cloud UI.

Run from the project root with ``venv/bin/python main.py``.
"""

import math
import time
import signal
import sys
from pathlib import Path

import cv2
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
from deep_rl import DepthPPOController, depth_observation
from deep_rl.depth_dataset import DepthFrameStore
from path_training import PathTaskConfig, PathTrainingTask
from surrounding_building.terrain_building import RandomTerrainConfig


class VideoRecorder:
    """Records simulation frames and saves as video."""

    def __init__(self, output_path, width=1280, height=720, fps=30):
        self.output_path = Path(output_path)
        self.width = width
        self.height = height
        self.fps = fps
        self.writer = None
        self.frame_count = 0

    def start(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            str(self.output_path), fourcc, self.fps, (self.width, self.height)
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {self.output_path}")

    def capture_frame(self, robot_id, orbit_radius=2.0, orbit_height=0.5, orbit_speed=0.2):
        """Capture a frame from an orbiting camera around the robot."""
        pos, _ = p.getBasePositionAndOrientation(robot_id)
        robot_x, robot_y, robot_z = pos

        # Orbit angle based on time
        angle = time.time() * orbit_speed
        cam_x = robot_x + orbit_radius * math.cos(angle)
        cam_y = robot_y + orbit_radius * math.sin(angle)
        cam_z = robot_z + orbit_height

        # Look at robot
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[cam_x, cam_y, cam_z],
            cameraTargetPosition=[robot_x, robot_y, robot_z],
            cameraUpVector=[0, 0, 1],
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=self.width / self.height, nearVal=0.1, farVal=100
        )
        _, _, rgb, _, _ = p.getCameraImage(
            width=self.width, height=self.height,
            viewMatrix=view_matrix, projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        # Convert BGRA to BGR for OpenCV
        frame = np.array(rgb, dtype=np.uint8).reshape(self.height, self.width, 4)
        frame = frame[:, :, :3]  # Drop alpha
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def write_frame(self, frame):
        if self.writer is not None:
            self.writer.write(frame)
            self.frame_count += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print(f"Video saved: {self.output_path} ({self.frame_count} frames)")


DT = 1.0 / 120.0
MAX_FORCE = 33.5
Z_DOWN = -0.10
Z_STAND = -0.22
CG_X = -0.085
PROJECT_DIR = Path(__file__).resolve().parent
DEPTH_FRAME_DIR = PROJECT_DIR / "depth_frames"
PPO_MODEL_PATH = PROJECT_DIR / "models" / "a1_depth_ppo.best"

# Global for signal handler
video_recorder_global = None

def add_arch(position, opening=1.05, height=1.5, depth=0.65, thickness=0.28,
             color=(0.82, 0.22, 0.12, 1), yaw=0.0):
    """Create a three-piece arch as one colored, collidable PyBullet body.

    Args:
        position: (x, y) base position
        yaw: rotation around z-axis in radians (aligns arch opening with path)
    """
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
    orn = p.getQuaternionFromEuler([0, 0, yaw])
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=[position[0], position[1], 0],
        baseOrientation=orn,
    )


def draw_path(path):
    points = path.points
    # Debug drawing every few samples keeps a smooth-looking route without
    # creating hundreds of line segments in the GUI.
    for start, end in zip(points[::8], points[8::8]):
        p.addUserDebugLine(
            [float(start[0]), float(start[1]), 0.035],
            [float(end[0]), float(end[1]), 0.035],
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


def path_command(robot_id, path, path_index):
    """Choose a differential crawl command toward a look-ahead route point."""
    position, orientation = p.getBasePositionAndOrientation(robot_id)
    points = path.points
    if path_index == len(points) - 1 and math.dist(position[:2], points[-1]) < 0.45:
        return CycleCommand.uniform(
            length_x=0.0,
            height=0.02,
            stance_z=Z_STAND,
            cg_offset_x=CG_X,
        )
    yaw = p.getEulerFromQuaternion(orientation)[2]
    # Aim ahead on the sampled curve so bends are followed smoothly.
    target = points[min(path_index + 12, len(points) - 1)]
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


def _command_turn(command):
    """Recover the signed differential turn from left/right leg strides."""
    right_stride = command.legs["FR"].length_x
    left_stride = command.legs["FL"].length_x
    average = right_stride + left_stride
    if abs(average) < 1e-9:
        return 0.0
    return max(-1.0, min(1.0, (right_stride - left_stride) / average))


def policy_state(robot_id, path=None):
    """Compact state paired with depth for the PPO policy."""
    position, quaternion = p.getBasePositionAndOrientation(robot_id)
    linear_velocity, angular_velocity = p.getBaseVelocity(robot_id)
    roll, pitch, yaw = p.getEulerFromQuaternion(quaternion)
    route_features = (0.0, 0.0, 0.0, 0.0)
    if path is not None:
        _, distance, lateral_offset, tangent = path.nearest(position[:2])
        route_features = (
            float(tangent[0]), float(tangent[1]),
            float(np.clip(lateral_offset / 2.0, -1.0, 1.0)),
            float(np.clip(distance / 2.0, 0.0, 1.0)),
        )
    return np.asarray([
        roll / math.pi, pitch / math.pi, math.sin(yaw), math.cos(yaw),
        *np.clip(np.asarray(linear_velocity) / 2.0, -1.0, 1.0),
        *np.clip(np.asarray(angular_velocity) / 4.0, -1.0, 1.0),
        float(position[2]) / 0.5,
        *route_features,
    ], dtype=np.float32)


def reached_or_passed(position, path, waypoint_index):
    """Move the route tracker forward to the nearest nearby sample."""
    points = path.points
    search_end = min(waypoint_index + 32, len(points))
    delta = points[waypoint_index:search_end] - np.asarray(position[:2])
    nearest = waypoint_index + int(np.argmin(np.einsum("ij,ij->i", delta, delta)))
    return max(waypoint_index, nearest)


def main():
    global video_recorder_global
    video_recorder_global = None

    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}. Cleaning up...")
        global video_recorder_global
        if video_recorder_global is not None:
            try:
                video_recorder_global.close()
            except Exception:
                pass
        if p.isConnected():
            p.disconnect()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    client = p.connect(p.GUI)
    if client < 0:
        raise RuntimeError("Could not open the PyBullet GUI")

    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(DT)
        p.loadURDF("plane.urdf")
        # A fresh route and obstacle field are sampled on every launch.
        # Extend terrain ahead of the full route while keeping the start clear.
        task = PathTrainingTask(PathTaskConfig(
            path_length=7.2,
            path_width=2.4,
            waypoint_count=10,
            path_samples=360,
            terrain=RandomTerrainConfig(
                extent=9.0,
                spacing=0.20,
                size=(0.08, 0.18),
                height=(0.015, 0.06),
                density=0.82,
                jitter=0.04,
                clear_radius=0.75,
            ),
        ))
        route = task.reset_episode(load_terrain=True)
        draw_path(route)
        # Add 2 gates along the path for visual guidance, aligned with path tangent.
        for fraction in (0.33, 0.66):
            gate_index = int(fraction * (len(route.points) - 1))
            gate_x, gate_y = route.points[gate_index]
            # Get tangent at gate position to align arch with path direction.
            tangent = route.tangents[gate_index]
            gate_yaw = math.atan2(tangent[1], tangent[0])
            add_arch((gate_x, gate_y), yaw=gate_yaw)
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
        fallback_gait = lambda: path_command(robot_id, route, waypoint)
        controller = DepthPPOController(PPO_MODEL_PATH)
        policy_loaded = False
        try:
            policy_loaded = controller.load()
        except RuntimeError as error:
            print(f"DeepRL unavailable ({error}); using analytic crawl gait")
        if policy_loaded:
            print(f"Loaded depth PPO policy: {PPO_MODEL_PATH}")
        else:
            print("No trained PPO checkpoint found; using analytic crawl gait")
        gait = CrawlGait(fallback_gait())
        gait.cycle_provider = lambda _cycle: fallback_gait()
        camera = DepthCamera(robot_id)
        frame_store = DepthFrameStore(DEPTH_FRAME_DIR)
        cloud = PointCloudAccumulator()

        # Point cloud save directory
        cloud_dir = PROJECT_DIR / "point_clouds"
        cloud_dir.mkdir(exist_ok=True)
        cloud_path = cloud_dir / f"pointcloud_{time.strftime('%Y%m%d_%H%M%S')}.npz"
        last_cloud_save = 0.0
        cloud_save_interval = 30.0  # Save every 30 seconds

        # Video recorder for orbiting camera
        video_dir = PROJECT_DIR / "videos"
        video_dir.mkdir(exist_ok=True)
        video_path = video_dir / f"simulation_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        video_recorder = VideoRecorder(video_path, width=1280, height=720, fps=30)
        video_recorder.start()
        video_recorder_global = video_recorder

        last_cloud_update = 0.0
        last_policy_update = 0.0
        policy_update_interval = 0.1
        latest_depth = None
        policy_action = None
        print("PyBullet GUI is running. Close its window or press Ctrl+C to stop.")
        while p.isConnected(client):
            # Advance the look-ahead waypoint once the robot reaches it.
            position = p.getBasePositionAndOrientation(robot_id)[0]
            next_waypoint = reached_or_passed(position, route, waypoint)
            if next_waypoint > waypoint:
                waypoint = next_waypoint
                gait.set_next_cycle(path_command(robot_id, route, waypoint))
            if (
                policy_loaded
                and latest_depth is not None
                and time.monotonic() - last_policy_update >= policy_update_interval
            ):
                state = policy_state(robot_id, route)
                observation = depth_observation(latest_depth, state)
                policy_action = controller.predict(observation)
                last_policy_update = time.monotonic()
                policy_command = controller.action_to_gait(
                    policy_action, CycleCommand,
                    stance_z=Z_STAND, cg_offset_x=CG_X,
                )
                # The route planner supplies heading correction; PPO selects
                # stride, clearance, and cadence from depth and body state.
                route_command = fallback_gait()
                policy_command = CycleCommand.differential(
                    length_x=min(0.16, max(0.02, policy_command.legs["FR"].length_x)),
                    turn=_command_turn(route_command),
                    height=policy_command.legs["FR"].height,
                    cycle_time=policy_command.cycle_time,
                    stance_z=Z_STAND,
                    cg_offset_x=CG_X,
                )
                gait.set_next_cycle(policy_command)
            command_legs(robot_id, joints, gait.update(DT))
            p.stepSimulation()

            # Record video frame
            frame = video_recorder.capture_frame(robot_id)
            video_recorder.write_frame(frame)

            now = time.monotonic()
            if now - last_cloud_update >= 0.1:
                depth, segmentation = camera.capture_frame()
                latest_depth = depth
                frame_store.save(depth, policy_state(robot_id, route))
                points, colors = depth_to_point_cloud(
                    depth, camera, stride=6, segmentation=segmentation,
                )
                cloud.add(points, colors)
                last_cloud_update = now

            # Periodically save point cloud to disk
            if now - last_cloud_save >= cloud_save_interval:
                cloud.save(cloud_path)
                last_cloud_save = now

            time.sleep(DT)

    except KeyboardInterrupt:
        print("\nInterrupted. Saving video and point cloud...")
    finally:
        # Always close video recorder
        try:
            video_recorder.close()
        except Exception as e:
            print(f"Warning: Could not close video recorder: {e}")

        # Save final point cloud
        try:
            cloud.save(cloud_path)
            print(f"Final point cloud saved: {cloud_path}")
        except Exception as e:
            print(f"Warning: Could not save point cloud: {e}")

        # Generate and display point cloud from accumulated data
        try:
            print("Generating point cloud from accumulated frames...")
            final_points = cloud.points()
            final_colors = cloud.colors()
            if len(final_points) > 0:
                if len(final_points) > 5000:
                    display_indices = np.linspace(
                        0, len(final_points) - 1, 5000, dtype=int,
                    )
                    final_points = final_points[display_indices]
                    final_colors = final_colors[display_indices]
                print(f"Scan complete: displaying {len(final_points)} points")
                viewer = PointCloudViewer()
                viewer.update(final_points, final_colors)
                viewer.show()
            else:
                print("No point cloud data accumulated.")
        except Exception as e:
            print(f"Warning: Could not generate point cloud: {e}")

        if p.isConnected(client):
            p.disconnect(client)


if __name__ == "__main__":
    main()
