"""PPO policy adapter that maps depth and robot state to crawl-gait commands.

Install ``stable-baselines3`` and ``torch`` (see requirements.txt) to train or
run a learned policy. Importing this module does not require those packages.
"""

from pathlib import Path

import numpy as np


DEPTH_WIDTH = 64
DEPTH_HEIGHT = 48
STATE_SIZE = 11
OBSERVATION_SIZE = DEPTH_WIDTH * DEPTH_HEIGHT + STATE_SIZE


def depth_observation(depth, robot_state, width=DEPTH_WIDTH, height=DEPTH_HEIGHT):
    """Resize metric depth and append normalized proprioceptive signals."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth image must be two-dimensional")
    try:
        import cv2
        resized = cv2.resize(depth, (width, height), interpolation=cv2.INTER_AREA)
    except ImportError:
        rows = np.linspace(0, depth.shape[0] - 1, height).astype(int)
        cols = np.linspace(0, depth.shape[1] - 1, width).astype(int)
        resized = depth[np.ix_(rows, cols)]

    state = np.asarray(robot_state, dtype=np.float32).reshape(-1)
    if state.size != STATE_SIZE:
        raise ValueError(f"robot_state must contain {STATE_SIZE} values")
    # Metric depth is clipped to the camera's 10 m range, then scaled to [0, 1].
    depth_feature = np.nan_to_num(resized, nan=10.0, posinf=10.0, neginf=0.0)
    depth_feature = np.clip(depth_feature, 0.0, 10.0) / 10.0
    return np.concatenate((depth_feature.reshape(-1), state)).astype(np.float32)


class DepthPPOController:
    """Load/train PPO over compact depth+state observations.

    Actions are normalized to [-1, 1] and map to forward stride, turn, swing
    height, and cycle speed, respectively.
    """

    def __init__(self, model_path="models/a1_depth_ppo", seed=7):
        self.model_path = Path(model_path)
        self.seed = seed
        self.model = None

    @staticmethod
    def _dependencies():
        try:
            import gymnasium as gym
            from gymnasium import spaces
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise RuntimeError(
                "DeepRL requires stable-baselines3 and torch. Install project "
                "requirements with: venv/bin/python -m pip install -r requirements.txt"
            ) from exc
        return gym, spaces, PPO

    @classmethod
    def make_policy_env(cls, seed=7):
        """Create a PyBullet rollout environment with real depth observations."""
        gym, spaces, _ = cls._dependencies()
        import pybullet as p
        import pybullet_data
        import math
        from depth_processing import DepthCamera
        from Inverse_Kinematics.IK_A1 import (
            LEG_NAMES, CrawlGait, CycleCommand, foot_to_joints,
        )
        from path_training import PathTaskConfig, PathTrainingTask

        dt = 1.0 / 120.0
        z_stand, z_down, cg_x = -0.22, -0.10, -0.085

        class PolicySpace(gym.Env):
            metadata = {"render_modes": []}

            def __init__(self):
                super().__init__()
                self.observation_space = spaces.Box(
                    low=0.0, high=1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32,
                )
                self.action_space = spaces.Box(
                    low=-1.0, high=1.0, shape=(4,), dtype=np.float32,
                )
                self.client = p.connect(p.DIRECT)
                if self.client < 0:
                    raise RuntimeError("Could not create the PyBullet training client")
                p.setAdditionalSearchPath(pybullet_data.getDataPath())
                p.setTimeStep(dt)
                p.setGravity(0, 0, -9.81)
                self.task = PathTrainingTask(
                    PathTaskConfig(path_length=2.5, path_width=0.8,
                                   waypoint_count=5, depth_width=DEPTH_WIDTH,
                                   depth_height=DEPTH_HEIGHT), seed=seed,
                )
                self.robot = None
                self.camera = None
                self.joints = None
                self.gait = None
                self.steps = 0
                self.max_steps = 1200
                self._last_observation = None

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                p.resetSimulation()
                p.setAdditionalSearchPath(pybullet_data.getDataPath())
                p.setGravity(0, 0, -9.81)
                p.setTimeStep(dt)
                p.loadURDF("plane.urdf")
                self.task.reset_episode(seed=seed, load_terrain=True)
                self.robot = p.loadURDF(
                    str(Path(__file__).resolve().parents[1] / "urdf" / "a1_unitree.urdf"),
                    [0, 0, 0.12], useFixedBase=False,
                )
                names = {
                    p.getJointInfo(self.robot, joint)[1].decode("utf-8"): joint
                    for joint in range(p.getNumJoints(self.robot))
                }
                self.joints = {
                    leg: tuple(names[f"{leg}_{part}_joint"]
                                for part in ("hip", "thigh", "calf"))
                    for leg in LEG_NAMES
                }
                for joint in range(p.getNumJoints(self.robot)):
                    p.setJointMotorControl2(self.robot, joint, p.VELOCITY_CONTROL, force=0)
                for leg in LEG_NAMES:
                    angles = foot_to_joints(leg, cg_x, 0, z_down)
                    for joint, angle in zip(self.joints[leg], angles):
                        p.resetJointState(self.robot, joint, angle)
                for step_index in range(int(0.5 / dt)):
                    blend = 0.5 * (1.0 - math.cos(
                        math.pi * step_index / max(int(0.5 / dt) - 1, 1)
                    ))
                    z = z_down + (z_stand - z_down) * blend
                    for leg in LEG_NAMES:
                        angles = foot_to_joints(leg, cg_x, 0, z)
                        for joint, angle in zip(self.joints[leg], angles):
                            p.setJointMotorControl2(
                                self.robot, joint, p.POSITION_CONTROL,
                                targetPosition=angle, force=33.5,
                            )
                    p.stepSimulation()
                self.camera = DepthCamera(
                    self.robot, width=DEPTH_WIDTH, height=DEPTH_HEIGHT,
                )
                self.gait = CrawlGait(CycleCommand.uniform(
                    length_x=0.08, height=0.10, stance_z=z_stand,
                    cycle_time=0.6, cg_offset_x=cg_x,
                ))
                self.steps = 0
                for _ in range(int(0.25 / dt)):
                    p.stepSimulation()
                self._last_task_observation, self._last_observation = self._observe()
                return self._last_observation.copy(), {}

            def step(self, action):
                action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
                command = cls.action_to_gait(
                    action, CycleCommand, stance_z=z_stand, cg_offset_x=cg_x,
                )
                self.gait.set_next_cycle(command)
                for _ in range(12):
                    targets = self.gait.update(dt)
                    for leg, angles in targets.items():
                        for joint, angle in zip(self.joints[leg], angles):
                            p.setJointMotorControl2(
                                self.robot, joint, p.POSITION_CONTROL,
                                targetPosition=angle, force=33.5,
                            )
                    p.stepSimulation()
                self.steps += 1
                task_observation, observation = self._observe()
                position = p.getBasePositionAndOrientation(self.robot)[0]
                terminated = bool(position[2] < 0.12 or not np.isfinite(position).all())
                truncated = self.steps >= self.max_steps
                reward_parts = self.task.reward(task_observation, terminated=terminated)
                reward = reward_parts["total"] - 0.01 * abs(float(action[1]))
                return observation, float(reward), terminated, truncated, reward_parts

            def _observe(self):
                task_observation = self.task.observe(self.robot, self.camera)
                pos, orn = p.getBasePositionAndOrientation(self.robot)
                linear, angular = p.getBaseVelocity(self.robot)
                roll, pitch, yaw = p.getEulerFromQuaternion(orn)
                state = np.asarray([
                    roll / math.pi, pitch / math.pi, math.sin(yaw), math.cos(yaw),
                    *np.clip(np.asarray(linear) / 2.0, -1.0, 1.0),
                    *np.clip(np.asarray(angular) / 4.0, -1.0, 1.0),
                    float(pos[2]) / 0.5,
                ], dtype=np.float32)
                vector = depth_observation(task_observation["depth"], state)
                return task_observation, vector

            def close(self):
                if p.isConnected(self.client):
                    p.disconnect(self.client)

        return PolicySpace()

    def load(self):
        _, _, ppo = self._dependencies()
        path = self.model_path
        if path.suffix != ".zip":
            path = Path(str(path) + ".zip")
        if not path.exists():
            return False
        self.model = ppo.load(str(path.with_suffix("")))
        return True

    def train(self, total_timesteps=100_000):
        """Train a PPO checkpoint; use a PyBullet rollout env for real learning."""
        _, _, ppo = self._dependencies()
        env = self.make_policy_env(self.seed)
        model = ppo("MlpPolicy", env, verbose=1, seed=self.seed,
                    learning_rate=3e-4, n_steps=256, batch_size=64,
                    gamma=0.99, ent_coef=0.01)
        try:
            model.learn(total_timesteps=int(total_timesteps))
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(self.model_path))
            self.model = model
        finally:
            env.close()
        return self.model_path

    def predict(self, observation):
        if self.model is None:
            raise RuntimeError("Load or train a PPO model before prediction")
        action, _ = self.model.predict(
            np.asarray(observation, dtype=np.float32), deterministic=True,
        )
        return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

    @staticmethod
    def action_to_gait(action, command_type, *, stance_z=-0.22, cg_offset_x=-0.085):
        """Convert PPO action [stride, turn, clearance, cadence] to gait command."""
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        # Keep useful forward motion even while PPO actions are near zero;
        # the policy adjusts a stable crawl baseline rather than selecting
        # from a range whose midpoint is stopped.
        stride = float(np.clip(0.10 + action[0] * 0.06, 0.04, 0.16))
        turn = float(action[1])
        height = float(np.clip(0.10 + action[2] * 0.04, 0.06, 0.14))
        cycle_time = float(np.clip(0.55 - action[3] * 0.20, 0.35, 0.75))
        return command_type.differential(
            length_x=stride, turn=turn, height=height,
            cycle_time=cycle_time, stance_z=stance_z,
            cg_offset_x=cg_offset_x,
        )
