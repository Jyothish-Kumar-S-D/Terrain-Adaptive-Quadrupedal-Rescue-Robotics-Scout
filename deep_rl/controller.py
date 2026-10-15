"""PPO policy adapter that maps depth and robot state to crawl-gait commands.

Install ``stable-baselines3`` and ``torch`` (see requirements.txt) to train or
run a learned policy. Importing this module does not require those packages.
"""

from pathlib import Path

import numpy as np


DEPTH_WIDTH = 64
DEPTH_HEIGHT = 48
STATE_SIZE = 15
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

        def add_training_arch(x, y, opening=0.9, height=0.75,
                             depth=0.35, thickness=0.16, yaw=0.0):
            """Add a low collidable gate centered on the episode route, aligned with path."""
            pillar_half = [depth / 2, thickness / 2, height / 2]
            beam_half = [depth / 2, (opening + 2 * thickness) / 2,
                         thickness / 2]
            pillar_y = opening / 2 + thickness / 2
            collision = p.createCollisionShapeArray(
                shapeTypes=[p.GEOM_BOX] * 3,
                halfExtents=[pillar_half, pillar_half, beam_half],
                collisionFramePositions=[
                    [0, -pillar_y, height / 2],
                    [0, pillar_y, height / 2],
                    [0, 0, height],
                ],
            )
            visual = p.createVisualShapeArray(
                shapeTypes=[p.GEOM_BOX] * 3,
                halfExtents=[pillar_half, pillar_half, beam_half],
                visualFramePositions=[
                    [0, -pillar_y, height / 2],
                    [0, pillar_y, height / 2],
                    [0, 0, height],
                ],
                rgbaColors=[[0.82, 0.22, 0.12, 1.0]] * 3,
            )
            orn = p.getQuaternionFromEuler([0, 0, yaw])
            return p.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual, basePosition=[x, y, 0],
                baseOrientation=orn,
            )

        class PolicySpace(gym.Env):
            metadata = {"render_modes": []}

            def __init__(self, seed=7):
                super().__init__()
                self._seed = seed
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(OBSERVATION_SIZE,),
                    dtype=np.float32,
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
                    PathTaskConfig(
                        path_length=4.0, path_width=1.2,
                        waypoint_count=7, path_samples=280,
                        depth_width=DEPTH_WIDTH, depth_height=DEPTH_HEIGHT,
                        path_tolerance=0.25,
                    ), seed=seed,
                )
                self.robot = None
                self.camera = None
                self.joints = None
                self.gait = None
                self.steps = 0
                self.max_steps = 1200
                self._last_observation = None
                self._path_debug_ids = []
                self._fall_reward = -10.0
                self._living_penalty = -0.005
                self._off_path_reward = -0.3

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                p.resetSimulation()
                p.setAdditionalSearchPath(pybullet_data.getDataPath())
                p.setGravity(0, 0, -9.81)
                p.setTimeStep(dt)
                p.loadURDF("plane.urdf")
                route = self.task.reset_episode(seed=seed, load_terrain=True)
                self._draw_path(route)
                # Put gates at randomized positions along the path with jitter.
                # Number of gates varies (2-4) and positions are randomized.
                n_gates = 3
                rng = np.random.default_rng(self._seed + self.task.episode_index + 12345)
                gate_fractions = rng.uniform(0.15, 0.85, n_gates)
                gate_fractions.sort()
                for frac in gate_fractions:
                    gate_index = int(frac * (len(route.points) - 1))
                    gate_x, gate_y = route.points[gate_index]
                    # Add lateral jitter to gate position.
                    tangent = route.tangents[gate_index]
                    normal = np.array([-tangent[1], tangent[0]])
                    lateral_jitter = rng.uniform(-0.5, 0.5)
                    gate_x += normal[0] * lateral_jitter
                    gate_y += normal[1] * lateral_jitter
                    # Align arch with path tangent so robot passes through along path.
                    gate_yaw = math.atan2(tangent[1], tangent[0])
                    add_training_arch(float(gate_x), float(gate_y), yaw=gate_yaw)
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
                orientation = p.getBasePositionAndOrientation(self.robot)[1]
                roll, pitch, yaw = p.getEulerFromQuaternion(orientation)
                path_distance = float(task_observation["path_distance"])
                too_far_from_route = path_distance > self.task.config.off_path_termination_distance
                # Detect falls in any direction: low height, excessive tilt, or invalid position.
                height_fall = position[2] < 0.12
                tilt_fall = abs(roll) > 0.7 or abs(pitch) > 0.7  # ~40 degrees
                invalid_pos = not np.isfinite(position).all()
                has_fallen = height_fall or tilt_fall or invalid_pos
                terminated = bool(has_fallen or too_far_from_route)
                truncated = self.steps >= self.max_steps
                reward_parts = self.task.reward(task_observation, terminated=terminated)
                reward = reward_parts["total"] - 0.01 * abs(float(action[1]))
                reward += self._living_penalty
                if has_fallen:
                    reward += self._fall_reward
                    reward_parts["fall_penalty"] = self._fall_reward
                if too_far_from_route and not has_fallen:
                    reward += self._off_path_reward
                    reward_parts["off_path_reward"] = self._off_path_reward
                return observation, float(reward), terminated, truncated, reward_parts

            def _draw_path(self, route):
                """Visualize the training path in PyBullet GUI."""
                for debug_id in self._path_debug_ids:
                    try:
                        p.removeUserDebugItem(debug_id)
                    except Exception:
                        pass
                self._path_debug_ids.clear()
                points = route.points
                for start, end in zip(points[::8], points[8::8]):
                    debug_id = p.addUserDebugLine(
                        [float(start[0]), float(start[1]), 0.035],
                        [float(end[0]), float(end[1]), 0.035],
                        lineColorRGB=[0.1, 0.95, 0.25],
                        lineWidth=5,
                        lifeTime=0,
                    )
                    self._path_debug_ids.append(debug_id)

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
                    float(task_observation["path_tangent"][0]),
                    float(task_observation["path_tangent"][1]),
                    float(np.clip(task_observation["path_offset"] / 2.0, -1.0, 1.0)),
                    float(np.clip(task_observation["path_distance"] / 2.0, 0.0, 1.0)),
                ], dtype=np.float32)
                vector = depth_observation(task_observation["depth"], state)
                return task_observation, vector

            def close(self):
                if p.isConnected(self.client):
                    p.disconnect(self.client)

        return PolicySpace(seed=seed)

    def load(self):
        _, _, ppo = self._dependencies()
        path = self.model_path
        if path.suffix != ".zip":
            path = Path(str(path) + ".zip")
        if not path.exists():
            return False
        model = ppo.load(str(path.with_suffix("")))
        model_shape = tuple(model.observation_space.shape)
        if model_shape != (OBSERVATION_SIZE,):
            raise RuntimeError(
                f"Checkpoint expects observations {model_shape}; this route-aware "
                f"environment uses {(OBSERVATION_SIZE,)}. Retrain the policy."
            )
        self.model = model
        return True

    def _make_custom_policy(self):
        """Create a custom policy with CNN for depth + MLP for state."""
        try:
            from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
            import torch as th
            import torch.nn as nn
            from gymnasium import spaces
        except ImportError:
            return "MlpPolicy"

        class DepthStateExtractor(BaseFeaturesExtractor):
            def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
                super().__init__(observation_space, features_dim)
                depth_size = DEPTH_WIDTH * DEPTH_HEIGHT
                state_size = STATE_SIZE

                self.cnn = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=2),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Flatten(),
                )

                with th.no_grad():
                    sample = th.zeros(1, 1, DEPTH_HEIGHT, DEPTH_WIDTH)
                    cnn_out = self.cnn(sample).shape[1]

                self.state_net = nn.Sequential(
                    nn.Linear(state_size, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                )

                combined = cnn_out + 128
                self.fusion = nn.Sequential(
                    nn.Linear(combined, 512),
                    nn.ReLU(),
                    nn.Linear(512, features_dim),
                    nn.ReLU(),
                )

            def forward(self, obs: th.Tensor) -> th.Tensor:
                depth_size = DEPTH_WIDTH * DEPTH_HEIGHT
                depth = obs[:, :depth_size].view(-1, 1, DEPTH_HEIGHT, DEPTH_WIDTH)
                state = obs[:, depth_size:]
                depth_feat = self.cnn(depth)
                state_feat = self.state_net(state)
                return self.fusion(th.cat([depth_feat, state_feat], dim=1))

        from stable_baselines3.ppo import MlpPolicy
        class CustomPolicy(MlpPolicy):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        return lambda *args, **kwargs: CustomPolicy(
            *args,
            features_extractor_class=DepthStateExtractor,
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
            activation_fn=nn.ReLU,
            **kwargs,
        )

    def train(self, total_timesteps=100_000, initial_lr=3e-4, final_lr=3e-5,
                n_steps=128, batch_size=64, gamma=0.99, ent_coef=0.01):
        """Train a PPO checkpoint; use a PyBullet rollout env for real learning."""
        _, _, ppo = self._dependencies()
        env = self.make_policy_env(self.seed)
        lr_schedule = lambda progress: initial_lr + (final_lr - initial_lr) * progress
        policy = self._make_custom_policy()
        model = ppo(policy, env, verbose=1, seed=self.seed,
                    learning_rate=lr_schedule, n_steps=n_steps, batch_size=batch_size,
                    gamma=gamma, ent_coef=ent_coef)

        best_reward = -float('inf')
        best_model_path = self.model_path.with_suffix('.best.zip')

        class BestModelCallback:
            def __init__(self, model, best_model_path):
                self.model = model
                self.best_model_path = best_model_path
                self.best_score = -float('inf')
                self.episode_rewards = []
                self.episode_lengths = []
                self.losses = []

            def __call__(self, locals_, globals_):
                # Track episode metrics from infos
                if 'infos' in locals_:
                    for info in locals_['infos']:
                        if 'episode' in info:
                            ep_reward = info['episode']['r']
                            ep_len = info['episode']['l']
                            self.episode_rewards.append(ep_reward)
                            self.episode_lengths.append(ep_len)

                # Track losses from logger
                if 'losses' in locals_:
                    self.losses.append(locals_['losses'])

                # Compute composite score every 10 episodes
                if len(self.episode_rewards) % 10 == 0 and len(self.episode_rewards) > 0:
                    score = self._compute_score()
                    if score > self.best_score:
                        self.best_score = score
                        self.model.save(str(self.best_model_path))
                        print(f"New best model saved: reward={np.mean(self.episode_rewards[-10:]):.2f}, "
                              f"len={np.mean(self.episode_lengths[-10:]):.1f}, "
                              f"score={score:.4f}")
                return True

            def _compute_score(self):
                """Composite score: reward + normalized length - normalized loss."""
                recent_rewards = np.array(self.episode_rewards[-10:])
                recent_lengths = np.array(self.episode_lengths[-10:])

                # Normalize reward (higher is better)
                reward_score = np.mean(recent_rewards)

                # Normalize length (higher is better, cap at max_steps)
                max_steps = 1200
                length_score = np.mean(recent_lengths) / max_steps

                # Loss penalty (lower is better)
                loss_score = 0.0
                if self.losses:
                    recent_losses = self.losses[-5:]
                    avg_policy_loss = np.mean([l.get('policy_loss', 0) for l in recent_losses])
                    avg_value_loss = np.mean([l.get('value_loss', 0) for l in recent_losses])
                    loss_score = -(avg_policy_loss + avg_value_loss) * 100  # Scale down

                # Weighted combination
                return 1.0 * reward_score + 0.5 * length_score + 0.3 * loss_score

        callback = BestModelCallback(model, best_model_path)
        try:
            model.learn(total_timesteps=int(total_timesteps), callback=callback)
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(self.model_path))
            self.model = model
            if callback.episode_rewards:
                print(f"Training complete. Best composite score: {callback.best_score:.4f}")
                print(f"  Avg reward (last 10): {np.mean(callback.episode_rewards[-10:]):.2f}")
                print(f"  Avg episode length (last 10): {np.mean(callback.episode_lengths[-10:]):.1f}")
                if callback.losses:
                    recent = callback.losses[-5:]
                    pl = np.mean([l.get('policy_loss', 0) for l in recent])
                    vl = np.mean([l.get('value_loss', 0) for l in recent])
                    print(f"  Avg policy/value loss: {pl:.4f}/{vl:.4f}")
            print(f"Best model saved to: {best_model_path}")
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
