"""test_path_terrain_env.py -- Comprehensive test script for QuadrupedPathTerrainEnv.

Follows the testing conventions from terrain_circle_walk_test.py and circle_wlk_test.py:
- Tests the Gymnasium environment using either:
  1) 'crawl': Analytical CrawlGait walking over the dense pyramid terrain
  2) 'stand': Nominal standing pose stability check
  3) 'random': Stochastic exploration policy
- Evaluates:
  - Randomized continuous path initialization per episode
  - Dense packed pyramid terrain generation
  - Observation dictionary structure (joint states, trunk orientation, path offset, depth frame)
  - Multi-objective reward telemetry (path position, velocity tangent alignment, trunk stability, progress)
- Supports both interactive PyBullet GUI with camera tracking and fast headless (--direct) mode.

Usage:
  python tests/test_path_terrain_env.py --policy crawl --episodes 2
  python tests/test_path_terrain_env.py --direct --episodes 3 --max-steps 150
  python tests/test_path_terrain_env.py --policy stand --seed 42 --gui
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure project root is in sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

import gymnasium as gym

# Import environment and task components
import path_training
from path_training.gym_env import (
    JOINT_NAMES,
    NOMINAL_JOINT_ANGLES,
    QuadrupedPathTerrainEnv,
)
from path_training.task import PathTaskConfig
from surrounding_building.terrain_building import RandomTerrainConfig

# Import IK gait components for realistic quadruped testing
from Inverse_Kinematics.IK_A1 import (
    LEG_NAMES,
    SIDE,
    IK_X_SIGN,
    L1,
    CrawlGait,
    CycleCommand,
    leg_fk,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test QuadrupedPathTerrainEnv with randomized path and dense pyramid terrain."
    )

    # Simulation & environment args
    parser.add_argument("--episodes", type=int, default=1, help="Number of test episodes")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument(
        "--policy",
        choices=["crawl", "stand", "random"],
        default="crawl",
        help="Evaluation policy to drive the robot: 'crawl' (IK gait), 'stand', or 'random'",
    )
    parser.add_argument("--direct", action="store_true", help="Headless mode (no PyBullet GUI window)")
    parser.add_argument("--urdf", default="urdf/a1_unitree.urdf", help="Path to robot URDF")
    parser.add_argument("--action-scale", type=float, default=1.60, help="Action scaling factor")

    # Depth camera args
    d_grp = parser.add_argument_group("depth camera options")
    d_grp.add_argument("--no-depth", action="store_true", help="Disable depth camera observation")
    d_grp.add_argument("--depth-width", type=int, default=64, help="Downscaled depth frame width")
    d_grp.add_argument("--depth-height", type=int, default=48, help="Downscaled depth height")

    # Path configuration args
    p_grp = parser.add_argument_group("path generator options")
    p_grp.add_argument("--path-length", type=float, default=3.0, help="Target path length in meters")
    p_grp.add_argument("--path-width", type=float, default=1.0, help="Path corridor lateral width in meters")
    p_grp.add_argument("--waypoints", type=int, default=6, help="Number of spline control waypoints")
    p_grp.add_argument("--path-samples", type=int, default=300, help="Number of spline curve samples")
    p_grp.add_argument("--target-speed", type=float, default=0.15, help="Target path-following speed [m/s]")

    # Dense pyramid terrain args (matches terrain_circle_walk_test.py)
    t_grp = parser.add_argument_group("random packed-pyramid terrain options")
    t_grp.add_argument("--seed", type=int, default=7, help="Base RNG seed; same seed -> same layout")
    t_grp.add_argument("--extent", type=float, default=5.0, help="Side length [m] of square terrain patch")
    t_grp.add_argument("--spacing", type=float, default=0.24, help="Grid pitch [m]; <= size packs densely")
    t_grp.add_argument("--size-min", type=float, default=0.16, help="Min pyramid base side length [m]")
    t_grp.add_argument("--size-max", type=float, default=0.34, help="Max pyramid base side length [m]")
    t_grp.add_argument("--height-min", type=float, default=0.02, help="Min pyramid height [m]")
    t_grp.add_argument("--height-max", type=float, default=0.08, help="Max pyramid height [m]")
    t_grp.add_argument("--density", type=float, default=0.95, help="Fraction of grid cells with pyramids")
    t_grp.add_argument("--jitter", type=float, default=0.05, help="Random position jitter [m]")
    t_grp.add_argument("--clear-radius", type=float, default=0.45, help="Flat spawn disc radius [m]")

    # Crawl gait tuning (when --policy crawl is selected)
    c_grp = parser.add_argument_group("crawl gait options")
    c_grp.add_argument("--stride-x", type=float, default=0.055, help="Forward stride length [m]")
    c_grp.add_argument("--swing-height", type=float, default=0.13, help="Foot clearance in swing [m]")
    c_grp.add_argument("--cycle-time", type=float, default=1.2, help="Gait cycle period [s]")

    return parser.parse_args()


class CrawlPolicy:
    """Analytical IK crawl gait policy mapped into Gymnasium action space with path steering."""

    def __init__(
        self,
        stride_x: float = 0.055,
        swing_height: float = 0.13,
        cycle_time: float = 1.2,
        action_scale: float = 1.60,
        dt: float = 4.0 / 240.0,
    ):
        self.stride_x = stride_x
        self.swing_height = swing_height
        self.cycle_time = cycle_time
        self.cmd = CycleCommand.uniform(
            length_x=stride_x,
            length_y=0.0,
            height=swing_height,
            cycle_time=cycle_time,
        )
        self.gait = CrawlGait(self.cmd, cycle_provider=lambda n: self.cmd)
        self.action_scale = max(action_scale, 1e-4)
        self.dt = dt
        self._last_joint_positions: Optional[np.ndarray] = None

    def reset(self):
        self.cmd = CycleCommand.uniform(
            length_x=self.stride_x,
            length_y=0.0,
            height=self.swing_height,
            cycle_time=self.cycle_time,
        )
        self.gait = CrawlGait(self.cmd, cycle_provider=lambda n: self.cmd)
        self._last_joint_positions = None

    def steer(self, obs: Dict[str, np.ndarray], info: Dict[str, Any], path_tangents: np.ndarray):
        """Pure-pursuit lookahead steering with inner-leg deceleration to prevent singularity."""
        self._last_joint_positions = obs.get("joint_positions")

        offset = float(obs["path_offset"][0])
        curr_idx = int(info.get("path_index", 0))
        # Look ahead 6 waypoints (~9 cm ahead) to anticipate curvature
        lookahead_idx = min(curr_idx + 6, len(path_tangents) - 1)
        tangent = path_tangents[lookahead_idx]
        target_yaw = math.atan2(tangent[1], tangent[0])
        current_yaw = float(obs["trunk_orientation"][2])
        yaw_err = (target_yaw - current_yaw + math.pi) % (2.0 * math.pi) - math.pi

        # Reduce forward stride on sharp bends so differential steering pivots cleanly
        speed_factor = max(0.40, 1.0 - 0.50 * min(1.0, abs(yaw_err) / 0.8))
        effective_stride = self.stride_x * speed_factor

        turn = np.clip(1.5 * offset + 1.2 * yaw_err, -0.70, 0.70)

        # Active terrain attitude leveling:
        # If trunk is tilted because one side/end is high on a pyramid, retract elevated legs
        # to level the trunk and keep the center of gravity safely within the support polygon.
        roll = float(obs["trunk_orientation"][0])
        pitch = float(obs["trunk_orientation"][1])
        leg_z = {}
        for leg in LEG_NAMES:
            y_leg = 0.13 if 'L' in leg else -0.13
            x_leg = 0.18 if 'F' in leg else -0.18
            z_rel = y_leg * math.sin(roll) - x_leg * math.sin(pitch)
            # Only retract elevated legs (up to 5.5 cm); NEVER extend deeper than -0.21m
            dz_retract = max(0.0, min(0.055, z_rel))
            leg_z[leg] = -0.21 + dz_retract

        self.cmd = CycleCommand.differential(
            length_x=effective_stride,
            turn=float(turn),
            height=self.swing_height,
            cycle_time=self.cycle_time,
            leg_z=leg_z,
            stance_width=0.025,
        )
        self.gait.cmd = self.cmd

    def get_action(self) -> np.ndarray:
        # Compute actual foot positions from observation FK for obstacle-aware liftoff
        actual_feet = None
        if self._last_joint_positions is not None and len(self._last_joint_positions) == 12:
            actual_feet = {}
            for i, leg in enumerate(LEG_NAMES):
                q_leg = self._last_joint_positions[i * 3 : (i + 1) * 3]
                side = SIDE[leg]
                a, y_fk, z_fk = leg_fk(float(q_leg[0]), float(q_leg[1]), float(q_leg[2]), side)
                actual_feet[leg] = (a * IK_X_SIGN, y_fk - side * L1, z_fk)

        targets_dict = self.gait.update(self.dt, actual_foot_pos=actual_feet)
        # Flatten joint targets in the canonical environment order: FR, FL, RR, RL
        target_list = []
        for leg in LEG_NAMES:
            q_leg = targets_dict[leg]
            target_list.extend(q_leg)

        target_q = np.array(target_list, dtype=np.float32)
        # Invert action mapping: action = (target_q - nominal_q) / action_scale
        raw_action = (target_q - NOMINAL_JOINT_ANGLES) / self.action_scale
        return np.clip(raw_action, -1.0, 1.0)


def print_banner(text: str):
    print("\n" + "=" * 90)
    print(f" {text}")
    print("=" * 90)


def run_environment_test():
    args = parse_args()
    gui = not args.direct
    render_mode = "human" if gui else None

    print_banner("TESTING QUADRUPED PATH-FOLLOWING TERRAIN GYMNASIUM ENVIRONMENT")
    print(f"Policy:           {args.policy.upper()}")
    print(f"Render Mode:      {render_mode or 'DIRECT (Headless)'}")
    print(f"Episodes:         {args.episodes}")
    print(f"Max Steps/Ep:     {args.max_steps}")
    print(f"Use Depth Frame:  {not args.no_depth} ({args.depth_width}x{args.depth_height})")
    print(f"Terrain Patch:    {args.extent:.1f}m extent, {args.spacing:.2f}m pitch, density={args.density}")
    print(f"Pyramid Sizes:    [{args.size_min:.2f}, {args.size_max:.2f}]m base, [{args.height_min:.2f}, {args.height_max:.2f}]m height")

    # Build custom terrain & task configurations
    terrain_cfg = RandomTerrainConfig(
        seed=args.seed,
        extent=args.extent,
        spacing=args.spacing,
        size=(args.size_min, args.size_max),
        height=(args.height_min, args.height_max),
        density=args.density,
        jitter=args.jitter,
        clear_radius=args.clear_radius,
    )

    task_cfg = PathTaskConfig(
        path_length=args.path_length,
        path_width=args.path_width,
        waypoint_count=args.waypoints,
        path_samples=args.path_samples,
        target_speed=args.target_speed,
        depth_width=args.depth_width,
        depth_height=args.depth_height,
        terrain=terrain_cfg,
    )

    # 1. Instantiate environment via gym.make
    try:
        env = gym.make(
            "QuadrupedPathTerrain-v0",
            render_mode=render_mode,
            urdf_path=args.urdf,
            task_config=task_cfg,
            terrain_config=terrain_cfg,
            action_scale=args.action_scale,
            use_depth=not args.no_depth,
            depth_width=args.depth_width,
            depth_height=args.depth_height,
            max_episode_steps=args.max_steps,
        )
    except Exception as e:
        print(f"Direct gym.make failed ({e}), falling back to direct instantiation...")
        env = QuadrupedPathTerrainEnv(
            render_mode=render_mode,
            urdf_path=args.urdf,
            task_config=task_cfg,
            terrain_config=terrain_cfg,
            action_scale=args.action_scale,
            use_depth=not args.no_depth,
            depth_width=args.depth_width,
            depth_height=args.depth_height,
            max_episode_steps=args.max_steps,
        )

    print("\n[Environment Verification]")
    print(f"  Action Space:      {env.action_space}")
    print(f"  Observation Space: {type(env.observation_space).__name__}")
    if isinstance(env.observation_space, gym.spaces.Dict):
        for k, v in env.observation_space.spaces.items():
            print(f"    - '{k}': shape {v.shape}, dtype {v.dtype}")

    # Set up policy
    dt_control = env.unwrapped.sim_time_step * env.unwrapped.action_repeat
    crawl_policy = CrawlPolicy(
        stride_x=args.stride_x,
        swing_height=args.swing_height,
        cycle_time=args.cycle_time,
        action_scale=args.action_scale,
        dt=dt_control,
    )

    episode_summaries = []

    for ep in range(args.episodes):
        ep_seed = args.seed + ep * 100
        print_banner(f"EPISODE {ep + 1} / {args.episodes} (Seed={ep_seed})")

        obs, info = env.reset(seed=ep_seed)
        crawl_policy.reset()

        path_pts = env.unwrapped.episode_path.points
        path_tangents = env.unwrapped.episode_path.tangents
        print(f"Generated Continuous Path: {len(path_pts)} samples, length = {info['path_length']:.2f} m")
        print(f"Loaded Dense Pyramids:     {info['num_pyramids']} collision bodies on top of plane")
        print(f"Initial Spawn Pose:        pos=(0, 0, ~0.28m) heading={math.degrees(obs['trunk_orientation'][2]):.1f} deg")
        if "depth" in obs and obs["depth"] is not None:
            depth_img = obs["depth"]
            print(f"Captured Depth Frame:      shape={depth_img.shape}, min={depth_img.min():.2f}m, max={depth_img.max():.2f}m")

        print("\n" + "-" * 115)
        print(f"{'Step':>4} | {'Rew Total':>9} | {'R_Path':>6} {'R_Vel':>6} {'R_Stab':>6} {'R_Prog':>6} | "
              f"{'Roll':>6} {'Pitch':>6} {'Yaw':>6} | {'Offset':>7} {'Dist':>6} | {'VelAngle':>8} | {'Status'}")
        print("-" * 115)

        total_reward = 0.0
        step_count = 0
        terminated = False
        truncated = False
        last_step_info = info
        lateral_offsets = []
        trunk_tilts = []

        try:
            for step in range(args.max_steps):
                step_count += 1

                # Select action based on chosen policy
                if args.policy == "crawl":
                    crawl_policy.steer(obs, last_step_info, path_tangents)
                    action = crawl_policy.get_action()
                elif args.policy == "stand":
                    action = np.zeros(12, dtype=np.float32)
                else:  # random
                    action = env.action_space.sample() * 0.15

                obs, reward, terminated, truncated, step_info = env.step(action)
                total_reward += reward
                last_step_info = step_info

                r_comp = step_info.get("reward_components", {})
                roll_deg = math.degrees(float(obs["trunk_orientation"][0]))
                pitch_deg = math.degrees(float(obs["trunk_orientation"][1]))
                yaw_deg = math.degrees(float(obs["trunk_orientation"][2]))
                offset_m = float(obs["path_offset"][0])
                dist_m = float(obs["path_distance"][0])
                vel_angle_deg = math.degrees(float(obs["velocity_tangent_angle"][0]))

                lateral_offsets.append(abs(offset_m))
                trunk_tilts.append(math.hypot(roll_deg, pitch_deg))

                # Print telemetry periodically or upon key events
                if step % 15 == 0 or terminated or truncated or step == 0:
                    status_str = "OK"
                    if step_info.get("fallen", False):
                        status_str = "FALLEN"
                    elif step_info.get("far_from_path", False):
                        status_str = "OFF-PATH"
                    elif step_info.get("reached_end", False):
                        status_str = "REACHED-END"

                    print(
                        f"{step_count:4d} | {reward:+9.3f} | "
                        f"{r_comp.get('reward_path', 0.0):6.2f} "
                        f"{r_comp.get('reward_velocity', 0.0):6.2f} "
                        f"{r_comp.get('reward_stability', 0.0):6.2f} "
                        f"{r_comp.get('reward_progress', 0.0):6.2f} | "
                        f"{roll_deg:+6.1f} {pitch_deg:+6.1f} {yaw_deg:+6.1f} | "
                        f"{offset_m:+7.3f} {dist_m:6.3f} | "
                        f"{vel_angle_deg:+8.1f} | {status_str}"
                    )

                if render_mode == "human":
                    time.sleep(1.0 / 60.0)

                if terminated or truncated:
                    break
        except KeyboardInterrupt:
            print("\n[Simulation interrupted by user (Ctrl+C)]")

        # Compute episode summary
        survived = not last_step_info.get("fallen", False)
        mean_offset = float(np.mean(lateral_offsets)) if lateral_offsets else 0.0
        max_tilt = float(np.max(trunk_tilts)) if trunk_tilts else 0.0
        final_idx = int(last_step_info.get("path_index", 0))
        path_progress_pct = 100.0 * final_idx / max(len(path_pts) - 1, 1)

        summary = {
            "episode": ep + 1,
            "steps": step_count,
            "total_reward": total_reward,
            "survived": survived,
            "progress_pct": path_progress_pct,
            "mean_offset_m": mean_offset,
            "max_tilt_deg": max_tilt,
        }
        episode_summaries.append(summary)

        print("-" * 115)
        print(f"Episode {ep + 1} Result:")
        print(f"  Survived:         {'YES' if survived else 'NO (robot fell)'}")
        print(f"  Steps completed:  {step_count} / {args.max_steps}")
        print(f"  Path Progress:    {path_progress_pct:.1f}% (waypoint index {final_idx}/{len(path_pts)})")
        print(f"  Mean Path Offset: {mean_offset:.3f} m")
        print(f"  Max Trunk Tilt:   {max_tilt:.1f} deg")
        print(f"  Cumulative Rew:   {total_reward:.2f}")

    # Final Test Report
    print_banner("TEST RUN EVALUATION SUMMARY")
    print(f"{'Episode':>7} | {'Steps':>6} | {'Total Reward':>13} | {'Progress':>9} | {'Mean Offset':>12} | {'Max Tilt':>9} | {'Status'}")
    print("-" * 80)
    for s in episode_summaries:
        status = "PASSED (Survived)" if s["survived"] else "FAILED (Fell)"
        print(
            f"{s['episode']:7d} | {s['steps']:6d} | {s['total_reward']:13.2f} | "
            f"{s['progress_pct']:8.1f}% | {s['mean_offset_m']:10.3f} m | "
            f"{s['max_tilt_deg']:7.1f} deg | {status}"
        )
    print("-" * 80)
    print("Environment test completed successfully!\n")

    if render_mode == "human":
        print("\n[PyBullet GUI] Episode finished successfully!")
        print("Inspection Mode: GUI is fully interactive (drag mouse to rotate/zoom).")
        print("Press any key in this terminal (or Ctrl+C) to close the window...")
        try:
            import msvcrt
            while True:
                p.stepSimulation()
                time.sleep(1.0 / 60.0)
                if msvcrt.kbhit():
                    msvcrt.getch()
                    break
        except (KeyboardInterrupt, Exception):
            pass

    env.close()


if __name__ == "__main__":
    run_environment_test()
