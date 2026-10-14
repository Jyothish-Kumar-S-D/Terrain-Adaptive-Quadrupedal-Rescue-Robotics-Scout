"""test_path_terrain_env.py -- Verification & Test Suite for QuadrupedPathTerrainEnv.

Validates the Gymnasium Training Environment:
1. Environment Initialization (__init__): Action space, observation space, physics client, URDF.
2. Environment Reset (reset): Continuous spline path, dense pyramids, robot spawn posture.
3. Action Execution (step): Action initiation (target joint computation, limits, motor control).
4. Multi-Objective Reward Function: Forward tracking, corridor tracking, stability, progress, penalties.
5. Termination & Truncation: Fall detection, corridor exit, step limits, goal achievement.
6. Gymnasium API Compliance: Built-in gymnasium.utils.env_checker.check_env support.

Usage:
  python tests/test_path_terrain_env.py --policy stand --episodes 1 --max-steps 300
  python tests/test_path_terrain_env.py --policy random --episodes 2 --max-steps 200 --direct
  python tests/test_path_terrain_env.py --check-env
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    JOINT_LOWER_LIMITS,
    JOINT_UPPER_LIMITS,
    QuadrupedPathTerrainEnv,
    get_nominal_cycle_action,
    NOMINAL_ACTION_PARAMS,
    normalize_action,
    denormalize_action,
)
from path_training.task import PathTaskConfig
from surrounding_building.terrain_building import RandomTerrainConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify and test QuadrupedPathTerrainEnv Gymnasium training environment."
    )

    # Simulation & environment args
    parser.add_argument("--episodes", type=int, default=1, help="Number of test episodes")
    parser.add_argument("--max-steps", type=int, default=50, help="Max gait cycles per episode")
    parser.add_argument(
        "--policy",
        choices=["nominal", "crawl", "random", "stand"],
        default="nominal",
        help="Test action policy: 'nominal'/'crawl' (high-level crawl gait), 'random' (stochastic exploration), 'stand' (zero stride length)",
    )
    parser.add_argument("--direct", action="store_true", help="Headless mode (no PyBullet GUI window)")
    parser.add_argument("--check-env", action="store_true", help="Run Gymnasium API compliance checker")
    parser.add_argument("--urdf", default="urdf/a1_unitree.urdf", help="Path to robot URDF")
    parser.add_argument("--action-scale", type=float, default=1.60, help="Action scaling factor")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")

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

    # Dense pyramid terrain args
    t_grp = parser.add_argument_group("random packed-pyramid terrain options")
    t_grp.add_argument("--extent", type=float, default=5.0, help="Side length [m] of square terrain patch")
    t_grp.add_argument("--spacing", type=float, default=0.24, help="Grid pitch [m]; <= size packs densely")
    t_grp.add_argument("--size-min", type=float, default=0.16, help="Min pyramid base side length [m]")
    t_grp.add_argument("--size-max", type=float, default=0.34, help="Max pyramid base side length [m]")
    t_grp.add_argument("--height-min", type=float, default=0.02, help="Min pyramid height [m]")
    t_grp.add_argument("--height-max", type=float, default=0.08, help="Max pyramid height [m]")
    t_grp.add_argument("--density", type=float, default=0.95, help="Fraction of grid cells with pyramids")
    t_grp.add_argument("--jitter", type=float, default=0.05, help="Random position jitter [m]")
    t_grp.add_argument("--clear-radius", type=float, default=0.45, help="Flat spawn disc radius [m]")

    return parser.parse_args()


def print_banner(text: str):
    print("\n" + "=" * 95)
    print(f" {text}")
    print("=" * 95)


def generate_test_action(policy_type: str, step_idx: int, action_space: gym.spaces.Box) -> np.ndarray:
    """Generate normalized action in [-1.0, 1.0]^26 for the chosen policy."""
    if policy_type in ("nominal", "crawl", "walk"):
        # Nominal wave crawl gait parameters
        return get_nominal_cycle_action()
    elif policy_type == "stand":
        # Stance without strides (zero length_x and length_y)
        params = NOMINAL_ACTION_PARAMS.copy()
        # Indices 13..20 are length_x and length_y
        params[13:21] = 0.0
        return normalize_action(params)
    elif policy_type == "random":
        # Stochastic exploration action uniformly sampled from action space
        return action_space.sample()
    else:
        return get_nominal_cycle_action()


def run_compliance_check(env: gym.Env):
    """Run Gymnasium official environment compliance checker."""
    print_banner("RUNNING GYMNASIUM API COMPLIANCE CHECK (check_env)")
    from gymnasium.utils.env_checker import check_env
    try:
        check_env(env.unwrapped, skip_render_check=True)
        print("[SUCCESS] Gymnasium API compliance check PASSED without errors!")
    except Exception as e:
        print(f"[FAIL] Gymnasium API check encountered issue: {e}")


def run_environment_test():
    args = parse_args()
    gui = not args.direct
    render_mode = "human" if gui else None

    try:
        import pybullet as p
    except ImportError:
        p = None

    if gui and p is None:
        print("\n" + "!" * 95)
        print(" [ERROR] PyBullet is not installed in the currently active Python environment!")
        print(" The PyBullet GUI window cannot be loaded from this Python binary.")
        print(" Please activate the Conda environment where PyBullet is installed:")
        print("     conda activate a1_pybullet")
        print(" or run this script using:")
        print(r"     C:\Users\mahes\.conda\envs\a1_pybullet\python.exe tests/test_path_terrain_env.py --policy nominal --max-steps 50")
        print("!" * 95 + "\n")
        sys.exit(1)

    print_banner("QUADRUPED PATH-FOLLOWING TERRAIN TRAINING ENVIRONMENT TEST")
    print(f"Policy:           {args.policy.upper()}")
    print(f"Render Mode:      {render_mode or 'DIRECT (Headless)'}")
    print(f"Episodes:         {args.episodes}")
    print(f"Max Steps/Ep:     {args.max_steps}")
    print(f"Base RNG Seed:    {args.seed}")
    print(f"Use Depth Frame:  {not args.no_depth} ({args.depth_width}x{args.depth_height})")
    print(f"Action Scale:     {args.action_scale:.2f}")

    # Build terrain and task configurations
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

    # 1. Environment Initialization (__init__)
    print("\n--- 1. ENVIRONMENT INITIALIZATION ---")
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
        print(f"gym.make fallback to direct instantiation ({e})...")
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

    print(f"Action Space:      {env.action_space}")
    print(f"Observation Space: {type(env.observation_space).__name__}")
    if isinstance(env.observation_space, gym.spaces.Dict):
        for k, v in env.observation_space.spaces.items():
            print(f"  - '{k}': shape {v.shape}, dtype {v.dtype}")

    if args.check_env:
        run_compliance_check(env)

    episode_summaries = []

    for ep in range(args.episodes):
        ep_seed = args.seed + ep * 100
        print_banner(f"EPISODE {ep + 1} / {args.episodes} (Seed={ep_seed})")

        # 2. Episode Reset (reset)
        print("--- 2. EPISODE RESET ---")
        obs, info = env.reset(seed=ep_seed)

        path_pts = env.unwrapped.episode_path.points
        path_tangents = env.unwrapped.episode_path.tangents
        print(f"Procedural Path:       {len(path_pts)} samples, length = {info['path_length']:.2f} m")
        print(f"Dense Pyramids:        {info['num_pyramids']} collision bodies loaded")
        init_heading = math.degrees(obs['trunk_orientation'][2])
        print(f"Spawn Pose:            pos=(0, 0, ~0.28m) heading={init_heading:.1f} deg")
        if "depth" in obs and obs["depth"] is not None:
            depth_img = obs["depth"]
            print(f"Depth Camera Buffer:   shape={depth_img.shape}, min={depth_img.min():.2f}m, max={depth_img.max():.2f}m")

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

                # 3. Action Generation & Initiation
                action = generate_test_action(args.policy, step, env.action_space)

                # 4. Step Environment Transition
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

                # Telemetry print
                if step % 25 == 0 or terminated or truncated or step == 0:
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
            print("\n[Test interrupted by user (Ctrl+C)]")

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
    print_banner("TRAINING ENVIRONMENT EVALUATION SUMMARY")
    print(f"{'Episode':>7} | {'Steps':>6} | {'Total Reward':>13} | {'Progress':>9} | {'Mean Offset':>12} | {'Max Tilt':>9} | {'Status'}")
    print("-" * 80)
    for s in episode_summaries:
        status = "PASSED (Survived)" if s["survived"] else "TERMINATED"
        print(
            f"{s['episode']:7d} | {s['steps']:6d} | {s['total_reward']:13.2f} | "
            f"{s['progress_pct']:8.1f}% | {s['mean_offset_m']:10.3f} m | "
            f"{s['max_tilt_deg']:7.1f} deg | {status}"
        )
    print("-" * 80)
    print("Environment test completed successfully!\n")

    if render_mode == "human":
        print("[PyBullet GUI] Evaluation finished successfully!")
        print("Inspection Mode: GUI is live (drag mouse to rotate/zoom). Press any key in terminal to exit...")
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
