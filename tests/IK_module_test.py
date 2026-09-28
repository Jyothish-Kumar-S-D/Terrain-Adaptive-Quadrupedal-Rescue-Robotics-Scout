"""
walk_test.py -- test a1_ik.CrawlGait in PyBullet.

  python walk_test.py --urdf urdf/a1_unitree.urdf --mode forward
  python walk_test.py --mode sequence            # runs through every demo
  python walk_test.py --mode sidestep --direct   # headless, prints result

Modes: forward | backward | sidestep | turn | sequence | custom
The step params are re-issued every cycle by a cycle_provider, so you
can see per-cycle changes take effect ("sequence" and "custom").
"""

import argparse
import math
import time

import pybullet as p
import sys
import pybullet_data
from pathlib import Path

# Navigates up two levels (from script.py -> scripts/ -> project/) and adds it to Python's search path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)


from Inverse_Kinematics.IK_A1 import (
    LEG_NAMES, SIDE, L1, MAX_REACH,
    CrawlGait, CycleCommand, LegStep, foot_to_joints,
)

DT = 1.0 / 240.0
MAX_FORCE = 33.5
Z_DOWN, Z_STAND = -0.10, -0.22  # Z_STAND must equal CycleCommand.stance_z
CG_X = -0.085                   # must equal CycleCommand.cg_offset_x


# ----------------------------------------------------------------------
# Cycle schedules. Each returns the CycleCommand for cycle number n.
# ----------------------------------------------------------------------
def demo_forward(n):
    return CycleCommand.uniform(length_x=0.08, height=0.10)


def demo_backward(n):
    return CycleCommand.uniform(length_x=-0.08, height=0.10)


def demo_sidestep(n):
    # pure lateral crawl to the left; flip the sign of length_y for right
    return CycleCommand.uniform(length_x=0.0, length_y=0.05, height=0.10)


def demo_turn(n):
    return CycleCommand.differential(length_x=0.08, turn=0.6, height=0.10)


def demo_custom(n):
    """Fully per-leg example: every leg gets its own numbers, changing
    every cycle. Edit freely."""
    base = CycleCommand.uniform(length_x=0.08, height=0.10)
    cmd = base.with_leg("FL", height=0.12 if n % 2 else 0.08)  # per-leg height
    cmd = cmd.with_leg("RR", length_x=0.06, length_y=0.01)      # per-leg strides
    return cmd


def demo_sequence(n):
    # 4 cycles each: forward -> forward+drift left -> turn -> sidestep -> back
    stage = (n // 4) % 5
    return [demo_forward, lambda k: CycleCommand.uniform(0.08, 0.02, 0.10),
            demo_turn, demo_sidestep, demo_backward][stage](n)


DEMOS = {
    "forward": demo_forward, "backward": demo_backward,
    "sidestep": demo_sidestep, "turn": demo_turn,
    "sequence": demo_sequence, "custom": demo_custom,
}


# ----------------------------------------------------------------------
# PyBullet helpers
# ----------------------------------------------------------------------
def leg_joint_map(robot):
    name_to_idx = {}
    for j in range(p.getNumJoints(robot)):
        name_to_idx[p.getJointInfo(robot, j)[1].decode()] = j
    return {
        leg: (name_to_idx[f"{leg}_hip_joint"],
              name_to_idx[f"{leg}_thigh_joint"],
              name_to_idx[f"{leg}_calf_joint"])
        for leg in LEG_NAMES
    }


def command_legs(robot, joints, targets, reset=False):
    for leg, q in targets.items():
        for j, qi in zip(joints[leg], q):
            if reset:
                p.resetJointState(robot, j, qi)
            p.setJointMotorControl2(robot, j, p.POSITION_CONTROL,
                                    targetPosition=qi, force=MAX_FORCE)


def crouch_targets(z):
    return {leg: foot_to_joints(leg, CG_X, 0.0, z) for leg in LEG_NAMES}


def step(realtime):
    p.stepSimulation()
    if realtime:
        time.sleep(DT)


def stand_up(robot, joints, realtime, duration=2.0, hold=0.5):
    n = int(duration / DT)
    for i in range(n + 1):
        s = i / n
        z = Z_DOWN + (Z_STAND - Z_DOWN) * 0.5 * (1 - math.cos(math.pi * s))
        command_legs(robot, joints, crouch_targets(z))
        step(realtime)
    for _ in range(int(hold / DT)):
        step(realtime)


def yaw_of(robot):
    _, orn = p.getBasePositionAndOrientation(robot)
    return p.getEulerFromQuaternion(orn)[2]


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default="urdf/a1_unitree.urdf")
    ap.add_argument("--mode", choices=DEMOS, default="forward")
    ap.add_argument("--duration", type=float, default=20.0, help="sim seconds")
    ap.add_argument("--direct", action="store_true", help="headless, no GUI/realtime")
    args = ap.parse_args()

    gui = not args.direct
    p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(DT)
    p.loadURDF("plane.urdf")

    robot = p.loadURDF(args.urdf, [0, 0, abs(Z_DOWN) + 0.02], useFixedBase=False)
    joints = leg_joint_map(robot)

    for j in range(p.getNumJoints(robot)):  # disable default velocity motors
        p.setJointMotorControl2(robot, j, p.VELOCITY_CONTROL, force=0)

    command_legs(robot, joints, crouch_targets(Z_DOWN), reset=True)
    for _ in range(int(0.3 / DT)):
        step(gui)
    stand_up(robot, joints, gui)
    print("stand-up done, starting gait:", args.mode)

    provider = DEMOS[args.mode]
    gait = CrawlGait(provider(0), cycle_provider=provider)

    pos0, yaw0 = p.getBasePositionAndOrientation(robot)[0], yaw_of(robot)
    steps = int(args.duration / DT)
    last_cycle, next_print = -1, 0.0
    try:
        for i in range(steps):
            command_legs(robot, joints, gait.update(DT))
            step(gui)

            if gait.cycle_index != last_cycle:
                last_cycle = gait.cycle_index
                c = gait.cmd
                print(f"cycle {last_cycle:3d}  " + "  ".join(
                    f"{l}:(x={c.legs[l].length_x:+.3f} y={c.legs[l].length_y:+.3f} "
                    f"h={c.legs[l].height:.2f})" for l in LEG_NAMES))
            if gait.t >= next_print:
                next_print += 1.0
                (x, y, z), yaw = p.getBasePositionAndOrientation(robot)[0], yaw_of(robot)
                print(f"  t={gait.t:5.1f}s  base=({x:+.3f},{y:+.3f},{z:.3f}) "
                      f"yaw={math.degrees(yaw):+6.1f} deg  swinging={gait.swinging}")
                if z < 0.12:
                    print("  !! base is very low / fallen, stopping")
                    break
    except KeyboardInterrupt:
        pass

    pos1, yaw1 = p.getBasePositionAndOrientation(robot)[0], yaw_of(robot)
    dxw, dyw = pos1[0] - pos0[0], pos1[1] - pos0[1]
    c, s = math.cos(yaw0), math.sin(yaw0)
    fwd, left = c * dxw + s * dyw, -s * dxw + c * dyw  # into initial body frame
    print(f"\nnet motion (initial body frame): forward={fwd:+.3f} m  left={left:+.3f} m  "
          f"yaw={math.degrees(yaw1 - yaw0):+.1f} deg")
    if args.mode in ("forward", "backward"):
        expect = 1 if args.mode == "forward" else -1
        if fwd * expect < 0:
            print("=> moved the WRONG way: flip IK_X_SIGN in a1_ik.py")
    p.disconnect()


if __name__ == "__main__":
    main()