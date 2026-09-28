"""
walk_circle.py -- walk the A1 on a circle using a1_ik.CrawlGait.

Per-leg step lengths come from rigid-body kinematics. For body speed v
and yaw rate w = v / R, a foot at (fx, fy) relative to the rotation
centre of the body must sweep with velocity

    vx = v - w * fy        (outer legs longer, inner legs shorter)
    vy = w * fx            (front legs step sideways one way, rear the other)

and stride = velocity * stance_time, stance_time = (1 - swing_frac) * T.
So length_x and length_y differ on every leg.

  python walk_circle.py --urdf urdf/a1_unitree.urdf --radius 0.6
  python walk_circle.py --radius -0.6            # negative = clockwise (right)
  python walk_circle.py --direct --laps 1        # headless + circle fit

radius > 0: circle centre on the robot's left (counter-clockwise).
Needs a1_ik.py and walk_test.py in the same folder.
"""

import argparse
import math

import numpy as np
import pybullet as p
import pybullet_data
import sys
from pathlib import Path

# Navigates up two levels (from script.py -> scripts/ -> project/) and adds it to Python's search path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from Inverse_Kinematics.IK_A1 import LEG_NAMES, SIDE, L1, CrawlGait, CycleCommand, LegStep
from IK_module_test import (
    DT, Z_DOWN, Z_STAND, CG_X,
    leg_joint_map, command_legs, crouch_targets, step, stand_up, yaw_of,
)

# hip positions in the trunk frame (same values as the original script)
HIP_POS = {
    "FR": (0.1805, -0.0511),
    "FL": (0.1805, 0.0429),
    "RR": (-0.1805, -0.0511),
    "RL": (-0.1805, 0.0429),
}
# rotate about the centre of the four hips
CENTER = (
    sum(h[0] for h in HIP_POS.values()) / 4.0,
    sum(h[1] for h in HIP_POS.values()) / 4.0,
)


def circle_command(radius, speed, cycle_time=1.0, swing_frac=0.15,
                   height=0.10, max_lx=0.10, max_ly=0.05):
    """CycleCommand with per-leg (length_x, length_y) for a circle of
    signed `radius` at body speed `speed`. If any stride exceeds the
    limits, ALL legs are scaled by the same factor (radius preserved,
    speed reduced). Returns (command, effective_speed)."""
    omega = speed / radius
    stance_t = (1.0 - swing_frac) * cycle_time

    strides = {}
    for leg in LEG_NAMES:
        hx, hy = HIP_POS[leg]
        # neutral foot position relative to the rotation centre
        fx = hx + CG_X - CENTER[0]
        fy = hy + SIDE[leg] * L1 - CENTER[1]
        strides[leg] = ((speed - omega * fy) * stance_t, (omega * fx) * stance_t)

    scale = 1.0
    peak_x = max(abs(s[0]) for s in strides.values())
    peak_y = max(abs(s[1]) for s in strides.values())
    if peak_x > max_lx:
        scale = min(scale, max_lx / peak_x)
    if peak_y > max_ly:
        scale = min(scale, max_ly / peak_y)

    legs = {l: LegStep(s[0] * scale, s[1] * scale, height) for l, s in strides.items()}
    cmd = CycleCommand(legs=legs, cycle_time=cycle_time, stance_z=Z_STAND,
                       swing_frac=swing_frac, cg_offset_x=CG_X)
    return cmd, speed * scale


def fit_circle(xy):
    """Least-squares circle fit (Kasa). Returns (cx, cy, r)."""
    x, y = xy[:, 0], xy[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x * x + y * y
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    return cx, cy, math.sqrt(max(c + cx * cx + cy * cy, 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default="urdf/a1_unitree.urdf")
    ap.add_argument("--radius", type=float, default=0.6,
                    help="signed circle radius [m]; + = left/CCW, - = right/CW")
    ap.add_argument("--speed", type=float, default=0.09, help="body speed [m/s]")
    ap.add_argument("--cycle-time", type=float, default=1.0)
    ap.add_argument("--height", type=float, default=0.10)
    ap.add_argument("--laps", type=float, default=1.0)
    ap.add_argument("--direct", action="store_true", help="headless, no realtime")
    args = ap.parse_args()

    if abs(args.radius) < 0.25:
        print("warning: radius < ~0.25 m will hit stride limits; expect heavy scaling")

    cmd, v_eff = circle_command(args.radius, args.speed,
                                cycle_time=args.cycle_time, height=args.height)
    omega = v_eff / args.radius
    print(f"radius={args.radius:+.2f} m  v={v_eff:.3f} m/s (requested {args.speed:.3f})  "
          f"yaw rate={math.degrees(omega):+.2f} deg/s")
    print("per-leg strides (m):")
    for l in LEG_NAMES:
        s = cmd.legs[l]
        print(f"  {l}: length_x={s.length_x:+.4f}  length_y={s.length_y:+.4f}  height={s.height:.2f}")

    gui = not args.direct
    p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(DT)
    p.loadURDF("plane.urdf")

    robot = p.loadURDF(args.urdf, [0, 0, abs(Z_DOWN) + 0.02], useFixedBase=False)
    joints = leg_joint_map(robot)
    for j in range(p.getNumJoints(robot)):
        p.setJointMotorControl2(robot, j, p.VELOCITY_CONTROL, force=0)

    command_legs(robot, joints, crouch_targets(Z_DOWN), reset=True)
    for _ in range(int(0.3 / DT)):
        step(gui)
    stand_up(robot, joints, gui)
    print("stand-up done, starting circle walk")

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

    traj, next_log, next_print = [], 0.0, 0.0
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
                x, y = to_start_frame(pos)
                yaw = math.degrees(yaw_of(robot) - yaw0)
                print(f"  t={gait.t:5.1f}s  x={x:+.3f} y={y:+.3f} z={pos[2]:.3f}  "
                      f"yaw={yaw:+7.1f} deg (expected {math.degrees(omega * gait.t):+7.1f})")
                if pos[2] < 0.12:
                    print("  !! base very low / fallen, stopping")
                    break
    except KeyboardInterrupt:
        pass

    yaw_total = math.degrees(yaw_of(robot) - yaw0)
    print(f"\nyaw change (unwrapped by eye, wraps at +-180): {yaw_total:+.1f} deg, "
          f"expected {math.degrees(omega * gait.t):+.1f} deg")
    if len(traj) >= 10:
        cx, cy, r = fit_circle(np.array(traj))
        print(f"circle fit: radius={r:.3f} m (commanded {abs(args.radius):.3f}), "
              f"centre=({cx:+.3f}, {cy:+.3f}) m in start frame "
              f"(commanded (0, {args.radius:+.3f}))")
        print("If the fit centre is on the wrong side or the radius is far off, check "
              "IK_X_SIGN in a1_ik.py and the sign of length_y (lateral) in the IK frame.")
    p.disconnect()


if __name__ == "__main__":
    main()