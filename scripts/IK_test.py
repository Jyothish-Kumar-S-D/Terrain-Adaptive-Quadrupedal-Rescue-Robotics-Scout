"""
A1 quadruped — analytic IK crawl (wave) gait in PyBullet, with IMU gyro
readout for a future stabilization loop.

Kinematic constants (L1, L2, L3 below) are computed directly from the
parent-frame offsets in the joint dump you supplied, not from generic
Unitree A1 spec sheets — if your URDF differs from that dump, re-derive
them the same way (see comments).

Leg chain per leg: hip (revolute, X axis, ab/ad) -> thigh (revolute, Y axis,
pitch) -> calf (revolute, Y axis, pitch) -> foot (fixed). All joint frames
share orientation with their parent link (no rotation offsets in the URDF),
so the leg is a standard 3-DOF roll-pitch-pitch chain with closed-form IK.
Small x/z components of the thigh->calf and calf->foot offsets (~3-6 mm)
are folded into effective link lengths L2, L3, not modeled as an extra
offset — a sagittal-plane approximation, fine for open-loop walking, not
exact FK.

Gait: static-stability crawl / wave gait — only one leg swings at a time,
the other three stay planted, in the order RL -> FL -> RR -> FR (rear-left,
front-left, rear-right, front-right), the standard "creep gait" sequence
used for maximal support-polygon margin. Duty factor 0.75 (each leg is in
stance 3/4 of the cycle, swing 1/4).
"""

import math
import time
import pybullet as p
import pybullet_data

import sys
from pathlib import Path

# Navigates up two levels (from script.py -> scripts/ -> project/) and adds it to Python's search path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

#Import modules from files
from depth_processing.Depth_Processing import DepthCamera
#files import ends


# ----------------------------------------------------------------------
# Constants derived from the supplied joint dump
# ----------------------------------------------------------------------

# FR_hip -> FR_thigh offset: (0.003311, -0.083165, -3.1e-05) -> hip length
L1 = 0.083165

# FR_thigh -> FR_calf offset: (0.003237, -0.022327, -0.172674)
L2 = math.sqrt(0.003237**2 + 0.022327**2 + 0.172674**2)

# FR_calf -> FR_foot offset: (-0.006435, 0.0, -0.092612)
L3 = math.sqrt(0.006435**2 + 0.092612**2)

MAX_REACH = L2 + L3

# Per-leg: joint name prefix, hip position in trunk frame (x,y,z),
# sideSign (+1 = left, hip offset toward +y; -1 = right, offset toward -y)
LEGS = {
    "FR": {"prefix": "FR", "hip_pos": (0.1805, -0.0511, 0.0005), "side": -1},
    "FL": {"prefix": "FL", "hip_pos": (0.1805, 0.0429, 0.0005), "side": 1},
    "RR": {"prefix": "RR", "hip_pos": (-0.1805, -0.0511, 0.0005), "side": -1},
    "RL": {"prefix": "RL", "hip_pos": (-0.1805, 0.0429, 0.0005), "side": 1},
}

# Wave-gait swing order (standard creep-gait sequence): RL, FL, RR, FR,
# each offset by one quarter of the cycle. Only ONE leg is in its swing
# window [0, SWING_FRAC) of its own local phase at any instant; the other
# three are always in stance.
PHASE_OFFSET = {"RL": 0.0, "FL": 0.25, "RR": 0.5, "FR": 0.75}
SWING_FRAC = 0.05# duty factor: 0.75 stance / 0.25 swing per leg

# +1 = stance sweeps foot front-to-back (pushes body toward +x / forward).
# If the robot still walks backward with this gait, flip this to -1 —
# the actual sign depends on your base +x convention and was not something
# this script's author could verify without running the sim.
FORWARD_SIGN = -1

# Static CG trim: shifts every foot backward relative to its hip by this
# amount. Since the feet are planted and the hips are rigid on the trunk,
# that pulls the trunk's resting position forward over the support
# polygon (negative pulls it backward). Start small — a few cm.
CG_OFFSET_X = -0.085  # meters

# ----------------------------------------------------------------------
# Analytic IK — target (x, y, z) is the foot position relative to the
# hip joint origin, in the hip's local (non-rotated) frame.
# ----------------------------------------------------------------------

def leg_ik(x, y, z, side_sign, l1=L1, l2=L2, l3=L3):
    l1s = side_sign * l1

    d = y * y + z * z - l1s * l1s
    if d < 0.0:
        d = 0.0
    b = -math.sqrt(d)  # vertical extent of the leg in the pitch plane (<=0)

    # hip ab/ad angle: rotates (l1s, b) onto (y, z)
    q1 = math.atan2(z, y) - math.atan2(b, l1s)

    a = x  # forward component, unaffected by hip roll
    r2 = a * a + b * b
    r2 = min(r2, (l2 + l3) ** 2)  # clamp to reachable workspace

    cos_q3 = (r2 - l2 * l2 - l3 * l3) / (2 * l2 * l3)
    cos_q3 = max(-1.0, min(1.0, cos_q3))
    q3 = -math.acos(cos_q3)  # negative branch: matches calf joint limits

    phi = math.atan2(a, -b)
    q2 = phi - math.atan2(l3 * math.sin(q3), l2 + l3 * math.cos(q3))

    return q1, q2, q3


# ----------------------------------------------------------------------
# Foot trajectory generator (crawl / wave gait): hip-relative (x, y, z)
# target as a function of each leg's own local phase in [0, 1). y is held
# at the neutral ab/ad offset so the hip roll joint stays near zero.
# ----------------------------------------------------------------------

def foot_target(local_phase, side_sign, step_len, step_height, stance_z):
    local_phase = local_phase % 1.0
    y = side_sign * L1
    step_len *= FORWARD_SIGN

    if local_phase < SWING_FRAC:
        # swing: foot lifted, sweeping forward (back -> front)
        s = local_phase / SWING_FRAC  # 0 -> 1
        x = step_len * (-0.5 + s)
        z = stance_z + step_height * math.sin(math.pi * s)
    else:
        # stance: foot planted, sweeping backward (front -> back),
        # which is what drives the body forward
        s = (local_phase - SWING_FRAC) / (1.0 - SWING_FRAC)  # 0 -> 1
        x = step_len * (0.5 - s)
        z = stance_z

    return x - CG_OFFSET_X, y, z


# ----------------------------------------------------------------------
# IMU gyro readout — reads the rigid-body angular velocity at imu_link
# (fixed joint, so identical to the trunk's angular velocity) and rotates
# it into the IMU's own body frame, as a real gyro would report it.
# Not yet used for control — this is the sensor hookup for a future
# balance/stabilization loop.
# ----------------------------------------------------------------------

def read_gyro(robot_id, imu_link_index):
    state = p.getLinkState(
        robot_id, imu_link_index,
        computeLinkVelocity=1, computeForwardKinematics=1,
    )
    world_orn = state[5]        # worldLinkFrameOrientation
    world_ang_vel = state[7]    # worldAngularVelocity

    r = p.getMatrixFromQuaternion(world_orn)  # row-major 3x3, link->world
    # body_frame = R^T @ world_ang_vel
    wx = r[0] * world_ang_vel[0] + r[3] * world_ang_vel[1] + r[6] * world_ang_vel[2]
    wy = r[1] * world_ang_vel[0] + r[4] * world_ang_vel[1] + r[7] * world_ang_vel[2]
    wz = r[2] * world_ang_vel[0] + r[5] * world_ang_vel[1] + r[8] * world_ang_vel[2]
    return wx, wy, wz  # (roll_rate, pitch_rate, yaw_rate), rad/s



# ----------------------------------------------------------------------
# Crouch / stand poses. Feet stay directly under the hips (x=0, neutral
# y) throughout — only the leg depth (z) changes, so the base rises
# vertically with no fore-aft shift.
# ----------------------------------------------------------------------
 
Z_DOWN = -0.10   # crouched depth: feet tucked close to the body
Z_STAND = -0.22  # target standing depth — MUST match STANCE_Z in the gait script
 
assert abs(Z_DOWN) < MAX_REACH and abs(Z_STAND) < MAX_REACH, "depth exceeds leg reach"
 
 
def stand_up(robot_id, leg_joint_indices, duration=2.0, max_force=33.5, hold=0.5):
    """Smoothly interpolate all four feet from Z_DOWN to Z_STAND.
 
    leg_joint_indices: dict like {"FR": (hip_j, thigh_j, calf_j), ...}
    Call this once, right after the robot is loaded and already resting
    in the crouched pose (see set_crouch_pose below), before any gait
    loop starts commanding foot trajectories.
    """
    steps = max(1, int(duration * 240))
    for i in range(steps + 1):
        s = i / steps
        ease = 0.5 * (1 - math.cos(math.pi * s))  # cosine ease-in/ease-out
        z = Z_DOWN + (Z_STAND - Z_DOWN) * ease
 
        for leg, cfg in LEGS.items():
            y = cfg["side"] * L1
            q1, q2, q3 = leg_ik(-CG_OFFSET_X, y, z, cfg["side"])
            hip_j, thigh_j, calf_j = leg_joint_indices[leg]
            for j, q in zip((hip_j, thigh_j, calf_j), (q1, q2, q3)):
                p.setJointMotorControl2(
                    robot_id, j, p.POSITION_CONTROL,
                    targetPosition=q, force=max_force,
                )
 
        p.stepSimulation()
        time.sleep(1.0 / 240.0)
 
    # brief hold at full standing height to let the base settle before
    # a gait loop takes over
    hold_steps = max(1, int(hold * 240))
    for _ in range(hold_steps):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)
 
 
def set_crouch_pose(robot_id, leg_joint_indices, max_force=33.5):
    """Command (and hard-reset) all legs directly into the crouched pose,
    so the robot doesn't free-fall from a taller spawn height before
    stand_up() begins."""
    for leg, cfg in LEGS.items():
        y = cfg["side"] * L1
        q1, q2, q3 = leg_ik(-CG_OFFSET_X, y, Z_DOWN, cfg["side"])
        hip_j, thigh_j, calf_j = leg_joint_indices[leg]
        for j, q in zip((hip_j, thigh_j, calf_j), (q1, q2, q3)):
            p.resetJointState(robot_id, j, targetValue=q)
            p.setJointMotorControl2(
                robot_id, j, p.POSITION_CONTROL,
                targetPosition=q, force=max_force,
            )


# ----------------------------------------------------------------------
# Sim setup
# ----------------------------------------------------------------------

URDF_PATH = "urdf/a1_unitree.urdf"  # adjust to your local path

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(1.0 / 240.0)

plane_id = p.loadURDF("plane.urdf")
start_z = abs(Z_DOWN) + 0.02
robot_id = p.loadURDF(URDF_PATH, [0, 0, start_z], useFixedBase=False)
camera=DepthCamera(robot_id) #depth map stuff does here

joint_name_to_index = {}
for j in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, j)
    joint_name_to_index[info[1].decode("utf-8")] = j

leg_joint_indices = {}
for leg, cfg in LEGS.items():
    pfx = cfg["prefix"]
    leg_joint_indices[leg] = (
        joint_name_to_index[f"{pfx}_hip_joint"],
        joint_name_to_index[f"{pfx}_thigh_joint"],
        joint_name_to_index[f"{pfx}_calf_joint"],
    )

for j in range(p.getNumJoints(robot_id)):
    p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL, force=0)

set_crouch_pose(robot_id, leg_joint_indices)

# let it settle in the crouch for a moment before standing up
for _ in range(int(0.3 * 240)):
    p.stepSimulation()
    time.sleep(1.0 / 240.0)

stand_up(robot_id, leg_joint_indices)

print("Stand-up complete — ready to hand off to the gait loop.")


# Build joint name -> index map (don't hardcode indices)
joint_name_to_index = {}
for j in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, j)
    joint_name_to_index[info[1].decode("utf-8")] = j

leg_joint_indices = {}
for leg, cfg in LEGS.items():
    pfx = cfg["prefix"]
    leg_joint_indices[leg] = (
        joint_name_to_index[f"{pfx}_hip_joint"],
        joint_name_to_index[f"{pfx}_thigh_joint"],
        joint_name_to_index[f"{pfx}_calf_joint"],
    )

imu_link_index = joint_name_to_index["imu_joint"]  # fixed joint; child link = imu_link

# Disable default velocity motors before position control takes over
for j in range(p.getNumJoints(robot_id)):
    p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL, force=0)

MAX_FORCE = 33.5  # from joint dump

# ----------------------------------------------------------------------
# Gait parameters
# ----------------------------------------------------------------------

STANCE_Z = -0.22          # nominal standing height below hip (< MAX_REACH)
STEP_LEN = 0.05        # fore-aft stride length (kept modest: crawl gait
                           # has 3 legs bearing load while one swings, so a
                           # smaller step reduces tip-over risk than a trot)
STEP_HEIGHT = 0.05    # swing foot clearance
CYCLE_TIME = 2.0    # seconds per full gait cycle (slower than a trot,
                           # since each leg's window is a quarter-cycle)

assert abs(STANCE_Z) < MAX_REACH, "STANCE_Z exceeds leg reach"

GYRO_PRINT_INTERVAL = 0.5  # seconds, just for visibility during dev
_last_gyro_print = 0.0




# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------


t0 = time.time()
init_frames = 10
try:
    for i in  range(init_frames):
        p.stepSimulation()
    time.sleep(1.5)
    print("starting up!....")
    while True:
        t = time.time() - t0
        cycle_phase = (t / CYCLE_TIME) % 1.0

        for leg, cfg in LEGS.items():
            local_phase = (cycle_phase + PHASE_OFFSET[leg]) % 1.0
            x, y, z = foot_target(local_phase, cfg["side"], STEP_LEN, STEP_HEIGHT, STANCE_Z)
            q1, q2, q3 = leg_ik(x, y, z, cfg["side"])

            hip_j, thigh_j, calf_j = leg_joint_indices[leg]
            for j, q in zip((hip_j, thigh_j, calf_j), (q1, q2, q3)):
                p.setJointMotorControl2(
                    robot_id, j, p.POSITION_CONTROL,
                    targetPosition=q, force=MAX_FORCE,
                )

        p.stepSimulation()
        #depth camera code
        depth=camera.capture()
        camera.show(depth)
        #depth camera code ends

        if t - _last_gyro_print > GYRO_PRINT_INTERVAL:
            roll_rate, pitch_rate, yaw_rate = read_gyro(robot_id, imu_link_index)
            print(f"gyro (rad/s)  roll={roll_rate:+.3f}  pitch={pitch_rate:+.3f}  yaw={yaw_rate:+.3f}")
            _last_gyro_print = t

        time.sleep(1.0 / 240.0)
except KeyboardInterrupt:
    pass
finally:
    p.disconnect()