import pybullet as p
import pybullet_data
import time


physics_client = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -9.81)

plane_id = p.loadURDF("plane.urdf")

robot_id = p.loadURDF(
    "urdf/a1_unitree.urdf",
    basePosition=[0, 0, 0.5],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    useFixedBase=False
)


joint_type_names = {
    p.JOINT_REVOLUTE: "REVOLUTE",
    p.JOINT_PRISMATIC: "PRISMATIC",
    p.JOINT_SPHERICAL: "SPHERICAL",
    p.JOINT_PLANAR: "PLANAR",
    p.JOINT_FIXED: "FIXED",
}


print("\n" + "=" * 100)
print("ROBOT LINK AND JOINT INFORMATION")
print("=" * 100)

print(f"Robot ID: {robot_id}")
print(f"Number of joints/links: {p.getNumJoints(robot_id)}")

print("\nBASE LINK")
print("-" * 100)
print("Link Index : -1")
print("Link Name  : BASE")
print("Base Pos   :", p.getBasePositionAndOrientation(robot_id)[0])
print("Base Orient:", p.getBasePositionAndOrientation(robot_id)[1])


for joint_index in range(p.getNumJoints(robot_id)):

    joint_info = p.getJointInfo(robot_id, joint_index)

    joint_index = joint_info[0]
    joint_name = joint_info[1].decode("utf-8")
    joint_type = joint_info[2]

    q_index = joint_info[3]
    u_index = joint_info[4]

    flags = joint_info[5]

    damping = joint_info[6]
    friction = joint_info[7]

    lower_limit = joint_info[8]
    upper_limit = joint_info[9]

    max_force = joint_info[10]
    max_velocity = joint_info[11]

    joint_axis = joint_info[13]

    parent_frame_position = joint_info[14]
    parent_frame_orientation = joint_info[15]

    parent_index = joint_info[16]

    link_name = joint_info[12].decode("utf-8")

    joint_type_name = joint_type_names.get(
        joint_type,
        f"UNKNOWN ({joint_type})"
    )

    joint_state = p.getJointState(robot_id, joint_index)

    current_position = joint_state[0]
    current_velocity = joint_state[1]
    reaction_forces = joint_state[2]
    applied_motor_torque = joint_state[3]

    print("\n" + "=" * 100)

    print(f"JOINT / LINK {joint_index}")
    print("-" * 100)

    print(f"Joint Index              : {joint_index}")
    print(f"Joint Name               : {joint_name}")
    print(f"Joint Type               : {joint_type_name}")

    print()

    print(f"Link Index               : {joint_index}")
    print(f"Link Name                : {link_name}")
    print(f"Parent Link Index        : {parent_index}")

    print()

    print(f"Q Index                  : {q_index}")
    print(f"U Index                  : {u_index}")
    print(f"Flags                    : {flags}")

    print()

    print(f"Joint Axis               : {joint_axis}")

    print(
        f"Parent Frame Position   : "
        f"{parent_frame_position}"
    )

    print(
        f"Parent Frame Orientation: "
        f"{parent_frame_orientation}"
    )

    print()

    print(f"Lower Position Limit     : {lower_limit}")
    print(f"Upper Position Limit     : {upper_limit}")

    print()

    print(f"Damping                  : {damping}")
    print(f"Friction                 : {friction}")

    print()

    print(f"Maximum Force             : {max_force}")
    print(f"Maximum Velocity          : {max_velocity}")

    print()

    print(f"Current Position          : {current_position}")
    print(f"Current Velocity          : {current_velocity}")
    print(f"Reaction Forces           : {reaction_forces}")
    print(f"Applied Motor Torque      : {applied_motor_torque}")


print("\n" + "=" * 100)
print("END OF ROBOT INFORMATION")
print("=" * 100)

input("\nPress Enter to exit...")

p.disconnect()