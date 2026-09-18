import pybullet as p
import pybullet_data
import time


physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -9.81)

plane_id = p.loadURDF("plane.urdf")

a1_id = p.loadURDF(
    "urdf/a1_unitree.urdf",
    basePosition=[0, 0, 0.5],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    useFixedBase=False
)

print("A1 loaded with ID:", a1_id)
n_joints = p.getNumJoints(a1_id)

for i in range(n_joints):
    print(p.getJointInfo(a1_id, i))
while True:
    p.stepSimulation()
    time.sleep(1 / 240)