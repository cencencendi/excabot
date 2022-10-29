import pybullet as p
import time
import pybullet_data
import math
import numpy as np
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
 
planeId = p.loadURDF("plane.urdf")
 
startPos = [0, 0, 1.4054411813121799]
startOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF("aba_excavator/excavator.urdf",startPos, startOrientation)

for i in range(1000):
    p.setJointMotorControl2(boxId, 2 , p.POSITION_CONTROL, targetPosition = -0.4, force= 250_000)
    p.setJointMotorControl2(boxId, 3 , p.POSITION_CONTROL, targetPosition = -0.1214, force= 250_000)
    p.setJointMotorControl2(boxId, 4 , p.POSITION_CONTROL, targetPosition = -0.32)
    (linkWorldPosition,
            linkWorldOrientation,
            localInertialFramePosition,
            localInertialFrameOrientation,
            worldLinkFramePosition,
            worldLinkFrameOrientation,
            worldLinkLinearVelocity,
            worldLinkAngularVelocity) = p.getLinkState(boxId,4, computeLinkVelocity=1, computeForwardKinematics=1)
    print(linkWorldPosition)    
    p.stepSimulation()
    time.sleep(1.0/240.)

