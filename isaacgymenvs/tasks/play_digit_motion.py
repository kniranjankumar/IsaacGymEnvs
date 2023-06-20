import math
import numpy as np
from isaacgym import gymapi, gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.amp.utils_amp.motion_lib import MotionLib
import scipy.optimize
import pybullet as p
import pybullet_data
import sys
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setGravity(0,0,-9.81)
p.setRealTimeSimulation(0)
plane_id = p.loadURDF("plane.urdf",[0,0,0],[0,0,0,1])
digit_id = p.loadURDF("../assets/urdf/DigitRobot/DigitRobot/urdf/digit_model.urdf",[0,0,2.1],[0,0,0,1])
poses = np.load("retargeted_poses_digit_v2.npy")
for joint in range(p.getNumJoints(digit_id)):
    p.setJointMotorControl2(digit_id, joint, p.POSITION_CONTROL, targetVelocity=1, force=10)
idx = 0
height_offset = 0.3
while True:
    for i in range(p.getNumJoints(digit_id)):
        jointInfo = p.getJointInfo(digit_id, i)
        qIndex = jointInfo[3]
        if qIndex > -1:
                p.resetJointState(digit_id,i,poses[idx][7:][qIndex-7])
    p.resetBasePositionAndOrientation(digit_id, poses[idx][:3]+np.array([0,0,height_offset]), poses[idx][3:7])
    p.stepSimulation()
    idx +=1
    time.sleep(0.001)
    if idx >= len(poses):
        idx = 0
        # break
    
    
