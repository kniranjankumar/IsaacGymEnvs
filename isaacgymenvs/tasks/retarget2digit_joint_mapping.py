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
import matplotlib.pyplot as plt
import pybullet_data

def retarget_humanoid_to_digit(humanoid_dof):
    joint_mapping_digit2humanoid = [21,23,22,24,25,26,25,14,16,15,17,18,19,18,10,11,12,13,6,7,8,9] 
    digit2humanoid_offset = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    digit2humanoid_direction = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    # right hand
    digit2humanoid_direction[18]*=-1
    digit2humanoid_direction[19]*=-1
    digit2humanoid_direction[20]*=-1
    digit2humanoid_direction[21]*=-1
    # left hand
    digit2humanoid_direction[14]*=-1
    # digit2humanoid_direction[15]*=-1
    digit2humanoid_direction[16]*=-1
    # digit2humanoid_direction[17]*=1
    # right leg
    digit2humanoid_direction[7]*=-1
    digit2humanoid_direction[8]*=-1
    digit2humanoid_direction[9]*=-1
    digit2humanoid_direction[10]*=1
    # digit2humanoid_direction[13]*=-1
    # digit2humanoid_direction[11]*=-1
    
    # left leg
    digit2humanoid_direction[0]*=-1
    digit2humanoid_direction[1]*=-1 
    # digit2humanoid_direction[2]*=-1
    digit2humanoid_direction[3]*=-1
    digit2humanoid_direction[5]*=-1
    # digit2humanoid_direction[5]*=-1
    
    # digit2humanoid_direction[0]*=-1
    # digit2humanoid_direction[4]*=-1
    
    digit2humanoid_offset[21] = -1.4
    digit2humanoid_offset[17] = 1.4
    digit2humanoid_offset[19] = -0.75
    digit2humanoid_offset[15] = 0.75
    digit2humanoid_offset[7] = -0.3
    digit2humanoid_offset[0] = +0.3
    digit_dof = np.array(humanoid_dof[joint_mapping_digit2humanoid])*np.array(digit2humanoid_direction)+np.array(digit2humanoid_offset)
    digit_dof[11] = 0
    digit_dof[4] = 0
    return digit_dof

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setRealTimeSimulation(0)
plane_id = p.loadURDF("plane.urdf",[0,0,0],[0,0,0,1])
digit_id = p.loadURDF("../assets/mjcf/digit_description/urdf/digit_fixed_closed_loop.urdf",[0,0,2.1],[0,0,0,1])
# digit_id = p.loadMJCF("../assets/mjcf/digit_description/digit-v3/digit-v3-bullet.xml")[0]

humanoid_id = p.loadMJCF("../assets/mjcf/amp_humanoid_pybullet.xml")[0]
duck_id = p.loadURDF("duck_vhacd.urdf",[2,0,1],[0,0,0,1], useFixedBase=True, globalScaling=5.0)

humanoid_eff_names = ["right_hand", "left_hand", "right_foot", "left_foot"]
humanoid_bodies = [p.getJointInfo(humanoid_id, i)[12].decode() for i in range(p.getNumJoints(humanoid_id))]
humanoid_eff = [humanoid_bodies.index(name) for name in humanoid_eff_names]

digit_eff_names = ["right_hand", "left_hand", "right_toe_roll", "left_toe_roll"]
digit_bodies = [p.getJointInfo(digit_id, i)[12].decode() for i in range(p.getNumJoints(digit_id))]
digit_eff = [digit_bodies.index(name) for name in digit_eff_names]


motion_file = "amp_humanoid_hop.npy"
motion_file_path = "/home/dan/Projects/dynamic_motion_imitation/IsaacGymEnvs/assets/amp/motions/"+motion_file
mocap = MotionLib(motion_file=motion_file_path, 
                                     num_dofs=28,
                                     key_body_ids=torch.tensor([5,8,11,14],device=torch.device("cuda:0")), 
                                     device=torch.device("cuda:0"))

length = 2700
print(mocap, mocap.num_motions(),mocap._motion_lengths[0])
motion_times0 = mocap.sample_time(np.array([0]))
motion_times0 = [0.0]
print("motion_time 0:",motion_times0)
motion_times = np.expand_dims(motion_times0, axis=-1)
time_steps = 0.005 * np.arange(0, length)
motion_times = motion_times + time_steps
motion_ids = np.array([0]*length).flatten()
motion_times = motion_times.flatten()

frame = mocap.get_motion_state(np.array([0]*length), motion_times)
root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = frame
digit_dofs = retarget_humanoid_to_digit(dof_pos[0].cpu().numpy())

# for idx in range(len(root_pos)):
while True:
    # p.setJointMotorControlArray(humanoid_id, humanoid_joints, p.POSITION_CONTROL, targetPositions=dof_pos[idx].cpu().numpy())    
    
    p.stepSimulation()