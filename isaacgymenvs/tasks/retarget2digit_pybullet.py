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


def retarget_humanoid_to_digit_ik(marker_positions, digit_eff_idxs, digit_id, initPose):
    # print("marker_positions", marker_positions, "digit_eff_idxs", digit_eff_idxs)
    # pos, ori = p.getLinkState(digit_id, digit_eff_idxs[0])[4:6]
    # pos = marker_positions[2]
    # print(pos, p.getLinkState(digit_id, digit_eff_idxs[0])[4])
    # print("gonna IK")
    # time.sleep(1)
    curr_orientation = [p.getLinkState(digit_id, digit_eff_idx)[5] for digit_eff_idx in digit_eff_idxs]
    curr_orientation = np.array(curr_orientation)
    # curr_orientation[0,:] = np.array([0,0,0,1])
    # curr_orientation[1,:] = np.array([0,0,0,1])
    # curr_orientation[2,:] = np.array([0,0,0,1])
    # curr_orientation[3,:] = np.array([0,0,0,1])
    joint_pose = p.calculateInverseKinematics2(digit_id, digit_eff_idxs,
                                                    marker_positions,
                                                    curr_orientation,
                                                    # lowerLimits=[-1]*22,
                                                    # upperLimits=[1]*22,
                                                    restPoses=initPose,
                                                    solver=p.IK_DLS,
                                                    maxNumIterations=1000,
                                                    residualThreshold=1e-8)
    # time.sleep(1)
    # print("IK done")
    joint_pose = np.array(joint_pose)
    # joint_pose[:20]=0
    # joint_pose[27:]=0
    return  joint_pose

def get_error(marker_id, digit_eff_idxs, digit_id):
    eff_pos = [p.getLinkState(digit_id, idx)[4] for idx in digit_eff_idxs]
    eff_pos = np.array(eff_pos)
    marker_positions = np.array([p.getBasePositionAndOrientation(marker)[0] for marker in marker_id])
    return np.linalg.norm(eff_pos-marker_positions, axis=1)
# def retarget_humanoid_to_franka_ik(marker_positions, franka_eff_idxs, franka_id, initPose):
#     # print("marker_positions", marker_positions, "digit_eff_idxs", franka_eff_idxs)
#     # pos, ori = p.getLinkState(franka_id, franka_eff_idxs[0])[4:6]
#     pos = marker_positions[2]
#     # print(pos, p.getLinkState(franka_id, franka_eff_idxs[0])[4])
#     joint_pose = p.calculateInverseKinematics(franka_id, franka_eff_idxs[2],
#                                                     np.array(pos),
#                                                     # lowerLimits=[-1]*22,
#                                                     # upperLimits=[1]*22,
#                                                     # restPoses=initPose,
#                                                     solver=p.IK_DLS,
#                                                     maxNumIterations=100,
#                                                     residualThreshold=1e-5)
    return np.array(joint_pose)

def create_markers(humanoid_id, body_idxs, radius=0.05,colors=[[1,0,0,1], [0,1,0,1], [0,0,1,1], [1,1,0,1]]):
    markers = []
    for color, body_idx in zip(colors,body_idxs):
        pos, ori = p.getLinkState(humanoid_id, body_idx)[4:6]
        offset = np.array([0,-1,0.2])
        marker = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        marker_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=marker, basePosition=np.array(pos)+offset, baseOrientation=ori)
        markers.append(marker_id)
    return markers
    

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setGravity(0,0,-9.81)
p.setRealTimeSimulation(0)
# p.setPhysicsEngineParameter(numSolverIterations=150)
# p.setTimeStep(1./240.)
plane_id = p.loadURDF("plane.urdf",[0,0,0],[0,0,0,1])
digit_id = p.loadURDF("../assets/urdf/DigitRobot/DigitRobot/urdf/digit_model.urdf",[0,0,2.1],[0,0,0,1])
humanoid_id = p.loadMJCF("../assets/mjcf/amp_humanoid_pybullet.xml")[0]
# franka_id = p.loadURDF("../assets/urdf/franka_description/robots/franka_panda.urdf",[2,0,0],[0,0,0,1])
# humanoid_id = humanoid_ids[1]
# print("humanoid_ids", humanoid_ids)
motion_file = "/home/dan/Projects/dynamic_motion_imitation/IsaacGymEnvs/assets/amp/motions/amp_humanoid_walk.npy"
mocap = MotionLib(motion_file=motion_file, 
                                     num_dofs=28,
                                     key_body_ids=torch.tensor([5,8,11,14],device=torch.device("cuda:0")), 
                                     device=torch.device("cuda:0"))
length = 3000
print(mocap, mocap.num_motions(),mocap._motion_lengths[0])
motion_times0 = mocap.sample_time(np.array([0]))
motion_times0 = [1.0]
print("motion_time 0:",motion_times0)
motion_times = np.expand_dims(motion_times0, axis=-1)
time_steps = 0.005 * np.arange(0, length)
motion_times = motion_times + time_steps
motion_ids = np.array([0]*length).flatten()
motion_times = motion_times.flatten()

frame = mocap.get_motion_state(np.array([0]*length), motion_times)
root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = frame
digit_dofs = retarget_humanoid_to_digit(dof_pos[0].cpu().numpy())
print("num frames", len(root_pos))

mapped_joints = [
        "hip_abduction_left",
        "hip_rotation_left",
        "hip_flexion_left",
        "knee_joint_left",
        "shin_to_tarsus_left",
        "toe_pitch_joint_left",
        "toe_roll_joint_left",
        "hip_abduction_right",
        "hip_rotation_right",
        "hip_flexion_right",
        "knee_joint_right",
        "shin_to_tarsus_right",
        "toe_pitch_joint_right",
        "toe_roll_joint_right",
        "shoulder_roll_joint_left",
        "shoulder_pitch_joint_left",
        "shoulder_yaw_joint_left",
        "elbow_joint_left",
        "shoulder_roll_joint_right",
        "shoulder_pitch_joint_right",
        "shoulder_yaw_joint_right",
        "elbow_joint_right"
]
pybullet_joints = [(p.getJointInfo(digit_id, i)[1]).decode() for i in range(p.getNumJoints(digit_id))]
mapping = {}
humanoid_joints = [0,1,2,4,5,6,8,9,10,12,15,16,17,19,22,23, 24, 26, 28, 29, 30, 32, 33, 34, 36, 38, 39, 40]
for i in range(len(mapped_joints)):
    mapping[pybullet_joints.index(mapped_joints[i])] = i
print(mapping)
count = 0
for joint in range(p.getNumJoints(digit_id)):
    if p.getJointInfo(digit_id, joint)[2]==0:
        print(count, p.getJointInfo(digit_id, joint)[1])
        count+=1
    p.setJointMotorControl2(digit_id, joint, p.POSITION_CONTROL, targetVelocity=1, force=10)
for joint in range(p.getNumJoints(humanoid_id)):
        # print(joint, p.getJointInfo(humanoid_id, joint))
        p.setJointMotorControl2(humanoid_id, joint, p.POSITION_CONTROL, targetVelocity=1, force=10)
# for joint in range(p.getNumJoints(franka_id)):
#         # print(joint, p.getJointInfo(humanoid_id, joint))
#         p.setJointMotorControl2(franka_id, joint, p.POSITION_CONTROL, targetVelocity=1, force=10)
idx = 0
# collisionFilterGroup = 0x2
# collisionFilterMask = 0x2
# for i in range(p.getNumJoints(humanoid_id)):
#     p.setCollisionFilterGroupMask(humanoid_id,i,collisionFilterGroup,collisionFilterMask)

# for i in range(p.getNumJoints(humanoid_ids[0])):
#     p.setCollisionFilterGroupMask(humanoid_ids[0],i,collisionFilterGroup,collisionFilterMask)


# collisionFilterGroup2 = 0x1
# collisionFilterMask2 = 0x1 
# for i in range(p.getNumJoints(digit_id)):
#     p.setCollisionFilterGroupMask(digit_id,i,collisionFilterGroup2,collisionFilterMask2)
for i in range(p.getNumJoints(humanoid_id)):
    for j in range(p.getNumJoints(digit_id)):
        p.setCollisionFilterPair(humanoid_id, digit_id, i, j, 0)
    # for j in range(p.getNumJoints(humanoid_ids[0])):
    #     p.setCollisionFilterPair(humanoid_ids[0], digit_id, i, j, 0)
joint_types = {0: p.JOINT_REVOLUTE, 1: p.JOINT_PRISMATIC, 2: p.JOINT_SPHERICAL, 3: p.JOINT_PLANAR, 4: p.JOINT_FIXED}
for i in range(29):
    info = p.getJointInfo(digit_id, i)
    print(info[12], info[1], joint_types[info[2]])
#     print(i, p.getBodyInfo(digit_id, i))
markers = None
count = 0
digit_dofs = retarget_humanoid_to_digit(dof_pos[idx].cpu().numpy())*0
# digit_dofs[9] = 1.4
# digit_dofs[11] = 1.7
# digit_dofs[21] = -1.7
# digit_dofs[14] = -0.4
# digit_dofs[0] = 0.4

p.setJointMotorControlArray(digit_id, list(mapping.keys()), p.POSITION_CONTROL, targetPositions=digit_dofs)
humanoid_eff = [14, 21, 31, 41]
digit_eff = [26, 12, 21, 7]
count = 0
for digit_eff_id, humanoid_eff_id in zip(digit_eff, humanoid_eff):
    print(p.getJointInfo(digit_id, digit_eff_id), p.getJointInfo(humanoid_id, humanoid_eff_id))
p.resetBasePositionAndOrientation(humanoid_id, root_pos[idx].cpu().numpy(), root_rot[idx].cpu().numpy())
p.resetBasePositionAndOrientation(digit_id, root_pos[idx].cpu().numpy()+np.array([0,-1,0.5]), root_rot[idx].cpu().numpy())
# p.resetBasePositionAndOrientation(franka_id, np.array([0.5,-1,0.0]), [0,0,0,1])
p.createConstraint(digit_id, -1, plane_id, -1, p.JOINT_FIXED, [0,0,0], [0,0,0], root_pos[idx].cpu().numpy()+np.array([0,-1,0.5]),[0,0,0,1],root_rot[idx].cpu().numpy())
p.createConstraint(humanoid_id, -1, plane_id, -1, p.JOINT_FIXED, [0,0,0], [0,0,0], root_pos[idx].cpu().numpy()+np.array([0,0,0.05]),[0,0,0,1],root_rot[idx].cpu().numpy())
#kness constraints
global_com_knee_l = np.array(p.getLinkState(digit_id, 4)[4:6])
global_com_knee_r = np.array(p.getLinkState(digit_id, 18)[4:6])
global_com_tarsus_l = np.array(p.getLinkState(digit_id, 5)[4:6])
global_com_tarsus_r = np.array(p.getLinkState(digit_id, 19)[4:6])
global_com_toe_l = np.array(p.getLinkState(digit_id, 6)[4:6])
global_com_toe_r = np.array(p.getLinkState(digit_id, 20)[4:6])

rknee_offset = np.array([-0.02, 0.1, 0.0])
lknee_offset = np.array([-0.02, -0.1, 0.0])
rtarsus_offset2 = np.array([-0.1,0.01,0])
ltarsus_offset2 = np.array([-0.1,-0.01,0])
rtoe_offset = np.array([-0.049,0.01,0.0])
rtarsus_offset = np.array([0.11,0.085,0])
ltoe_offset = np.array([-0.049,-0.01,0.0])
ltarsus_offset = np.array([0.11,-0.085,0])

local_com_knee_l = np.array(p.getLinkState(digit_id, 4)[2:4])
local_com_knee_r = np.array(p.getLinkState(digit_id, 18)[2:4])
local_com_tarsus_l = np.array(p.getLinkState(digit_id, 5)[2:4])
local_com_tarsus_r = np.array(p.getLinkState(digit_id, 19)[2:4])
local_com_toe_l = np.array(p.getLinkState(digit_id, 6)[2:4])
local_com_toe_r = np.array(p.getLinkState(digit_id, 20)[2:4])

com2offset_knee_l = p.multiplyTransforms(*p.invertTransform(*local_com_knee_l),lknee_offset,[0,0,0,1])
com2offset_tarsus2_l = p.multiplyTransforms(*p.invertTransform(*local_com_tarsus_l),ltarsus_offset2,[0,0,0,1])
com2offset_knee_r = p.multiplyTransforms(*p.invertTransform(*local_com_knee_r),rknee_offset,[0,0,0,1])
com2offset_tarsus2_r = p.multiplyTransforms(*p.invertTransform(*local_com_tarsus_r),rtarsus_offset2,[0,0,0,1])

com2offset_toe_l = p.multiplyTransforms(*p.invertTransform(*local_com_toe_l),ltoe_offset,[0,0,0,1])
com2offset_tarsus_l = p.multiplyTransforms(*p.invertTransform(*local_com_tarsus_l),ltarsus_offset,[0,0,0,1])
com2offset_toe_r = p.multiplyTransforms(*p.invertTransform(*local_com_toe_r),rtoe_offset,[0,0,0,1])
com2offset_tarsus_r = p.multiplyTransforms(*p.invertTransform(*local_com_tarsus_r),rtarsus_offset,[0,0,0,1])

p.addUserDebugLine(p.multiplyTransforms(global_com_knee_r[0], global_com_knee_r[1],rknee_offset,[0,0,0,1])[0] ,
                p.multiplyTransforms(global_com_tarsus_r[0], global_com_tarsus_r[1],rtarsus_offset2,[0,0,0,1])[0], [1,0,0])

p.addUserDebugLine(p.multiplyTransforms(global_com_toe_r[0], global_com_toe_r[1],rtoe_offset,[0,0,0,1])[0] ,
                p.multiplyTransforms(global_com_tarsus_r[0], global_com_tarsus_r[1],rtarsus_offset,[0,0,0,1])[0], [1,0,0])

c1 = p.createConstraint(digit_id, 4, digit_id, 5, p.JOINT_POINT2POINT, [0,0,0], com2offset_knee_l,com2offset_tarsus2_l)
c2 = p.createConstraint(digit_id, 5, digit_id, 6, p.JOINT_POINT2POINT, [0,0,0], com2offset_tarsus_l, com2offset_toe_l)
c3 = p.createConstraint(digit_id, 18, digit_id, 19, p.JOINT_POINT2POINT, [0,0,0], com2offset_knee_r,com2offset_tarsus2_r)
c4 = p.createConstraint(digit_id, 19, digit_id, 20, p.JOINT_POINT2POINT, [0,0,0], com2offset_tarsus_r, com2offset_toe_r)
p.changeConstraint(c1, maxForce=1000)
p.changeConstraint(c2, maxForce=1000)
p.changeConstraint(c3, maxForce=1000)
p.changeConstraint(c4, maxForce=1000)


# c2 = p.createConstraint(digit_id, 12, digit_id, 14, p.JOINT_POINT2POINT, [0,0,0], [-0.02-0.04, 0.1+0.04, 0.0],[-0.1,0.01-0.029,0])
# toe constraints
# c3 =p.createConstraint(digit_id, 6, digit_id, 7, p.JOINT_POINT2POINT, [0,0,0], [0.11,-0.085+0.029,0],[-0.049, -0.01,0.0])
# c4 = p.createConstraint(digit_id, 14, digit_id, 15, p.JOINT_POINT2POINT, [0,0,0], [0.11,0.085-0.029,0],[-0.049,0.01,0.0])
# p.setConstraintEnable(c1, 1)
# p.setConstraintEnable(c2, 1)
# p.setConstraintEnable(c3, 1)
# p.setConstraintEnable(c4, 1)

retargetted_poses = []
body_orientations = []
error = []
time.sleep(5)
while True:
    # idx +=1
    # time.sleep(0.1)
    if idx >= len(root_pos)-1000:
        idx = 0
        break
    
    # [print(i, p.getJointInfo(humanoid_id, i)[1]) for i in range(p.getNumJoints(humanoid_id))]
    # print(p.getNumJoints(digit_id), len(digit_dofs))
    # mapped_joints=[digit_dofs[mapping[i]] for i in range(len(mapping))]
    # print(len(dof_pos[0].cpu().numpy())), print(len(list(mapping.values())), p.getNumJoints(humanoid_id))
    p.setJointMotorControlArray(humanoid_id, humanoid_joints, p.POSITION_CONTROL, targetPositions=dof_pos[idx].cpu().numpy())
    
    p.stepSimulation()
    count +=1
    # if markers is not None:
    #     # print(digit_dofs.shape)
    #     p.setJointMotorControlArray(digit_id, [i for i in range(22)], p.POSITION_CONTROL, targetPositions=digit_dofs)
        
    if markers is None and count ==100:
        markers_humanoid = create_markers(humanoid_id, body_idxs=humanoid_eff, radius=0.05, colors=[[1,0,0,1], [0,1,0,1], [0,0,1,1], [1,1,0,1]])
        markers_digit = create_markers(digit_id, body_idxs=digit_eff, radius=0.05, colors=[[1,0,0,1], [0,1,0,1], [0,0,1,1], [1,1,0,1]])
        
    if count >=1000:
        for i, marker in enumerate(markers_humanoid):
            offset = np.array([0,-1,0.2])
            pos = p.getLinkState(humanoid_id, humanoid_eff[i])[4]
            p.resetBasePositionAndOrientation(marker, np.array(pos)+offset, [0,0,0,1])
        for i, marker in enumerate(markers_digit):
            pos,_ = p.getLinkState(digit_id, digit_eff[i])[4:6]
            p.resetBasePositionAndOrientation(marker, pos, [0,0,0,1])
        if get_error(markers_digit, digit_eff, digit_id).sum()<0.0001:
            # digit_dofs = 
            digit_dofs = retarget_humanoid_to_digit_ik([p.getBasePositionAndOrientation(marker)[0] for marker in markers_humanoid], 
                                                    digit_eff,
                                                    digit_id,
                                                    p.getJointStates(digit_id,[i for i in range(p.getNumJoints(digit_id)) if p.getJointInfo(digit_id, i)[3]>-1])[0])
            numJoints = p.getNumJoints(digit_id)
            # for i in range(numJoints):
            #     jointInfo = p.getJointInfo(digit_id, i)
            #     qIndex = jointInfo[3]
            #     if qIndex > -1:
            #         # print(i, qIndex, digit_dofs[qIndex-7])
            #             p.resetJointState(digit_id,i,digit_dofs[qIndex-7])
            if get_error(markers_digit, digit_eff, digit_id).sum()<0.00001:
                for i in range(numJoints):
                    jointInfo = p.getJointInfo(digit_id, i)
                    qIndex = jointInfo[3]
                    if qIndex > -1:
                        # print(i, qIndex, digit_dofs[qIndex-7])
                            p.resetJointState(digit_id,i,digit_dofs[qIndex-7])
            retargetted_poses.append(np.concatenate((root_pos[idx].cpu().numpy(),root_rot[idx].cpu().numpy(), digit_dofs)))
            # print(retargetted_poses[0])
            # assert False
            body_orientations.append({p.getJointInfo(digit_id, digit_eff_idx)[12].decode():p.getLinkState(digit_id, digit_eff_idx)[5] for digit_eff_idx in range(30)})
            # print("link names:", [p.getJointInfo(digit_id, i)[12].decode() for i in range(p.getNumJoints(digit_id))])
            # print(orientations)
            error.append(get_error(markers_digit, digit_eff, digit_id))
            # print("error:",error[-1])
        idx +=1
    # time.sleep(0.01)
# np.save("retargeted_poses_digit_v3.npy", retargetted_poses)
np.save("retargetted_digit_body_orientations.npy", body_orientations)
error = np.array(error)**2
# np.save("error_digit_v2.npy", error)
fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0,0].plot(error[:,0])
axes[0,1].plot(error[:,1])
axes[1,0].plot(error[:,2])
axes[1,1].plot(error[:,3])
axes[0,0].set_title("squared error right hand")
axes[0,1].set_title("squared error left hand")
axes[1,0].set_title("squared error right foot")
axes[1,1].set_title("squared error left foot")
plt.show()

    
    # p.setJointMotorControlArray(digit_id, [i for i in range(22)], p.POSITION_CONTROL, targetPositions=digit_dofs)
    
        # print("num_joints", p.getNumJoints(franka_id))
        # franka_dofs = retarget_humanoid_to_franka_ik([p.getBasePositionAndOrientation(marker)[0] for marker in markers],
        #                                              [9],
        #                                             franka_id,
        #                                             p.getJointStates(franka_id,[i for i in range(p.getNumJoints(franka_id))])[0])
        # # print(franka_dofs.shape)
        # p.setJointMotorControlArray(franka_id, [i for i in range(p.getNumJoints(franka_id)-1)], p.POSITION_CONTROL, targetPositions=franka_dofs)
        
                                                   