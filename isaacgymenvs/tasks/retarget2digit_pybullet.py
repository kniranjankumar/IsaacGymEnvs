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
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("motion_file", help="path to the motion file", type=str, default="amp_humanoid_hop.npy")
# args = parser.parse_args()

# motion_file = args.motion_file
motion_file = "amp_humanoid_hop.npy"
class DebugDuckie:
    def __init__(self, position, body_id, link_id):
        self.position = position
        self.body_id = body_id
        self.link_id = link_id
        self.duck_id = p.loadURDF("duck_vhacd.urdf",self.position,[0,0,0,1], useFixedBase=True, globalScaling=5.0)
        self.base_rotation = p.getQuaternionFromEuler([np.pi/2,0,np.pi])
        orientation = p.getLinkState(self.body_id, self.link_id)[5]
        _, duckie_orientation = p.multiplyTransforms([0,0,0], orientation, [0,0,0], self.base_rotation)
        p.resetBasePositionAndOrientation(self.duck_id, self.position, duckie_orientation)


    
    def update(self, orientation=None):
        if orientation is None:
            orientation = p.getLinkState(self.body_id, self.link_id)[5]
        _, duckie_orientation = p.multiplyTransforms([0,0,0], orientation, [0,0,0], self.base_rotation)
        p.resetBasePositionAndOrientation(self.duck_id, self.position, duckie_orientation)

def compute_effector_pos_error(agent_id, body_idxs, ddof_pose, eff_pose_target, initPose=None):
    joint_body_ids = [digit_body_idx for digit_body_idx in range(30) if p.getJointInfo(digit_id, digit_body_idx)[2]==0]
    pos, ori = p.getBasePositionAndOrientation(agent_id)
    if initPose is None:
        for idx, pose in zip(joint_body_ids, ddof_pose):
            p.resetJointState(agent_id, idx, pose)
    else:
        for idx, dpose, init_pose_ in zip(joint_body_ids, ddof_pose, initPose):
            p.resetJointState(agent_id, idx, dpose+init_pose_)
    p.resetBasePositionAndOrientation(agent_id, pos, ori)
    eff_pos = [p.getLinkState(agent_id, body_idx)[4] for body_idx in body_idxs]
    eff_pos = np.array(eff_pos)
    error = np.linalg.norm(eff_pos-eff_pose_target, axis=1)
    error = np.linalg.norm(error)+ np.linalg.norm(ddof_pose)*0.1 if not initPose is None else 0
    # print("error", error)
    return error

def retarget_humanoid_to_digit_ik_constrained(marker_positions, marker_orientations, digit_eff_idxs, digit_id, initPose):
    objective = lambda x: compute_effector_pos_error(digit_id, digit_eff_idxs, x, marker_positions, initPose)
    # for i in range(10):
    if initPose is None:
        initPose_ = [p.getJointState(digit_id, digit_body_idx)[0] for digit_body_idx in range(30) if p.getJointInfo(digit_id, digit_body_idx)[2]==0]

        output = scipy.optimize.minimize(objective, 
                                       np.array(initPose_), method='SLSQP', jac='3-point', tol=1e-6)#, bounds=[(-1.57,1.57)]*22)
    else:
        output = scipy.optimize.minimize(objective, 
                                       np.array(initPose), method='SLSQP', jac='3-point', tol=1e-6, bounds=[(-0.1,0.1)]*22)

    return output.x

def retarget_humanoid_to_digit_ik(marker_positions, marker_orientations, digit_eff_idxs, digit_id, initPose):
    # curr_orientation = [p.getLinkState(digit_id, digit_eff_idx)[5] for digit_eff_idx in digit_eff_idxs]
    # curr_orientation = np.array(curr_orientation)
    # curr_orientation[2,:] = marker_orientations[2]
    # curr_orientation[3,:] = marker_orientations[3]
    joint_pose = p.calculateInverseKinematics2(digit_id, digit_eff_idxs,
                                                    marker_positions,
                                                    marker_orientations,
                                                    # lowerLimits=[-1]*22,
                                                    # upperLimits=[1]*22,
                                                    restPoses=initPose,
                                                    # solver=p.IK_DLS,
                                                    # maxNumIterations=1000,
                                                    # residualThreshold=1e-8,
                                                    jointDamping=[0.01]*22)
    # joint_pose = [0]*p.getNumJoints(digit_id)
    joint_pose = np.array(joint_pose)
    # joint_pose*=0
    joint_pose_left_leg = p.calculateInverseKinematics(digit_id, digit_eff_idxs[3],
                                                        marker_positions[3],
                                                        marker_orientations[3],
                                                        restPoses=initPose)
    joint_pose_right_leg = p.calculateInverseKinematics(digit_id, digit_eff_idxs[2],
                                                        marker_positions[2],
                                                        marker_orientations[2],
                                                        restPoses=initPose)
    joint_pose_left_arm = p.calculateInverseKinematics(digit_id, digit_eff_idxs[1],
                                                        marker_positions[1],
                                                        marker_orientations[1],
                                                        restPoses=initPose)
    joint_pose_right_arm = p.calculateInverseKinematics(digit_id, digit_eff_idxs[0],
                                                        marker_positions[0],
                                                        marker_orientations[0],
                                                        restPoses=initPose)
    
    left_leg_joints = ['hip_abduction_left', 'hip_rotation_left', 'hip_flexion_left', 'knee_joint_left', 'shin_to_tarsus_left', 'toe_pitch_joint_left', 'toe_roll_joint_left']
    right_leg_joints = ['hip_abduction_right', 'hip_rotation_right', 'hip_flexion_right', 'knee_joint_right', 'shin_to_tarsus_right', 'toe_pitch_joint_right', 'toe_roll_joint_right']
    left_arm_joints = ['shoulder_roll_joint_left',  'shoulder_pitch_joint_left', 'shoulder_yaw_joint_left', 'elbow_joint_left']
    right_arm_joints = ['shoulder_roll_joint_right', 'shoulder_pitch_joint_right', 'shoulder_yaw_joint_right', 'elbow_joint_right']
    # right_arm_joints = ['hip_abduction_right', 'hip_rotation_right', 'hip_flexion_right', 'knee_joint_right', 'knee_to_shin_right', 'shin_to_tarsus_right', 'toe_pitch_joint_right', 'toe_roll_joint_right']
    joints = [p.getJointInfo(digit_id, i)[1].decode() for i in range(p.getNumJoints(digit_id)) if p.getJointInfo(digit_id, i)[3] > -1]
    # print(joints)
    left_leg_idxs = [joints.index(name) for name in left_leg_joints]
    right_leg_idxs = [joints.index(name) for name in right_leg_joints]
    left_arm_idxs = [joints.index(name) for name in left_arm_joints]
    right_arm_idxs = [joints.index(name) for name in right_arm_joints]
    # left_leg_idxs = [i for i in range(p.getNumJoints(digit_id)) if p.getJointInfo(digit_id, i)[1].decode() in left_leg_joints]
    # right_leg_idxs = [i for i in range(p.getNumJoints(digit_id)) if p.getJointInfo(digit_id, i)[1].decode() in right_leg_joints]
    # left_arm_idxs = [i for i in range(p.getNumJoints(digit_id)) if p.getJointInfo(digit_id, i)[1].decode() in left_arm_joints]
    # right_arm_idxs = [i for i in range(p.getNumJoints(digit_id)) if p.getJointInfo(digit_id, i)[1].decode() in right_arm_joints]
    # print(left_leg_idxs, right_leg_idxs,left_arm_idxs, right_arm_idxs, len(joint_pose))
    # print(len(joint_pose_left_leg), len(joint_pose_right_leg))
    
    # print(np.array(joint_pose_leg)-np.array(joint_pose))
    joint_pose_left_leg = np.array(joint_pose_left_leg)
    joint_pose_right_leg = np.array(joint_pose_right_leg)
    joint_pose_left_arm = np.array(joint_pose_left_arm)
    joint_pose_right_arm = np.array(joint_pose_right_arm)
    
    joint_pose[right_leg_idxs] = joint_pose_right_leg[right_leg_idxs]
    joint_pose[left_leg_idxs] = joint_pose_left_leg[left_leg_idxs]
    # joint_pose[right_arm_idxs] = joint_pose_right_arm[right_arm_idxs]
    # joint_pose[left_arm_idxs] = joint_pose_left_arm[left_arm_idxs]
    # print(right_arm_idxs, left_arm_idxs, right_leg_idxs, left_leg_idxs)
    
    return  np.array(joint_pose)

def get_error(marker_id, digit_eff_idxs, digit_id):
    eff_pos = [p.getLinkState(digit_id, idx)[4] for idx in digit_eff_idxs]
    eff_pos = np.array(eff_pos)
    marker_positions = np.array([p.getBasePositionAndOrientation(marker)[0] for marker in marker_id])
    return np.linalg.norm(eff_pos-marker_positions, axis=1)

def get_quat_error(quat1,quat2):
    error_q = p.multiplyTransforms([0,0,0], quat1, [0,0,0], p.invertTransform([0,0,0], quat2)[1])[1]
    angles = p.getEulerFromQuaternion(error_q)
    print(angles)
    return angles

def create_markers(agent_id, body_idxs, radius=0.05,colors=[[1,0,0,1], [0,1,0,1], [0,0,1,1], [1,1,0,1]]):
    markers = []
    for color, body_idx in zip(colors,body_idxs):
        pos, ori = p.getLinkState(agent_id, body_idx)[4:6]
        offset = np.array([0,-1,0.2])
        marker = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        marker_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=marker, basePosition=np.array(pos)+offset, baseOrientation=ori)
        markers.append(marker_id)
    return markers   

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setRealTimeSimulation(0)

################################################################################
########################### Load digit and humanoid ############################
################################################################################


plane_id = p.loadURDF("plane.urdf",[0,0,0],[0,0,0,1])
digit_id = p.loadURDF("/home/dan/Projects/dynamic_motion_imitation/IsaacGymEnvs/assets/urdf/DigitRobot/DigitRobot/urdf/digit_model.urdf",[0,-1,1.1],[0,0,0,1])
# digit_id = p.loadURDF("../assets/urdf/digit_description-main/urdf/digit_float.urdf",[0,-1,1.1],[0,0,0,1])
humanoid_id = p.loadMJCF("/home/dan/Projects/dynamic_motion_imitation/IsaacGymEnvs/assets/mjcf/amp_humanoid_pybullet.xml")[0]

humanoid_eff_names = ["right_hand", "left_hand", "right_foot", "left_foot"]
humanoid_bodies = [p.getJointInfo(humanoid_id, i)[12].decode() for i in range(p.getNumJoints(humanoid_id))]
humanoid_eff = [humanoid_bodies.index(name) for name in humanoid_eff_names]
humanoid_joints = [i for i in range(p.getNumJoints(humanoid_id)) if p.getJointInfo(humanoid_id, i)[2] != p.JOINT_FIXED]

digit_eff_names = ["right_hand", "left_hand", "right_toe_roll", "left_toe_roll"]
digit_bodies = [p.getJointInfo(digit_id, i)[12].decode() for i in range(p.getNumJoints(digit_id))]
digit_eff = [digit_bodies.index(name) for name in digit_eff_names]

for joint in range(p.getNumJoints(digit_id)):
    p.setJointMotorControl2(digit_id, joint, p.POSITION_CONTROL, targetVelocity=1, force=10)
for joint in range(p.getNumJoints(humanoid_id)):
        p.setJointMotorControl2(humanoid_id, joint, p.POSITION_CONTROL, targetVelocity=1, force=10)
p.setJointMotorControlArray(digit_id, [i for i in range(p.getNumJoints(digit_id))], p.POSITION_CONTROL, targetPositions=[0]*p.getNumJoints(digit_id))


################################################################################
#################################### Load mocap ################################
################################################################################
# motion_file = "amp_humanoid_run.npy"
motion_file_path = "/home/dan/Projects/dynamic_motion_imitation/IsaacGymEnvs/assets/amp/motions/"+motion_file
mocap = MotionLib(motion_file=motion_file_path, 
                                     num_dofs=28,
                                     key_body_ids=torch.tensor([5,8,11,14],device=torch.device("cuda:0")), 
                                     device=torch.device("cuda:0"))
length = 2700
dt = 0.005
print(mocap, mocap.num_motions(),mocap._motion_lengths[0])
motion_times0 = mocap.sample_time(np.array([0]))
motion_times0 = [0.0]
print("motion_time 0:",motion_times0)
motion_times = np.expand_dims(motion_times0, axis=-1)
length = int(mocap._motion_lengths[0]/dt)
time_steps = dt * np.arange(0, length)
motion_times = motion_times + time_steps
motion_ids = np.array([0]*length).flatten()
motion_times = motion_times.flatten()

frame = mocap.get_motion_state(np.array([0]*length), motion_times)
root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = frame
print("num frames", len(root_pos))


############################################################################################################
###################################### Mount humanoid and digit to rack ####################################
############################################################################################################

p.createConstraint(digit_id, -1, plane_id, -1, p.JOINT_FIXED, [0,0,0], [0,0,0], root_pos[0].cpu().numpy()+np.array([0,-1,0.4]),[0,0,0,1],root_rot[0].cpu().numpy())
p.createConstraint(humanoid_id, -1, plane_id, -1, p.JOINT_FIXED, [0,0,0], [0,0,0], root_pos[0].cpu().numpy()+np.array([0,0,0.15]),[0,0,0,1],root_rot[0].cpu().numpy())

############################################################################################################
###################################### Create Rod constraints for digit ####################################
############################################################################################################

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

retargetted_poses = []
body_state = []
error = []
humanoid_dduckie = DebugDuckie([1,0,1], humanoid_id, humanoid_eff[2])
digit_dduckie = DebugDuckie([1,-1,1], digit_id, digit_eff[3])
digit_left_foot_base_rot = [-0.331,-0.367,-0.654, -0.573]
digit_right_foot_base_rot = [0.331,-0.367,0.654, -0.573]
digit_left_foot_base_rot_inv = p.invertTransform([0,0,0],digit_left_foot_base_rot)[1]
digit_right_foot_base_rot_inv = p.invertTransform([0,0,0],digit_right_foot_base_rot)[1]

digit_dduckie.update(p.multiplyTransforms([0,0,0], p.getLinkState(digit_id, digit_eff[2])[5],[0,0,0], digit_right_foot_base_rot_inv)[1])
for idx in range(len(root_pos)):

    p.setJointMotorControlArray(humanoid_id, humanoid_joints, p.POSITION_CONTROL, targetPositions=dof_pos[idx].cpu().numpy())    
    p.stepSimulation()
    humanoid_dduckie.update()
    # digit_dduckie.update(p.multiplyTransforms([0,0,0], p.getLinkState(digit_id, digit_eff[3])[5],*p.invertTransform([0,0,0],digit_left_foot_base_rot))[1])
    digit_dduckie.update(p.multiplyTransforms([0,0,0], p.getLinkState(digit_id, digit_eff[2])[5],[0,0,0], digit_right_foot_base_rot_inv)[1])
    
    # digit_dduckie.update(p.multiplyTransforms(*p.invertTransform([0,0,0],digit_left_foot_base_rot),[0,0,0], p.getLinkState(digit_id, digit_eff[3])[5])[1])
    
    # digit_dduckie.update(p.multiplyTransforms([0,0,0],p.invertTransform([0,0,0],[0.183,-0.613,-0.206,-0.740])[1],[0,0,0], p.getLinkState(digit_id, digit_eff[2])[5])[1])

    if idx == 0:
        for i in range(100):
            p.stepSimulation()
            time.sleep(1./240.)
        markers_humanoid = create_markers(humanoid_id, body_idxs=humanoid_eff, radius=0.05, colors=[[1,0,0,1], [0,1,0,1], [0,0,1,1], [1,1,0,1]])
        markers_digit = create_markers(digit_id, body_idxs=digit_eff, radius=0.05, colors=[[1,0,0,1], [0,1,0,1], [0,0,1,1], [1,1,0,1]])

    ################### update marker position from end-effector position and orientation ###################
    for i in range(len(markers_humanoid)):
        offset = np.array([0,-1,0.0])
        pos_eff_humanoid,rot_eff_humanoid = p.getLinkState(humanoid_id, humanoid_eff[i])[4:6]
        pos_eff_digit,rot_eff_digit = p.getLinkState(digit_id, digit_eff[i])[4:6]
        
        foot_factor = np.array([1,1,1])
        if i >1:
            foot_factor[-1]=1
        if i == 3:
           rot_eff_humanoid = p.multiplyTransforms([0,0,0], rot_eff_humanoid, [0,0,0], digit_left_foot_base_rot)[1]
        elif i == 2:
            rot_eff_humanoid = p.multiplyTransforms([0,0,0], rot_eff_humanoid, [0,0,0], digit_right_foot_base_rot)[1]
        else:
            rot_eff_humanoid = rot_eff_digit
        p.resetBasePositionAndOrientation(markers_humanoid[i], np.array(pos_eff_humanoid)*foot_factor+offset, rot_eff_humanoid)
        # p.resetBasePositionAndOrientation(markers_digit[i], pos_eff_digit, rot_eff_digit)
        left_foot_rot_error = np.linalg.norm(get_quat_error(p.getBasePositionAndOrientation(markers_humanoid[3])[1], p.getLinkState(digit_id, digit_eff[3])[5]))
        right_foot_rot_error = np.linalg.norm(get_quat_error(p.getBasePositionAndOrientation(markers_humanoid[2])[1], p.getLinkState(digit_id, digit_eff[2])[5]))
    # while get_error(markers_humanoid, digit_eff, digit_id).sum()>0.0001 or left_foot_rot_error>0.1:# or right_foot_rot_error>0.1:
    for k in range(1):
        if idx == 0:
            initPose = None
        else:
            initPose = [p.getJointState(digit_id, digit_body_idx)[0] for digit_body_idx in range(30) if p.getJointInfo(digit_id, digit_body_idx)[2]==0]
        digit_dofs = retarget_humanoid_to_digit_ik_constrained([p.getBasePositionAndOrientation(marker)[0] for marker in markers_humanoid], 
                                                    [p.getBasePositionAndOrientation(marker)[1] for marker in markers_humanoid],
                                                digit_eff,
                                                digit_id,
                                                initPose)
                                                # [p.getJointState(digit_id, digit_body_idx)[0] for digit_body_idx in range(30) if p.getJointInfo(digit_id, digit_body_idx)[2]==0])
                                                # p.getJointStates(digit_id,[i for i in range(p.getNumJoints(digit_id)) if p.getJointInfo(digit_id, i)[3]>-1])[0])

        # for i in range(p.getNumJoints(digit_id)):
        #     jointInfo = p.getJointInfo(digit_id, i)
        #     qIndex = jointInfo[3]
        #     if qIndex > -1:
        #             p.resetJointState(digit_id,i,digit_dofs[qIndex-7])
        left_foot_rot_error = np.linalg.norm(get_quat_error(p.getBasePositionAndOrientation(markers_humanoid[3])[1], p.getLinkState(digit_id, digit_eff[3])[5]))
        right_foot_rot_error = np.linalg.norm(get_quat_error(p.getBasePositionAndOrientation(markers_humanoid[2])[1], p.getLinkState(digit_id, digit_eff[2])[5]))
    retargetted_poses.append(np.concatenate((root_pos[idx].cpu().numpy(),root_rot[idx].cpu().numpy(), digit_dofs)))
    joint_info = {p.getJointInfo(digit_id, digit_body_idx)[12].decode():p.getLinkState(digit_id, digit_body_idx)[5] for digit_body_idx in range(30)}
    body_state.append((joint_info,root_pos[idx].cpu().numpy(),root_rot[idx].cpu().numpy(),digit_dofs))
    error.append(get_error(markers_digit, digit_eff, digit_id))
    quaternion_error = get_quat_error(p.getBasePositionAndOrientation(markers_humanoid[2])[1], p.getLinkState(digit_id, digit_eff[2])[5])
    print("q error:", quaternion_error)
    # time.sleep(1)

np.save("digit_state_"+motion_file.split("_")[-1], body_state)
print("saved digit state at", "digit_state_"+motion_file.split("_")[-1])
# error = np.array(error)**2
# # np.save("error_digit_v2.npy", error)
# fig, axes = plt.subplots(nrows=2, ncols=2)
# axes[0,0].plot(error[:,0])
# axes[0,1].plot(error[:,1])
# axes[1,0].plot(error[:,2])
# axes[1,1].plot(error[:,3])
# axes[0,0].set_title("squared error right hand")
# axes[0,1].set_title("squared error left hand")
# axes[1,0].set_title("squared error right foot")
# axes[1,1].set_title("squared error left foot")
# plt.show()
