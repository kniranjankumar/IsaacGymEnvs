import math
import numpy as np
from isaacgym import gymapi, gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from isaacgymenvs.utils.torch_jit_utils import *
from torch import nn
import torch.nn as nn
from isaacgymenvs.tasks.amp.utils_amp.motion_lib import MotionLib
import time
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from collections import OrderedDict
KEY_BODY_NAMES = ["right_hand", "left_hand", "left_toe_roll", "right_toe_roll"]

class PolicyNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], output_size)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

@torch.jit.script
def dof_to_obs(pose):
    return pose
    # type: (Tensor) -> Tensor
    #dof_obs_size = 64
    #dof_offsets = [0, 3, 6, 9, 12, 13, 16, 19, 20, 23, 24, 27, 30, 31, 34]
    dof_obs_size = 52
    dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

        # assume this is a spherical joint
        if (dof_size == 3):
            joint_pose_q = exp_map_to_quat(joint_pose)
            joint_dof_obs = quat_to_tan_norm(joint_pose_q)
            dof_obs_size = 6
        else:
            joint_dof_obs = joint_pose
            dof_obs_size = 1

        dof_obs[:, dof_obs_offset:(dof_obs_offset + dof_obs_size)] = joint_dof_obs
        dof_obs_offset += dof_obs_size

    return dof_obs

@torch.jit.script
def compute_digit_observations(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    root_vel = root_states[:, 7:10]
    root_ang_vel = root_states[:, 10:13]

    root_h = root_pos[:, 2:3]
    heading_rot = calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = quat_to_tan_norm(root_rot_obs)

    local_root_vel = my_quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = my_quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    dof_obs = dof_to_obs(dof_pos)

    obs = torch.cat((root_h, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    return obs

def apply_forces_to_rigid_bodies(_rigid_body_pos,_rigid_body_rot, _rigid_body_vel,_rigid_body_ang_vel):
        rtoe_offset = [[-0.049,0.01,0.0]]
        rtarsus_offset = [[0.11,0.085,0]]
        ltoe_offset = [[-0.049,-0.01,0.0]]
        ltarsus_offset = [[0.11,-0.085,0]]
        _rod_kp = 1000
        _rod_kd = 100
        knee_tarsus_rod_length = 0.5518
        toe_tarsus_rod_length = 0.29
        num_envs = 1
        rtoe_point = tf_apply(_rigid_body_rot[:,15],_rigid_body_pos[:,15],to_torch(rtoe_offset).repeat(num_envs,1))
        rtarsus_point = tf_apply(_rigid_body_rot[:,14],_rigid_body_pos[:,14],to_torch(rtarsus_offset).repeat(num_envs,1))
        ltoe_point = tf_apply(_rigid_body_rot[:,7],_rigid_body_pos[:,7],to_torch(ltoe_offset).repeat(num_envs,1))
        ltarsus_point = tf_apply(_rigid_body_rot[:,6],_rigid_body_pos[:,6],to_torch(ltarsus_offset).repeat(num_envs,1))
        
        rknee_offset = [[-0.02, 0.1, 0.0]]
        lknee_offset = [[-0.02, -0.1, 0.0]]
        rtarsus_offset2 = [[-0.1,0.01,0]]
        ltarsus_offset2 = [[-0.1,-0.01,0]]
        rknee_point = tf_apply(_rigid_body_rot[:,12],_rigid_body_pos[:,12],to_torch(rknee_offset).repeat(num_envs,1))
        lknee_point = tf_apply(_rigid_body_rot[:,4],_rigid_body_pos[:,4],to_torch(lknee_offset).repeat(num_envs,1))
        rtarsus_point2 = tf_apply(_rigid_body_rot[:,14],_rigid_body_pos[:,14],to_torch(rtarsus_offset2).repeat(num_envs,1))
        ltarsus_point2 = tf_apply(_rigid_body_rot[:,6],_rigid_body_pos[:,6],to_torch(ltarsus_offset2).repeat(num_envs,1))
        rknee_point_vel = _rigid_body_vel[:,12]+torch.cross(_rigid_body_ang_vel[:,12],to_torch(rknee_offset).repeat(num_envs,1))
        lknee_point_vel = _rigid_body_vel[:,4]+torch.cross(_rigid_body_ang_vel[:,4],to_torch(lknee_offset).repeat(num_envs,1))
        rtarsus_point2_vel = _rigid_body_vel[:,14]+torch.cross(_rigid_body_ang_vel[:,14],to_torch(rtarsus_offset2).repeat(num_envs,1))
        ltarsus_point2_vel = _rigid_body_vel[:,6]+torch.cross(_rigid_body_ang_vel[:,6],to_torch(ltarsus_offset2).repeat(num_envs,1))
        
        rknee_tarsus_rod = rknee_point - rtarsus_point2
        lknee_tarsus_rod = lknee_point - ltarsus_point2
        rtoe_tarsus_rod = rtoe_point - rtarsus_point
        ltoe_tarsus_rod = ltoe_point - ltarsus_point
        
        rknee_tarsus_rod_direction = rknee_tarsus_rod/torch.norm(rknee_tarsus_rod, dim=1).unsqueeze(1)
        lknee_tarsus_rod_direction = lknee_tarsus_rod/torch.norm(lknee_tarsus_rod, dim=1).unsqueeze(1)
        rknee_tarsus_projected_vel_error = torch.sum(rknee_tarsus_rod_direction*rknee_point_vel, dim=1)-torch.sum(rknee_tarsus_rod_direction*rtarsus_point2_vel, dim=1)
        lknee_tarsus_projected_vel_error = torch.sum(lknee_tarsus_rod_direction*lknee_point_vel, dim=1)-torch.sum(lknee_tarsus_rod_direction*ltarsus_point2_vel, dim=1)
        rknee_tarsus_rod_force = _rod_kp * (torch.norm(rknee_tarsus_rod, dim=1) - knee_tarsus_rod_length) + _rod_kd*(rknee_tarsus_projected_vel_error)
        lknee_tarsus_rod_force = _rod_kp * (torch.norm(lknee_tarsus_rod, dim=1) - knee_tarsus_rod_length) + _rod_kd*(lknee_tarsus_projected_vel_error)
        rtoe_tarsus_rod_force = _rod_kp * (torch.norm(rtoe_tarsus_rod, dim=1) - toe_tarsus_rod_length)
        ltoe_tarsus_rod_force = _rod_kp * (torch.norm(ltoe_tarsus_rod, dim=1) - toe_tarsus_rod_length)
        
        forces = torch.zeros((num_envs, digit_num_bodies, 3), device=device, dtype=torch.float)
        force_positions = torch.zeros((num_envs, digit_num_bodies, 3), device=device, dtype=torch.float)
        force_positions[:, 15, :] = to_torch(rtoe_offset)
        force_positions[:, 7, :] = to_torch(ltoe_offset)
        # force_positions[:, 14, :] = rtarsus_offset
        # force_positions[:, 6, :] = ltarsus_offset
        
        force_positions[:, 12, :] = to_torch(rknee_offset)
        force_positions[:, 4, :]  = to_torch(lknee_offset)
        force_positions[:, 14, :] = to_torch(rtarsus_offset2)
        force_positions[:, 6, :]  = to_torch(ltarsus_offset2)
        print("rod length error:", torch.norm(rknee_tarsus_rod, dim=1), (torch.norm(rknee_tarsus_rod, dim=1) - knee_tarsus_rod_length))
        # print("shapes", rknee_tarsus_rod_force.shape, rknee_tarsus_rod.shape)
        forces[:, 12, :] = rknee_tarsus_rod_force.unsqueeze(1)*rknee_tarsus_rod_direction
        forces[:, 4, :] = lknee_tarsus_rod_force.unsqueeze(1)*lknee_tarsus_rod_direction
        forces[:, 14, :] = -forces[:, 12, :]
        forces[:, 6, :] = -forces[:, 4, :]
        humanoid_forces = torch.zeros((num_envs, humanoid_num_bodies, 3), device=device, dtype=torch.float)
        humanoid_force_positions = torch.zeros((num_envs, humanoid_num_bodies, 3), device=device, dtype=torch.float)
        all_forces = torch.cat((humanoid_forces, forces), dim=1)
        all_force_positions = torch.cat((humanoid_force_positions, force_positions), dim=1)
        gym.apply_rigid_body_force_at_pos_tensors(sim, gymtorch.unwrap_tensor(all_forces), 
                                                  gymtorch.unwrap_tensor(all_force_positions), 
                                                  gymapi.LOCAL_SPACE)


####################  Load policy  ####################
policy = PolicyNet(69, [1024, 512], 22)
trained_weights = torch.load("/home/dan/Projects/dynamic_motion_imitation/IsaacGymEnvs/runs/Digit_hop_13-15-37-40/nn/final.pth")
policy.fc1.weight.data = trained_weights["model"]["a2c_network.actor_mlp.0.weight"]
policy.fc1.bias.data = trained_weights["model"]["a2c_network.actor_mlp.0.bias"]
policy.fc2.weight.data = trained_weights["model"]["a2c_network.actor_mlp.2.weight"]
policy.fc2.bias.data = trained_weights["model"]["a2c_network.actor_mlp.2.bias"]
policy.fc3.weight.data = trained_weights["model"]["a2c_network.mu.weight"]
policy.fc3.bias.data = trained_weights["model"]["a2c_network.mu.bias"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#################### Configure motion library  ####################
motion_file = "amp_humanoid_hop.npy"
motion_file_path = "/home/dan/Projects/dynamic_motion_imitation/IsaacGymEnvs/assets/amp/motions/"+motion_file
mocap = MotionLib(motion_file=motion_file_path, 
                                     num_dofs=28,
                                     key_body_ids=torch.tensor([5,8,11,14],device=torch.device("cuda:0")), 
                                     device=torch.device("cuda:0"))
length = 2700
dt = 1/60. #0.005
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
mc_root_pos, mc_root_rot, mc_dof_pos, mc_root_vel, mc_root_ang_vel, mc_dof_vel, mc_key_pos = frame
print([item.size() for item in frame])

####################  Configure sim  ####################
args = gymutil.parse_arguments(
    description="Run policy on digit")
gym = gymapi.acquire_gym()
device = args.sim_device if args.use_gpu_pipeline else 'cpu'

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.use_gpu_pipeline = args.use_gpu_pipeline
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.02
    sim_params.physx.bounce_threshold_velocity = 1.0
    sim_params.physx.max_depenetration_velocity = 100.0
    sim_params.physx.default_buffer_size_multiplier = 5.0
    sim_params.physx.max_gpu_contact_pairs= 8388608
    sim_params.physx.contact_collection = gymapi.ContactCollection(2)
    sim_params.physx.num_threads = 4
    sim_params.physx.num_subscenes = 4
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()
    
####################  Configure env  ####################    
# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
plane_params.static_friction = 1.0
plane_params.dynamic_friction = 1.0
plane_params.restitution = 0.0

gym.add_ground(sim, plane_params)
asset_root = "../assets"

asset_options = gymapi.AssetOptions()
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
asset_options.angular_damping = 0.0
asset_options.linear_damping = 0.0
asset_options.armature = 0.0001
asset_options.thickness = 0.01
asset_options.fix_base_link = True
# asset_options.self_collisions = False

print("Loading assets")
humanoid_asset = gym.load_asset(sim, "../../digit_hardware/digit_description/digit-v3","digit-v3.xml", asset_options)
# asset_options.fix_base_link = False
digit_asset = gym.load_asset(sim, asset_root, "urdf/DigitRobot/DigitRobot/urdf/digit_model.urdf", asset_options)
humanoid_num_dofs = gym.get_asset_dof_count(humanoid_asset)
digit_num_dofs = gym.get_asset_dof_count(digit_asset)
digit_num_bodies = gym.get_asset_rigid_body_count(digit_asset)
humanoid_num_bodies = gym.get_asset_rigid_body_count(humanoid_asset)

# digit_dof_props = gym.get_asset_dof_properties(digit_asset)
# digit_dof_props["driveMode"][:].fill(gymapi.DOF_MODE_POS)
# digit_dof_props["stiffness"][:].fill(400.0)
# digit_dof_props["damping"][:].fill(0.1)
# digit_dof_props["armature"][:] = 0.0001


# gym.set_asset_dof_properties(digit_asset, digit_dof_props)


num_envs = 1
num_per_row = 1
spacing = 2.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# position the camera
cam_pos = gymapi.Vec3(0, -3.0, 1)
cam_target = gymapi.Vec3(0, 3.5, 1)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

env = gym.create_env(sim, env_lower, env_upper, num_per_row)
humanoid_pose = gymapi.Transform()
humanoid_pose.p = gymapi.Vec3(0.0, 0, 1.0)
humanoid_pose.r = gymapi.Quat(0, 0.0, 0.0, 1)
digit_pose = gymapi.Transform()
digit_pose.p = gymapi.Vec3(0, 0, 1.00)
digit_pose.r = gymapi.Quat(0, 0.0, 0.0,1)

humanoid_actor_handle = gym.create_actor(env, humanoid_asset, humanoid_pose, "humanoid_actor", 0, 1,1)
digit_actor_handle = gym.create_actor(env, digit_asset, digit_pose, "digit_actor", 0, 1,1)
actuator_props = gym.get_actor_dof_properties(env, digit_actor_handle)
joint_order = ['hip_abduction_left', 
               'hip_rotation_left', 
               'hip_flexion_left', 
               'knee_joint_left', 
               'shin_to_tarsus_left', 
               'toe_pitch_joint_left', 
               'toe_roll_joint_left', 
               'hip_abduction_right', 
               'hip_rotation_right', 
               'hip_flexion_right', 
               'knee_joint_right', 
               'shin_to_tarsus_right', 
               'toe_pitch_joint_right', 
               'toe_roll_joint_right', 
               'shoulder_roll_joint_left', 
               'shoulder_pitch_joint_left', 
               'shoulder_yaw_joint_left', 
               'elbow_joint_left', 
               'shoulder_roll_joint_right', 
               'shoulder_pitch_joint_right', 
               'shoulder_yaw_joint_right', 
               'elbow_joint_right']
toe_joints = [i for i,name in enumerate(joint_order) if "toe" in name]
for i in range(digit_num_dofs):
    actuator_props['driveMode'][i] = gymapi.DOF_MODE_POS
    actuator_props['stiffness'][i] = 200
    actuator_props['damping'][i] = 10.0
    actuator_props["armature"][i] = 0.0001
    actuator_props["effort"][i] = 500
    if i in toe_joints:
        actuator_props["effort"][i] = 5000
        actuator_props['stiffness'][i] = 5000 #Kp
        actuator_props['damping'][i] = 10.
gym.set_actor_dof_properties(env, digit_actor_handle, actuator_props)
gym.prepare_sim(sim)

_key_body_ids = []
dof_limits_lower = []
dof_limits_upper = []
for body_name in KEY_BODY_NAMES:
    body_id = gym.find_actor_rigid_body_handle(env, digit_actor_handle, body_name)
    assert(body_id != -1)
    _key_body_ids.append(body_id)

for j in range(digit_num_dofs):
    # print(j,self.gym.get_actor_dof_names(env_ptr, handle))

    if actuator_props['lower'][j] > actuator_props['upper'][j]:
        dof_limits_lower.append(actuator_props['upper'][j])
        dof_limits_upper.append(actuator_props['lower'][j])
    else:
        dof_limits_lower.append(actuator_props['lower'][j])
        dof_limits_upper.append(actuator_props['upper'][j])
lim_low = to_torch(dof_limits_lower, device=device)
lim_high = to_torch(dof_limits_upper, device=device)
pd_action_offset = 0. * (lim_high + lim_low)
pd_action_scale = 0.5 * (lim_high - lim_low)
pd_action_offset[0] = 0.33
pd_action_offset[7] = -0.33
# hip flexion
# pd_action_offset[2] = -0.0
# pd_action_offset[9] = 0.0
# shoulder pitch
pd_action_offset[15] = 0.9
pd_action_offset[19] = -0.9
# toe pitch
# pd_action_offset[5] = 0.2*0
# pd_action_offset[12] = -0.2*0
# elbow
pd_action_offset[17] = 0.5
pd_action_offset[21] = -0.5
# shin to tarsus
# pd_action_offset[4] = -0.3*0
# pd_action_offset[11] = 0.3*0
#knee
# pd_action_offset[3] = 0.0
# pd_action_offset[10] = -0.0
# gym.set_actor_dof_states(env, humanoid_actor_handle, humanoid_dof_states, gymapi.STATE_ALL)
# gym.set_actor_dof_states(env, digit_actor_handle, digit_dof_states, gymapi.STATE_ALL)
dof_state_tensor_ = gym.acquire_dof_state_tensor(sim)
rigid_body_pos_ = gym.acquire_rigid_body_state_tensor(sim)
root_states_ = gym.acquire_actor_root_state_tensor(sim)

root_states = gymtorch.wrap_tensor(root_states_)
dof_states = gymtorch.wrap_tensor(dof_state_tensor_)
rigid_body_pos = gymtorch.wrap_tensor(rigid_body_pos_).view(humanoid_num_bodies+digit_num_bodies,13)

humanoid_dof_states = dof_states.view(humanoid_num_dofs+digit_num_dofs,2)[0:humanoid_num_dofs,:]
humanoid_dof_pos = dof_states.view(humanoid_num_dofs+digit_num_dofs,2)[0:humanoid_num_dofs,0]
humanoid_dof_vel = dof_states.view(humanoid_num_dofs+digit_num_dofs,2)[0:humanoid_num_dofs,1]
humanoid_rigid_body_states = rigid_body_pos.view(humanoid_num_bodies+digit_num_bodies,13)[0:humanoid_num_bodies,:]
humanoid_root_states = root_states.view(2,13)[0,:]


digit_dof_states = dof_states.view(humanoid_num_dofs+digit_num_dofs,2)[humanoid_num_dofs:humanoid_num_dofs+digit_num_dofs,:]
digit_dof_pos = dof_states.view(humanoid_num_dofs+digit_num_dofs,2)[humanoid_num_dofs:humanoid_num_dofs+digit_num_dofs,0]
digit_dof_vel = dof_states.view(humanoid_num_dofs+digit_num_dofs,2)[humanoid_num_dofs:humanoid_num_dofs+digit_num_dofs,1]
digit_rigid_body_states = rigid_body_pos.view(humanoid_num_bodies+digit_num_bodies,13)[humanoid_num_bodies:humanoid_num_bodies+digit_num_bodies,:]
digit_rigid_body_pos = digit_rigid_body_states[:,0:3]
digit_rigid_body_quat = digit_rigid_body_states[:,3:7]
digit_rigid_body_vel = digit_rigid_body_states[:,7:10]
digit_rigid_body_ang_vel = digit_rigid_body_states[:,10:13]
digit_root_states = root_states.view(2,13)[1,:]



_key_body_ids = to_torch(_key_body_ids, device=device, dtype=torch.long)
print(_key_body_ids, digit_rigid_body_pos)
key_body_pos = rigid_body_pos[_key_body_ids, :3]

dof_pos = torch.zeros((humanoid_num_dofs+digit_num_dofs,1),device=device)
# pd_action_scale = to_torch([1.0472, 0.6981, 1.3090, 1.0559, 1.0638, 0.7854, 0.6109, 1.0472, 0.6981,
#         1.3090, 1.0559, 1.0638, 0.7854, 0.6109, 1.3090, 2.5307, 1.7453, 1.3526,
#         1.3090, 2.5307, 1.7453, 1.3526],  device=digit_rigid_body_states.device, dtype=torch.long)
# pd_action_offset = to_torch([ 0.3300,  0.0000, -0.0000,  0.0000, -0.0000,  0.0000,  0.0000, -0.3300,
#          0.0000,  0.0000, -0.0000,  0.0000, -0.0000,  0.0000,  0.0000,  0.9000,
#          0.0000,  0.5000,  0.0000, -0.9000,  0.0000, -0.5000],  device=digit_rigid_body_states.device, dtype=torch.long)

_initial_dof_pos=torch.tensor([ 2.7886e-01, -1.5718e-03,  2.4364e-01, -1.6560e-01,  1.8958e-01,
    1.0037e-01,  8.0998e-06, -2.7884e-01,  1.6481e-03, -2.4295e-01,
    1.6479e-01, -1.8975e-01, -1.0040e-01,  1.0898e-04, -9.6694e-04,
    9.0086e-01,  1.3919e-05,  3.5007e-04,  9.6761e-04, -9.0091e-01,
    2.6128e-06, -3.3921e-04]).to(dof_states.device)
_initial_dof_pos = torch.tensor([ 3.3000e-01, -1.5718e-03, -0.0000e+00,  0.0000e+00, -0.0000e+00,
         0.0000e+00,  8.0998e-06, -3.3000e-01,  1.6481e-03,  0.0000e+00,
        -0.0000e+00,  0.0000e+00, -0.0000e+00,  1.0898e-04, -9.6694e-04,
         9.0000e-01,  1.3919e-05,  5.0000e-01,  9.6761e-04, -9.0000e-01,
         2.6128e-06, -5.0000e-01], device='cuda:0')
digit_root_states[:] = torch.tensor([4.7614e-08, 1.5862e-08, 9.3000e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00], device='cuda:0')
humanoid_root_states[:] = torch.tensor([0, 0.0, 0.93, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00], device='cuda:0')
# digit_root_states = root_pos
# _initial_dof_pos[:,0] = 0.33
# _initial_dof_pos[:,7] = -0.33
# # hip flexion
# _initial_dof_pos[:,2] = -0.
# _initial_dof_pos[:,9] = 0.
# # shoulder pitch
# _initial_dof_pos[:,15] = 0.9
# _initial_dof_pos[:,19] = -0.9
# # toe pitch
# _initial_dof_pos[:,5] = 0.2*0
# _initial_dof_pos[:,12] = -0.2*0
# # elbow
# _initial_dof_pos[:,17] = 0.5
# _initial_dof_pos[:,21] = -0.5
# # shin to tarsus
# _initial_dof_pos[:,4] = -0.3*0
# _initial_dof_pos[:,11] = 0.3*0
# #knee
# _initial_dof_pos[:,3] = 0.0
# _initial_dof_pos[:,10] = -0.0
digit_dof_pos =_initial_dof_pos
digit_dof_vel[:] =0


humanoid_dof_states *=0
gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_states))
print(root_states)
# assert False
gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states))
_amp_input_mean_std = RunningMeanStd(torch.tensor(69)).to(device)
_amp_input_mean_std.eval()

weights = OrderedDict()
weights["running_mean"] = trained_weights["model"]["running_mean_std.running_mean"]
weights["running_var"] = trained_weights["model"]["running_mean_std.running_var"]
weights["count"] = trained_weights["model"]["running_mean_std.count"]
# weights["count"] = torch.tensor(1.9665e+09, device='cuda:0')
_amp_input_mean_std.load_state_dict(weights)
step_count= 0
total_dofs = humanoid_num_dofs+digit_num_dofs
idx = 0
while not gym.query_viewer_has_closed(viewer):
    dof_states[:humanoid_num_dofs,0] = 0
    dof_states[:humanoid_num_dofs,1] = 0
    # root_states[0,0:3] = mc_root_pos[idx]
    # root_states[0,3:7] = mc_root_rot[idx]
    # root_states[0,7:10] = mc_root_vel[idx]
    # root_states[0,10:13] = mc_root_ang_vel[idx]
    gym.set_dof_state_tensor_indexed(sim, gymtorch.unwrap_tensor(dof_states),
                                     gymtorch.unwrap_tensor(torch.tensor([0], device=device, dtype=torch.int32)),
                                     1)
    gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(root_states),
                                     gymtorch.unwrap_tensor(torch.tensor([0], device=device, dtype=torch.int32)),
                                     1)
    idx +=1
    # gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_force_sensor_tensor(sim)
    gym.refresh_dof_force_tensor(sim)
    gym.refresh_net_contact_force_tensor(sim)
    
    key_body_pos = rigid_body_pos[_key_body_ids, :3]
    obs = compute_digit_observations(digit_root_states.unsqueeze(0), 
                                     dof_states.view(total_dofs,2)[humanoid_num_dofs:,0].unsqueeze(0), 
                                     dof_states.view(total_dofs,2)[humanoid_num_dofs:,1].unsqueeze(0),
                                            key_body_pos.unsqueeze(0), False)    
    # print(digit_root_states.size(), digit_dof_pos.size(), digit_dof_vel.size(), key_body_pos.size())
    # obs = compute_digit_observations(digit_root_states.unsqueeze(0), digit_dof_pos.unsqueeze(0), digit_dof_vel.unsqueeze(0),
    #                                         key_body_pos.unsqueeze(0), False)
    # print(obs)
    # print(digit_rigid_body_pos)
    # break

    # print(obs)
    obs = _amp_input_mean_std(obs)
    action = policy(obs).squeeze(0)*0
    # print(action)
    # print("dof", action*pd_action_scale+ pd_action_offset)
    apply_forces_to_rigid_bodies(digit_rigid_body_pos.unsqueeze(0), 
                                 digit_rigid_body_quat.unsqueeze(0), 
                                 digit_rigid_body_vel.unsqueeze(0), 
                                 digit_rigid_body_ang_vel.unsqueeze(0))
    step_count += 1 
    # if step_count == 2:
    #     break
    dof_pos[:,0] = 0
    dof_pos[humanoid_num_dofs+2,0] = 0.5
    # dof_pos[:3,0] = 2
    # dof_pos[:humanoid_num_dofs,0] = 0.5
    # dof_vel[:humanoid_num_dofs,0] = 0
    
    # dof_pos[humanoid_num_dofs:,0] = action*pd_action_scale+ pd_action_offset
    # print(actio)
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(dof_pos))

    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)
    gym.simulate(sim)
    # time.sleep(0.1)
    # gym.simulate(sim)
    # gym.fetch_results(sim, True)
    # gym.refresh_dof_state_tensor(sim)
    # gym.set_dof_state_tensor(sim, dof_state_tensor_)
    # # gym.set_actor_dof_states(env, humanoid_actor_handle, humanoid_dof_state, gymapi.STATE_ALL)    
    # # gym.set_actor_dof_states(env, digit_actor_handle, digit_dof_state, gymapi.STATE_ALL)
    # gym.step_graphics(sim)
    # gym.draw_viewer(viewer, sim, True)
    # gym.sync_frame_time(sim)
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
    