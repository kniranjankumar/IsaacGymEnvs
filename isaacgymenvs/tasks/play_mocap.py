import math
import numpy as np
from isaacgym import gymapi, gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.amp.utils_amp.motion_lib import MotionLib
import time

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

def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)

def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

def compute_IK(j_eef, reference_pos, curr_pos, reference_quat=None, curr_quat=None, damping=0.01):
    dpose = torch.zeros(6,device=j_eef.device)
    pos_err_vec = reference_pos - curr_pos
    pose_error_magnitude = pos_err_vec.length()
    pos_err_dir = pos_err_vec.normalize()
    # pos_err = to_torch(humanoid_rb_pose[7][0]) - to_torch([robot_hand_pose.x, robot_hand_pose.y, robot_hand_pose.z])   
    dpose[0] = pos_err_dir.x
    dpose[1] = pos_err_dir.y
    dpose[2] = pos_err_dir.z
    dpose[3:] = 0
    # print(reference, target,pose_error_magnitude)
    if pose_error_magnitude < 0.01:
        dpose[:3] *= pose_error_magnitude
        # count +=1\
    # else:
    dpose[:3] /=100
    if reference_quat is not None and curr_quat is not None:
        # reference_quat = torch.tensor([0.412,0.376,0.534,-0.636],device=reference_quat.device).view(1,-1)
        dpose[3:] = orientation_error(desired=reference_quat.view(1,-1), current=curr_quat.view(1,-1))/100
    lmbda = torch.eye(6, device=j_eef.device) * (damping ** 2)
    j_eef_T = torch.transpose(j_eef, 1, 2)
    u = (j_eef_T[3] @ torch.inverse(j_eef[3] @ j_eef_T[3] + lmbda) @ dpose)
    return u, pose_error_magnitude+torch.norm(dpose[3:]*10).tolist()

### Trajectories
# trajectories = trajectories[::-1]

### Isaac Gym Initialization
args = gymutil.parse_arguments(
    description="Joint monkey: Animate degree-of-freedom ranges")

# initialize gym
gym = gymapi.acquire_gym()
motion_file = "/home/dan/Projects/dynamic_motion_imitation/IsaacGymEnvs/assets/amp/motions/amp_humanoid_walk.npy"
mocap = MotionLib(motion_file=motion_file, 
                                     num_dofs=28,
                                     key_body_ids=torch.tensor([5,8,11,14],device=torch.device("cuda:0")), 
                                     device=torch.device("cuda:0"))
length = 1000
print(mocap, mocap.num_motions(),mocap._motion_lengths[0])
motion_times0 = mocap.sample_time(np.array([0]))
# motion_times0 = np.array([0])
motion_times = np.expand_dims(motion_times0, axis=-1)
time_steps = -0.0166 * np.arange(0, length)
motion_times = motion_times + time_steps
motion_ids = np.array([0]*length).flatten()
motion_times = motion_times.flatten()

frame = mocap.get_motion_state(np.array([0]*length), motion_times)
root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = frame
print([item.size() for item in frame])


# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 30.0
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 0
sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()
    
# load assets
asset_root = "../assets"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = False
asset_options.use_mesh_materials = True

print("Loading asset '%s' from '%s'" % ("mjcf/amp_humanoid.xml", asset_root))
asset_options.fix_base_link = True
digit_asset = gym.load_asset(sim, asset_root, "urdf/DigitRobot/DigitRobot/urdf/digit_model.urdf", asset_options)
# get array of DOF names
digit_dof_names = gym.get_asset_dof_names(digit_asset)
# get array of DOF properties
digit_dof_props = gym.get_asset_dof_properties(digit_asset)
# create an array of DOF states that will be used to update the actors
digit_num_dofs = gym.get_asset_dof_count(digit_asset)
digit_dof_states = np.zeros(digit_num_dofs, dtype=gymapi.DofState.dtype)

# get list of DOF types

# get the position slice of the DOF state array
digit_dof_positions = digit_dof_states['pos']

# initialize default positions, limits, and speeds (make sure they are in reasonable ranges)

# set up the env grid
num_envs = 1
num_per_row = 1
spacing = 2.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# position the camera
cam_pos = gymapi.Vec3(0, -10.0, 4)
cam_target = gymapi.Vec3(0, 3.5, 4)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# cache useful handles
envs = []
digit_actor_handles = []
trajectories = np.load("retargeted_poses_digit_v2.npy")
print("Creating %d environments" % num_envs)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add actor
    digit_pose = gymapi.Transform()
    digit_pose.p = gymapi.Vec3(0, 0, 2.1)
    digit_pose.r = gymapi.Quat(0, 0.0, 0.0,1)
    
    digit_actor_handle = gym.create_actor(env, digit_asset, digit_pose, "digit_actor", i*2+1, 1)
    digit_actor_handles.append(digit_actor_handle)
    # set default DOF positions
    gym.set_actor_dof_states(env, digit_actor_handle, digit_dof_states, gymapi.STATE_ALL)


count, idx = 0, 0 
_root_states = gym.acquire_actor_root_state_tensor(sim)
root_states = gymtorch.wrap_tensor(_root_states)
digit_pos = root_states.view(num_envs, 1, -1)[:,0,:]
dofs_mapping = [0,1,2,3,4,5,6,
                11,12,13,14,15,16,17,
                7,8,9,10,
                18,19,20,21]

length = len(trajectories)

while not gym.query_viewer_has_closed(viewer):
    time.sleep(0.01)
    # step the physics and refresh tensors
    idx += 1
    count = (idx)%(300) 
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    # gym.refresh_dof_state_tensor(sim)
    # gym.refresh_actor_root_state_tensor(sim)
    alpha = idx%5/5.0
    print(count, alpha)
    # position = (1-alpha)*np.array(trajectories[count][1])+(alpha)*np.array(trajectories[count+1][1])
    # dofs = (1-alpha)*np.array(trajectories[count][2])+(alpha)*np.array(trajectories[count+1][2])
    # position = np.array(trajectories[count][1])
    # dofs = np.array(trajectories[count][2])
    position = trajectories[count][:7]
    position[0] = (count%300)/100.0*0.8
    dofs = trajectories[count][7:]
    digit_pos[:,:7]=torch.tensor(position,device=digit_pos.device)
    digit_pos[:,2]+=0.05
    digit_dof_positions[:] = dofs[dofs_mapping]
    # digit_pos[:,2]-=1.1
    # digit_pos[:,3:7]=root_rot[idx,:]
    gym.set_actor_dof_states(envs[0], digit_actor_handles[0], digit_dof_states, gymapi.STATE_POS)
    gym.set_actor_root_state_tensor(sim, _root_states)
    
    # update viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
    
    
