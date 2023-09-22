import math
import numpy as np
from isaacgym import gymapi, gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.amp.utils_amp.motion_lib import MotionLib
import scipy.optimize


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

def compute_IK(j_eef, reference_pos, curr_pos, reference_quat=None, curr_quat=None, damping=0.01, prev_u=np.zeros(22)):
    dpose = torch.zeros(6,device=j_eef.device)
    pos_err_vec = reference_pos - curr_pos
    pose_error_magnitude = pos_err_vec.length()
    orientation_error_magnitude = 0
    pos_err_dir = pos_err_vec.normalize()
    # pos_err = to_torch(humanoid_rb_pose[7][0]) - to_torch([robot_hand_pose.x, robot_hand_pose.y, robot_hand_pose.z])   
    dpose[0] = pos_err_dir.x
    dpose[1] = pos_err_dir.y
    dpose[2] = pos_err_dir.z
    # dpose[:3] /=100
    dpose[:3] *= np.clip(pose_error_magnitude,0.0,0.01) 
    if pose_error_magnitude < 0.000001:
        return torch.zeros(22,device=j_eef.device), pose_error_magnitude
    # print(reference, target,pose_error_magnitude)
    # if pose_error_magnitude < 0.01:
    #     dpose[:3] *= pose_error_magnitude*100
        # count +=1\
    # else:
    
    if reference_quat is not None and curr_quat is not None:
        # reference_quat = torch.tensor([0.412,0.376,0.534,-0.636],device=reference_quat.device).view(1,-1)
        print(reference_quat, curr_quat)
        dpose[3:] = orientation_error(desired=reference_quat.view(1,-1), current=curr_quat.view(1,-1))
        orientation_error_magnitude = torch.norm(dpose[3:]).tolist()
        # if torch.norm(dpose[3:]) < 0.01:
        # dpose[3:] /= orientation_error_magnitude
        dpose[3:] *= np.clip(orientation_error_magnitude,0,0.01) 
        dpose[3:] *= 0
        # dpose[3] = 0.05
        # dpose[:3] *= 0
        # print("orientation_error_magnitude",orientation_error_magnitude)
    lmbda = torch.eye(6, device=j_eef.device) * (damping ** 2)
    j_eef_T = torch.transpose(j_eef, 1, 2)
    # u = (j_eef_T[3] @ torch.inverse(j_eef[3] @ j_eef_T[3] + lmbda) @ dpose)
    objective = lambda q_dot: torch.norm(j_eef@q_dot-dpose).numpy()
    output = scipy.optimize.minimize(objective, prev_u, method='SLSQP', bounds=[(-0.01,0.01)]*22, jac=False, tol=1e-6)
    u = torch.tensor(output.x,device=j_eef.device)
    return u, pose_error_magnitude#+orientation_error_magnitude

def compute_IK2(reference_pos, curr_pos, sim, dofs):
    objective = get_x_from_dofs(dofs, sim, curr_pos)
    output = scipy.optimize.minimize(objective, dofs[3,:], method='SLSQP', jac=False, tol=1e-6)
    return torch.tensor(output.x,device=j_eef.device)

def get_x_from_dofs(dofs, sim, curr_pos):
    gym.set_actor_dof_states(env, humanoid_actor_handle, dofs, gymapi.STATE_ALL)    
    gym.fetch_results(sim, True)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)
    return curr_pos

### Isaac Gym Initialization
args = gymutil.parse_arguments(
    description="Joint monkey: Animate degree-of-freedom ranges")

# if args.asset_id < 0 or args.asset_id >= len(asset_descriptors):
#     print("*** Invalid asset_id specified.  Valid range is 0 to %d" % (len(asset_descriptors) - 1))
#     quit()
# initialize gym
gym = gymapi.acquire_gym()
motion_file = "/home/dan/Projects/dynamic_motion_imitation/IsaacGymEnvs/assets/amp/motions/amp_humanoid_walk.npy"
mocap = MotionLib(motion_file=motion_file, 
                                     num_dofs=28,
                                     key_body_ids=torch.tensor([5,8,11,14],device=torch.device("cuda:0")), 
                                     device=torch.device("cuda:0"))
length = 10000
# assert False
print(mocap, mocap.num_motions(),mocap._motion_lengths[0])
motion_times0 = mocap.sample_time(np.array([0]))
motion_times0 = [1.0]
print("motion_time 0:",motion_times0)
motion_times = np.expand_dims(motion_times0, axis=-1)
time_steps = 0.001 * np.arange(0, length)
motion_times = motion_times + time_steps
motion_ids = np.array([0]*length).flatten()
motion_times = motion_times.flatten()

frame = mocap.get_motion_state(np.array([0]*length), motion_times)
root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = frame
print([item.size() for item in frame])


# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0
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
# asset_options.self_collisions = False

print("Loading asset '%s' from '%s'" % ("mjcf/amp_humanoid.xml", asset_root))
humanoid_asset = gym.load_asset(sim, asset_root, "mjcf/amp_humanoid.xml", asset_options)
asset_options.fix_base_link = True
digit_asset = gym.load_asset(sim, asset_root, "urdf/DigitRobot/DigitRobot/urdf/digit_model.urdf", asset_options)
# get array of DOF names
humanoid_dof_names = gym.get_asset_dof_names(humanoid_asset)
digit_dof_names = gym.get_asset_dof_names(digit_asset)
# get array of DOF properties
humanoid_dof_props = gym.get_asset_dof_properties(humanoid_asset)
digit_dof_props = gym.get_asset_dof_properties(digit_asset)
digit_lower_limits = digit_dof_props["lower"]
digit_upper_limits = digit_dof_props["upper"]
digit_ranges = digit_upper_limits - digit_lower_limits
digit_mids = 0.3 * (digit_upper_limits + digit_lower_limits)

# use position drive for all dofs
# if controller == "ik":
digit_dof_props["driveMode"][:].fill(gymapi.DOF_MODE_POS)
digit_dof_props["stiffness"][:].fill(400.0)
digit_dof_props["damping"][:].fill(100.0)
# create an array of DOF states that will be used to update the actors
humanoid_num_dofs = gym.get_asset_dof_count(humanoid_asset)
humanoid_dof_states = np.zeros(humanoid_num_dofs, dtype=gymapi.DofState.dtype)
digit_num_dofs = gym.get_asset_dof_count(digit_asset)
digit_dof_states = np.zeros(digit_num_dofs, dtype=gymapi.DofState.dtype)

# get list of DOF types
humanoid_dof_types = [gym.get_asset_dof_type(humanoid_asset, i) for i in range(humanoid_num_dofs)]

# get the position slice of the DOF state array
humanoid_dof_positions = humanoid_dof_states['pos']
digit_dof_positions = digit_dof_states['pos']
# get the limit-related slices of the DOF properties array
stiffnesses = humanoid_dof_props['stiffness']
dampings = humanoid_dof_props['damping']
armatures = humanoid_dof_props['armature']
has_limits = humanoid_dof_props['hasLimits']
humanoid_lower_limits = humanoid_dof_props['lower']
humanoid_upper_limits = humanoid_dof_props['upper']

# initialize default positions, limits, and speeds (make sure they are in reasonable ranges)
defaults_humanoid = np.zeros(humanoid_num_dofs)
speeds_humanoid = np.zeros(humanoid_num_dofs)

# set up the env grid
num_envs = 4
num_per_row = 2
spacing = 2.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# position the camera
cam_pos = gymapi.Vec3(0, -10.0, 4)
cam_target = gymapi.Vec3(0, 3.5, 4)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# cache useful handles
envs = []
humanoid_actor_handles = []
digit_actor_handles = []

print("Creating %d environments" % num_envs)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add actor
    humanoid_pose = gymapi.Transform()
    humanoid_pose.p = gymapi.Vec3(0.0, 0, 2.0)
    humanoid_pose.r = gymapi.Quat(0, 0.0, 0.0, 1)
    digit_pose = gymapi.Transform()
    digit_pose.p = gymapi.Vec3(0, 0, 2.1)
    digit_pose.r = gymapi.Quat(0, 0.0, 0.0,1)
    
    humanoid_actor_handle = gym.create_actor(env, humanoid_asset, humanoid_pose, "humanoid_actor", i*2, 1,0)
    digit_actor_handle = gym.create_actor(env, digit_asset, digit_pose, "digit_actor", i*2+1, 1,1)
    humanoid_actor_handles.append(humanoid_actor_handle)
    digit_actor_handles.append(digit_actor_handle)
    # set default DOF positions
    retargeted_digit_dof_positions = retarget_humanoid_to_digit(humanoid_dof_positions)
    for curr_dof in range(digit_num_dofs):
        digit_dof_positions[curr_dof] = retargeted_digit_dof_positions[curr_dof]
    gym.set_actor_dof_states(env, humanoid_actor_handle, humanoid_dof_states, gymapi.STATE_ALL)
    gym.set_actor_dof_states(env, digit_actor_handle, digit_dof_states, gymapi.STATE_ALL)


# joint animation states
_jacobian = gym.acquire_jacobian_tensor(sim, "digit_actor")
jacobian = gymtorch.wrap_tensor(_jacobian)
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)
_root_states = gym.acquire_actor_root_state_tensor(sim)
root_states = gymtorch.wrap_tensor(_root_states)
humanoid_pos = root_states.view(num_envs, 2, -1)[:,0,:]
digit_pos = root_states.view(num_envs, 2, -1)[:,1,:]
robot_rb_pose = gym.get_actor_rigid_body_states(envs[3], 1, gymapi.STATE_POS)['pose']
humanoid_rb_pose = gym.get_actor_rigid_body_states(envs[3], 0, gymapi.STATE_POS)['pose']
elbow2wrist_offset = gymapi.Vec3(0.35,0,-0.05)
j_LH = jacobian[:, 21,:,:]
j_RH = jacobian[:, 25,:,:]
j_LF = jacobian[:, 6,:,:]
j_RF = jacobian[:, 14,:,:]
count = 0
damping = 0.1

# use joint mapping to initialize digit DOF positions
gym.simulate(sim)
gym.fetch_results(sim, True)
gym.refresh_dof_state_tensor(sim)
gym.refresh_actor_root_state_tensor(sim)
gym.set_actor_dof_states(env, digit_actor_handle, digit_dof_states, gymapi.STATE_ALL)
dpose = torch.zeros(6,device=jacobian.device)
# humanoid_hand_pose_ = gymapi.Vec3(0.1,0.4, 0.5)

# IK to match end effector positions
idx = (count//5)%length
# humanoid_pos[:,:3]=root_pos[idx,:] 
# humanoid_pos[:,3:7]=root_rot[idx,:]
# digit_pos[:,:3]=root_pos[idx,:]
# digit_pos[:,3:7]=root_rot[idx,:]
humanoid_dof_states['pos'] = dof_pos[idx].cpu().numpy()
humanoid_dof_states['vel'] = dof_vel[idx].cpu().numpy()
retargeted_digit_dof_positions = retarget_humanoid_to_digit(humanoid_dof_positions)
for curr_dof in range(digit_num_dofs):
    digit_dof_positions[curr_dof] = retargeted_digit_dof_positions[curr_dof]
gym.set_actor_dof_states(env, digit_actor_handle, digit_dof_states, gymapi.STATE_ALL)
gym.set_actor_dof_states(env, humanoid_actor_handle, humanoid_dof_states, gymapi.STATE_ALL)
gym.set_actor_root_state_tensor(sim, _root_states)
old_u = [np.zeros(22) for i in range(4)]

while not gym.query_viewer_has_closed(viewer):

    # step the physics and refresh tensors
    count += 1
    # idx = (count//5)%length
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)
    
    humanoid_pos[:,:3]=root_pos[idx,:] 
    humanoid_pos[:,2] += 1.0
    humanoid_pos[:,3:7]=root_rot[idx,:]
    humanoid_dof_states['pos'] = dof_pos[idx].cpu().numpy()
    humanoid_dof_states['vel'] = dof_vel[idx].cpu().numpy()
    
    digit_pos[:,:3]=root_pos[idx,:]
    digit_pos[:,2]+=1.1
    digit_pos[:,3:7]=root_rot[idx,:]
    robot_base_pose = gymapi.Transform.from_buffer(robot_rb_pose[0])
    humanoid_base_pose = gymapi.Transform.from_buffer(humanoid_rb_pose[0])
    left_elbow_pose = gymapi.Transform.from_buffer(robot_rb_pose[22])
    right_elbow_pose = gymapi.Transform.from_buffer(robot_rb_pose[26])
    
    # humanoid_hand_pose = robot_base_pose.transform_point(humanoid_hand_pose_)
    # All these are in robot frame of reference
    reference_points = []
    robot_points = []
    
    robot_left_hand_pose = robot_base_pose.inverse().transform_point(left_elbow_pose.transform_point(elbow2wrist_offset))
    humanoid_left_hand_pose = robot_base_pose.inverse().transform_point(gymapi.Vec3(*humanoid_rb_pose[8][0]))
    
    robot_right_hand_pose = robot_base_pose.inverse().transform_point(right_elbow_pose.transform_point(elbow2wrist_offset))
    humanoid_right_hand_pose = robot_base_pose.inverse().transform_point(gymapi.Vec3(*humanoid_rb_pose[5][0]))
    
    robot_left_foot_pose = robot_base_pose.inverse().transform_point(gymapi.Vec3(*robot_rb_pose[7][0]))
    humanoid_left_foot_pose = robot_base_pose.inverse().transform_point(gymapi.Vec3(*humanoid_rb_pose[14][0]))
    
    robot_right_foot_pose = robot_base_pose.inverse().transform_point(gymapi.Vec3(*robot_rb_pose[15][0]))
    humanoid_right_foot_pose = robot_base_pose.inverse().transform_point(gymapi.Vec3(*humanoid_rb_pose[11][0]))
    
    robot_left_elbow_pose = robot_base_pose.inverse().transform_point(gymapi.Vec3(*robot_rb_pose[22][0]))
    humanoid_left_elbow_pose = robot_base_pose.inverse().transform_point(gymapi.Vec3(*humanoid_rb_pose[7][0]))
    
    robot_right_elbow_pose = robot_base_pose.inverse().transform_point(gymapi.Vec3(*robot_rb_pose[26][0]))
    humanoid_right_elbow_pose = robot_base_pose.inverse().transform_point(gymapi.Vec3(*humanoid_rb_pose[4][0]))
    
    left_foot_rot_ = to_torch(robot_rb_pose[8][1]).view(1,-1)
    right_foot_rot_ = to_torch(robot_rb_pose[16][1]).view(1,-1)
    
    left_foot_rot = (robot_base_pose.inverse()*gymapi.Transform.from_buffer(humanoid_rb_pose[8])).r
    # left_foot_rot = quat_mul(torch.tensor([0.412,0.376,0.534,-0.636],device=left_foot_rot_.device).view(1,-1), left_foot_rot_)
    right_foot_rot = quat_mul(torch.tensor([0.412,0.376,0.534,-0.636],device=right_foot_rot_.device).view(1,-1), right_foot_rot_)
    robot_points = [robot_left_hand_pose, robot_right_hand_pose, robot_left_foot_pose, robot_right_foot_pose]#, robot_left_elbow_pose, robot_right_elbow_pose]
    reference_points = [humanoid_left_hand_pose, humanoid_right_hand_pose, humanoid_left_foot_pose, humanoid_right_foot_pose]#, humanoid_left_elbow_pose, humanoid_right_elbow_pose]
    reference_rot = [None]*2+[torch.tensor([-0.354,-0.331,-0.643,-0.592],device=left_foot_rot_.device), torch.tensor([0.363,-0.343,0.596,-0.629],device=left_foot_rot_.device)]
    # robot_rot = [None]*2+[left_foot_rot_, right_foot_rot_]
    robot_rot = [None]*4
    # robot_rot[2] = left_foot_rot
    # robot_rot[2] = torch.tensor([left_foot_rot.x,left_foot_rot.y,left_foot_rot.z,left_foot_rot.w],device=left_foot_rot_.device).view(1,-1)
    jacobians = [j_LH, j_RH, j_LF, j_RF]#, j_LH, j_RH]
    IK_output = [compute_IK(j, ref_pos, robot_pos, ref_quat, robot_quat, damping,u) \
        for j, ref_pos, robot_pos,ref_quat, robot_quat,u in zip(jacobians, reference_points, robot_points, reference_rot, robot_rot,old_u)]
    # IK_output = IK_output[:-2]
    old_u = [u_ for u_, error_ in IK_output]
    u = np.sum([u_ for u_, error_ in IK_output])
    error = [error_ for u_, error_ in IK_output]
    # print(error)
    # print(humanoid_rb_pose)
    # assert False
    # u_LH, error_LH = compute_IK(j_LH, humanoid_left_hand_pose, robot_left_hand_pose, damping)
    # u_RH, error_RH = compute_IK(j_RH, humanoid_right_hand_pose, robot_right_hand_pose, damping)
    # u_LF, error_LF = compute_IK(j_LF, humanoid_left_foot_pose, robot_left_foot_pose, damping)
    # u_RF, error_RF = compute_IK(j_RF, humanoid_right_foot_pose, robot_right_foot_pose, damping)
    # u = u_LF + u_RF + u_LH + u_RH
    if np.sum(error) < 0.001:
        # print("IK converged", idx, error)
        print("[",idx,",",digit_pos[3,:7].tolist(),",",list(digit_dof_positions), "],")
        idx +=1
        humanoid_dof_states['pos'] = dof_pos[idx].cpu().numpy()
        humanoid_dof_states['vel'] = dof_vel[idx].cpu().numpy()
    # pos_err_vec = humanoid_hand_pose - robot_hand_pose
    # pose_error_magnitude = pos_err_vec.length()
    # pos_err_dir = pos_err_vec.normalize()
    # # pos_err = to_torch(humanoid_rb_pose[7][0]) - to_torch([robot_hand_pose.x, robot_hand_pose.y, robot_hand_pose.z])   
    # dpose[0] = pos_err_dir.x
    # dpose[1] = pos_err_dir.y
    # dpose[2] = pos_err_dir.z
    # dpose[3:] = 0
    # print(humanoid_hand_pose, robot_hand_pose,pose_error_magnitude)
    # if pose_error_magnitude < 0.1:
    #     dpose[:3] *= pose_error_magnitude
    #     # count +=1\
    # else:
    #     dpose[:3] /=100
    # dpose[3:] /= 100
    # lmbda = torch.eye(6, device=j_eef.device) * (damping ** 2)
    # j_eef_T = torch.transpose(j_eef, 1, 2)
    # u = (j_eef_T[3] @ torch.inverse(j_eef[3] @ j_eef_T[3] + lmbda) @ dpose)
    # print(u)
    # print(dpose.size(),u.size())
    digit_dof_positions += u.cpu().numpy()
    # for i in range(num_envs):
    success = gym.set_actor_dof_states(envs[3], digit_actor_handles[3], digit_dof_states, gymapi.STATE_POS)
    assert success
    # print(success)
    gym.set_actor_dof_states(env, humanoid_actor_handle, humanoid_dof_states, gymapi.STATE_ALL)    
    # gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states.contiguous()))
    gym.set_actor_root_state_tensor(sim, _root_states)
    
    # update viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
    
    
    
    
    
# while not gym.query_viewer_has_closed(viewer):
#     count += 1
#     idx = (count//5)%length
#     # step the physics
#     gym.simulate(sim)
#     gym.fetch_results(sim, True)
#     gym.refresh_dof_state_tensor(sim)
#     gym.refresh_actor_root_state_tensor(sim)
#     humanoid_pos[:,:3]=root_pos[idx,:] 
#     humanoid_pos[:,3:7]=root_rot[idx,:]
#     digit_pos[:,:3]=root_pos[idx,:]
#     digit_pos[:,3:7]=root_rot[idx,:]
#     humanoid_dof_states['pos'] = dof_pos[idx].cpu().numpy()
#     humanoid_dof_states['vel'] = dof_vel[idx].cpu().numpy()
#     retargeted_digit_dof_positions = retarget_humanoid_to_digit(humanoid_dof_positions)
    
#     for curr_dof in range(digit_num_dofs):
#         digit_dof_positions[curr_dof] = retargeted_digit_dof_positions[curr_dof]
#     gym.set_actor_dof_states(env, humanoid_actor_handle, humanoid_dof_states, gymapi.STATE_ALL)
#     gym.set_actor_dof_states(env, digit_actor_handle, digit_dof_states, gymapi.STATE_ALL)
#     gym.set_actor_root_state_tensor(sim, _root_states)




    # update the viewer
    # gym.step_graphics(sim)
    # gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
#     gym.sync_frame_time(sim)
# gym.destroy_viewer(viewer)
# gym.destroy_sim(sim)
