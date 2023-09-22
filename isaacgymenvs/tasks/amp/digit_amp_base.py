# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from ..base.vec_task import VecTask
# DOF_BODY_IDS = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
# DOF_BODY_IDS = [1,2,3,4,6,7,8,9,11,12,13,14,15,16,17,19,20,21,22,26,27,28]
# DOF_BODY_IDS = [i for i in range(22)]
DOF_BODY_IDS = [
                 1,2,3,5,6,7,
                14,15,16,18,19,20,
                8,10,11,12,13,
                21,23,24,25
                ]
# DOF_OFFSETS = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
DOF_OFFSETS = [i for i in range(23)]
joint_frame_rot = {         "hip_abduction_left": [1.57079632679, -1.1955506, -1.57079632679],
                            "hip_rotation_left": [0, -1.57079632679, 0],
                            "hip_flexion_left": [-1.57079632679, -0.785398163397, 3.14159265359],
                            "knee_joint_left": [0, 0, -1.57079632679],
                            "knee_to_shin_left": [0,0,0],
                            "shin_to_tarsus_left": [0, 0, 1.7976891],
                            "toe_pitch_joint_left": [0, 0, 1.1956],
                            "toe_roll_joint_left": [0, 1.57079632679, 0],
                            
                            "shoulder_roll_joint_left": [-1.57079632679, -1.3962633, 1.57079632679],
                            "shoulder_roll_cap_left": [0.001, 0.12, 0.4],
                            "shoulder_pitch_joint_left": [1.57079632679, 0.785398163397, -0.2792527],
                            "shoulder_yaw_joint_left": [1.57079632679, 0 ,0],
                            "elbow_joint_left": [1.57079632679, -0.3926991, 0],
                            
                            "hip_abduction_right": [-1.57079632679, -1.1955506, 1.57079632679],
                            "hip_rotation_right": [0, -1.57079632679, 0],
                            "hip_flexion_right": [1.57079632679, -0.785398163397, -3.14159265359],
                            "knee_joint_right": [0, 0, 1.57079632679],
                            "knee_to_shin_right": [0,0,0],
                            "shin_to_tarsus_right": [0, 0, -1.7976891],
                            "toe_pitch_joint_right": [0, 0, -1.1956],
                            "toe_roll_joint_right": [0, 1.57079632679, 0],
                            "shoulder_roll_joint_right": [1.57079632679, -1.3962633, -1.57079632679],
                            "shoulder_cap_joint_right": [0.001, -0.12, 0.4],
                            "waist_cap_joint_right": [-0.001, -0.09, 0.0],
                            "waist_cap_joint_left": [-0.001, 0.09, 0.0],
                            "shoulder_pitch_joint_right": [-1.57079632679, 0.785398163397, 0.2792527],
                            "shoulder_yaw_joint_right": [-1.57079632679, 0 ,0],
                            "elbow_joint_right": [-1.57079632679, -0.3926991, 0],
                            "right_elbow_to_hand": [0,0,0],
                            "left_elbow_to_hand": [0,0,0]
        }




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
shuffle_idxs = [list(joint_frame_rot.keys()).index(joint) for joint in joint_order]
joint_frame_rot = {k: torch.tensor(v) for k, v in joint_frame_rot.items()}
# NUM_OBS = 13 + 22 + 22 + 12 
NUM_OBS = sum([1, 6, 3, 3, 22, 22, 12])# [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
NUM_ACTIONS = 22


KEY_BODY_NAMES = ["right_hand", "left_hand", "left_toe_roll", "right_toe_roll"]

class DigitAMPBase(VecTask):

    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = config

        self._pd_control = self.cfg["env"]["pdControl"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.randomize = self.cfg["task"]["randomize"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.camera_follow = self.cfg["env"].get("cameraFollow", False)
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._contact_bodies = self.cfg["env"]["contactBodies"]
        self._termination_height = self.cfg["env"]["terminationHeight"]
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()

        self.randomization_params = self.cfg["task"]["randomization_params"]
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        
        dt = self.cfg["sim"]["dt"]
        self.dt = self.control_freq_inv * dt
        
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        sensors_per_env = 2
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        self._initial_root_states = self._root_states.clone()
        self._initial_root_states[:, 7:13] = 0
        self._initial_root_states[:,3:6] = 0
        self._initial_root_states[:,2]=0.93
        

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self._dof_pos = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self._dof_vel = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)
        right_shoulder_x_handle = self.gym.find_actor_dof_handle(self.envs[0], self.digit_handles[0], "right_shoulder_x")
        left_shoulder_x_handle = self.gym.find_actor_dof_handle(self.envs[0], self.digit_handles[0], "left_shoulder_x")
        # self._initial_dof_pos[:, right_shoulder_x_handle] = 0.5 * np.pi
        # self._initial_dof_pos[:, left_shoulder_x_handle] = -0.5 * np.pi
        # self._initial_dof_pos[:,0] = 0.28
        # self._initial_dof_pos[:,7] = -0.28

        # self._initial_dof_pos[:,5] = 0.1
        # self._initial_dof_pos[:,12] = -0.1
        self._initial_dof_pos=torch.tensor([[ 2.7886e-01, -1.5718e-03,  2.4364e-01, -1.6560e-01,  1.8958e-01,
         1.0037e-01,  8.0998e-06, -2.7884e-01,  1.6481e-03, -2.4295e-01,
         1.6479e-01, -1.8975e-01, -1.0040e-01,  1.0898e-04, -9.6694e-04,
         9.0086e-01,  1.3919e-05,  3.5007e-04,  9.6761e-04, -9.0091e-01,
         2.6128e-06, -3.3921e-04]]).repeat(self.num_envs,1).to(self.device)
        self._initial_dof_pos[:,0] = 0.33
        self._initial_dof_pos[:,7] = -0.33
        # hip flexion
        self._initial_dof_pos[:,2] = -0.
        self._initial_dof_pos[:,9] = 0.
        # shoulder pitch
        self._initial_dof_pos[:,15] = 0.9
        self._initial_dof_pos[:,19] = -0.9
        # toe pitch
        self._initial_dof_pos[:,5] = 0.2*0
        self._initial_dof_pos[:,12] = -0.2*0
        # elbow
        self._initial_dof_pos[:,17] = 0.5
        self._initial_dof_pos[:,21] = -0.5
        # shin to tarsus
        self._initial_dof_pos[:,4] = -0.3*0
        self._initial_dof_pos[:,11] = 0.3*0
        #knee
        self._initial_dof_pos[:,3] = 0.0
        self._initial_dof_pos[:,10] = -0.0
        
        self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)
        
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        self._rigid_body_pos = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 0:3]
        self._rigid_body_rot = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 3:7]
        self._rigid_body_vel = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 10:13]
        self._contact_forces = gymtorch.wrap_tensor(contact_force_tensor).view(self.num_envs, self.num_bodies, 3)
        
        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.knee_tarsus_rod_length = 0.5518
        self.toe_tarsus_rod_length = 0.29
        self._rod_kp = 1000
        self._rod_kd = 100
        
        if self.viewer != None:
            self._init_camera()
            
        return

    def get_obs_size(self):
        return NUM_OBS

    def get_action_size(self):
        return NUM_ACTIONS

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        return

    def reset_idx(self, env_ids):
        self._reset_actors(env_ids)
        self._refresh_sim_tensors()
        self._compute_observations(env_ids)
        return

    def set_char_color(self, col):
        for i in range(self.num_envs):
            env_ptr = self.envs[i]
            handle = self.digit_handles[i]

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                              gymapi.Vec3(col[0], col[1], col[2]))

        return

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../assets')
        # asset_file = "mjcf/amp_humanoid.xml"
        asset_file = "urdf/DigitRobot/DigitRobot/urdf/digit_model.urdf"

        if "asset" in self.cfg["env"]:
            #asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        # asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0001
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False
        asset_options.fix_base_link = False
        # asset_options.vhacd_enabled = True
        digit_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)


        # print(motor_efforts)
        # assert False
        # create force sensors at the feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(digit_asset, "right_toe_roll")
        left_foot_idx = self.gym.find_asset_rigid_body_index(digit_asset, "left_toe_roll")
        sensor_pose = gymapi.Transform()

        self.gym.create_asset_force_sensor(digit_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(digit_asset, left_foot_idx, sensor_pose)



        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(digit_asset)
        self.num_dof = self.gym.get_asset_dof_count(digit_asset)
        self.num_joints = self.gym.get_asset_joint_count(digit_asset)
        actuator_props = self.gym.get_asset_dof_properties(digit_asset)
        for i in range(self.num_dof):
            actuator_props['driveMode'][i] = gymapi.DOF_MODE_POS
            actuator_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"] #self.Kp
            actuator_props['damping'][i] = self.cfg["env"]["control"]["damping"] #self.Kd
            actuator_props["armature"][i] = 0.0001
        motor_efforts = [prop.item() for prop in actuator_props["effort"]]
        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(1.0, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        self.digit_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        
        toe_joints = [i for i,name in enumerate(joint_order) if "toe" in name]
        self.toe_joints = toe_joints
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            contact_filter = 1
            
            handle = self.gym.create_actor(env_ptr, digit_asset, start_pose, "digit", i, contact_filter, 0)
            self.gym.set_actor_dof_properties(env_ptr, handle, actuator_props)
            actuator_props = self.gym.get_actor_dof_properties(env_ptr, handle)
            
            for i in range(self.num_dof):
                actuator_props['driveMode'][i] = gymapi.DOF_MODE_POS
                actuator_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"] #self.Kp
                actuator_props['damping'][i] = self.cfg["env"]["control"]["damping"] #self.Kd
                actuator_props["armature"][i] = 0.0001
                actuator_props["effort"][i] = 500
                if i in toe_joints:
                    actuator_props["effort"][i] = 5000
                    actuator_props['stiffness'][i] = 5000 #self.Kp
                    # actuator_props["upper"][i] = 0
                    # actuator_props["lower"][i] = 0
                    
                
            self.gym.set_actor_dof_properties(env_ptr, handle, actuator_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, handle)

            # for j in range(self.num_bodies):
            #     self.gym.set_rigid_body_color(
            #         env_ptr, handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.4706, 0.549, 0.6863))

            self.envs.append(env_ptr)
            self.digit_handles.append(handle)


        dof_prop = actuator_props #self.gym.get_actor_dof_properties(env_ptr, handle)
        # print(self.gym.get_actor_dof_names(env_ptr, handle))
        # assert False
        for j in range(self.num_dof):
            # print(j,self.gym.get_actor_dof_names(env_ptr, handle))

            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])
            

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)
        # print(self.dof_limits_lower, self.dof_limits_upper)
        # assert False
        self._key_body_ids = self._build_key_body_ids_tensor(env_ptr, handle)
        self._contact_body_ids = self._build_contact_body_ids_tensor(env_ptr, handle)
        
        if (self._pd_control):
            self._build_pd_action_offset_scale()

        return

    def _build_pd_action_offset_scale(self):
        num_joints = len(DOF_OFFSETS) - 1
        
        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        # for j in range(num_joints):
        #     dof_offset = DOF_OFFSETS[j]
        #     dof_size = DOF_OFFSETS[j + 1] - DOF_OFFSETS[j]

        #     if (dof_size == 3):
        #         lim_low[dof_offset:(dof_offset + dof_size)] = -np.pi
        #         lim_high[dof_offset:(dof_offset + dof_size)] = np.pi

        #     elif (dof_size == 1):
        #         curr_low = lim_low[dof_offset]
        #         curr_high = lim_high[dof_offset]
        #         curr_mid = 0.5 * (curr_high + curr_low)
                
        #         # extend the action range to be a bit beyond the joint limits so that the motors
        #         # don't lose their strength as they approach the joint limits
        #         curr_scale = 0.7 * (curr_high - curr_low)
        #         curr_low = curr_mid - curr_scale
        #         curr_high = curr_mid + curr_scale

        #         lim_low[dof_offset] = curr_low
        #         lim_high[dof_offset] =  curr_high

        self._pd_action_offset = 0. * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset[0] = 0.33
        self._pd_action_offset[7] = -0.33
        # hip flexion
        self._pd_action_offset[2] = -0.0
        self._pd_action_offset[9] = 0.0
        # shoulder pitch
        self._pd_action_offset[15] = 0.9
        self._pd_action_offset[19] = -0.9
        # toe pitch
        self._pd_action_offset[5] = 0.2*0
        self._pd_action_offset[12] = -0.2*0
        # elbow
        self._pd_action_offset[17] = 0.5
        self._pd_action_offset[21] = -0.5
        # shin to tarsus
        self._pd_action_offset[4] = -0.3*0
        self._pd_action_offset[11] = 0.3*0
        #knee
        self._pd_action_offset[3] = 0.0
        self._pd_action_offset[10] = -0.0
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)

        return

    def _compute_reward(self, actions):
        self.rew_buf[:] = compute_digit_reward(self.obs_buf)
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_digit_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces, self._contact_body_ids,
                                                   self._rigid_body_pos, self._rigid_body_rot, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_height)
        return

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return

    def _compute_observations(self, env_ids=None):
        obs = self._compute_digit_obs(env_ids)

        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs

        return

    def _compute_digit_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._root_states
            dof_pos = self._dof_pos
            dof_vel = self._dof_vel
            key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        else:
            root_states = self._root_states[env_ids]
            dof_pos = self._dof_pos[env_ids]
            dof_vel = self._dof_vel[env_ids]
            key_body_pos = self._rigid_body_pos[env_ids][:, self._key_body_ids, :]
        
        obs = compute_digit_observations(root_states, dof_pos, dof_vel,
                                            key_body_pos, self._local_root_obs)
        return obs

    def _reset_actors(self, env_ids):
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.simulate(self.sim)



        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return

    def pre_physics_step(self, actions):
        ## compute rod forces
        # print(self._dof_pos[0])
        rtoe_offset = [[-0.049,0.01,0.0]]
        rtarsus_offset = [[0.11,0.085,0]]
        ltoe_offset = [[-0.049,-0.01,0.0]]
        ltarsus_offset = [[0.11,-0.085,0]]
        rtoe_point = tf_apply(self._rigid_body_rot[:,15],self._rigid_body_pos[:,15],to_torch(rtoe_offset).repeat(self.num_envs,1))
        rtarsus_point = tf_apply(self._rigid_body_rot[:,14],self._rigid_body_pos[:,14],to_torch(rtarsus_offset).repeat(self.num_envs,1))
        ltoe_point = tf_apply(self._rigid_body_rot[:,7],self._rigid_body_pos[:,7],to_torch(ltoe_offset).repeat(self.num_envs,1))
        ltarsus_point = tf_apply(self._rigid_body_rot[:,6],self._rigid_body_pos[:,6],to_torch(ltarsus_offset).repeat(self.num_envs,1))
        
        rknee_offset = [[-0.02, 0.1, 0.0]]
        lknee_offset = [[-0.02, -0.1, 0.0]]
        rtarsus_offset2 = [[-0.1,0.01,0]]
        ltarsus_offset2 = [[-0.1,-0.01,0]]
        rknee_point = tf_apply(self._rigid_body_rot[:,12],self._rigid_body_pos[:,12],to_torch(rknee_offset).repeat(self.num_envs,1))
        lknee_point = tf_apply(self._rigid_body_rot[:,4],self._rigid_body_pos[:,4],to_torch(lknee_offset).repeat(self.num_envs,1))
        rtarsus_point2 = tf_apply(self._rigid_body_rot[:,14],self._rigid_body_pos[:,14],to_torch(rtarsus_offset2).repeat(self.num_envs,1))
        ltarsus_point2 = tf_apply(self._rigid_body_rot[:,6],self._rigid_body_pos[:,6],to_torch(ltarsus_offset2).repeat(self.num_envs,1))
        rknee_point_vel = self._rigid_body_vel[:,12]+torch.cross(self._rigid_body_ang_vel[:,12],to_torch(rknee_offset).repeat(self.num_envs,1))
        lknee_point_vel = self._rigid_body_vel[:,4]+torch.cross(self._rigid_body_ang_vel[:,4],to_torch(lknee_offset).repeat(self.num_envs,1))
        rtarsus_point2_vel = self._rigid_body_vel[:,14]+torch.cross(self._rigid_body_ang_vel[:,14],to_torch(rtarsus_offset2).repeat(self.num_envs,1))
        ltarsus_point2_vel = self._rigid_body_vel[:,6]+torch.cross(self._rigid_body_ang_vel[:,6],to_torch(ltarsus_offset2).repeat(self.num_envs,1))
        
        rknee_tarsus_rod = rknee_point - rtarsus_point2
        lknee_tarsus_rod = lknee_point - ltarsus_point2
        rtoe_tarsus_rod = rtoe_point - rtarsus_point
        ltoe_tarsus_rod = ltoe_point - ltarsus_point
        
        rknee_tarsus_rod_direction = rknee_tarsus_rod/torch.norm(rknee_tarsus_rod, dim=1).unsqueeze(1)
        lknee_tarsus_rod_direction = lknee_tarsus_rod/torch.norm(lknee_tarsus_rod, dim=1).unsqueeze(1)
        rknee_tarsus_projected_vel_error = torch.sum(rknee_tarsus_rod_direction*rknee_point_vel, dim=1)-torch.sum(rknee_tarsus_rod_direction*rtarsus_point2_vel, dim=1)
        lknee_tarsus_projected_vel_error = torch.sum(lknee_tarsus_rod_direction*lknee_point_vel, dim=1)-torch.sum(lknee_tarsus_rod_direction*ltarsus_point2_vel, dim=1)
        rknee_tarsus_rod_force = self._rod_kp * (torch.norm(rknee_tarsus_rod, dim=1) - self.knee_tarsus_rod_length) + self._rod_kd*(rknee_tarsus_projected_vel_error)
        lknee_tarsus_rod_force = self._rod_kp * (torch.norm(lknee_tarsus_rod, dim=1) - self.knee_tarsus_rod_length) + self._rod_kd*(lknee_tarsus_projected_vel_error)
        rtoe_tarsus_rod_force = self._rod_kp * (torch.norm(rtoe_tarsus_rod, dim=1) - self.toe_tarsus_rod_length)
        ltoe_tarsus_rod_force = self._rod_kp * (torch.norm(ltoe_tarsus_rod, dim=1) - self.toe_tarsus_rod_length)
        
        forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        force_positions = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        force_positions[:, 15, :] = to_torch(rtoe_offset)
        force_positions[:, 7, :] = to_torch(ltoe_offset)
        # force_positions[:, 14, :] = rtarsus_offset
        # force_positions[:, 6, :] = ltarsus_offset
        
        force_positions[:, 12, :] = to_torch(rknee_offset)
        force_positions[:, 4, :]  = to_torch(lknee_offset)
        force_positions[:, 14, :] = to_torch(rtarsus_offset2)
        force_positions[:, 6, :]  = to_torch(ltarsus_offset2)
        
        # print("shapes", rknee_tarsus_rod_force.shape, rknee_tarsus_rod.shape)
        forces[:, 12, :] = rknee_tarsus_rod_force.unsqueeze(1)*rknee_tarsus_rod_direction
        forces[:, 4, :] = lknee_tarsus_rod_force.unsqueeze(1)*lknee_tarsus_rod_direction
        forces[:, 14, :] = -forces[:, 12, :]
        forces[:, 6, :] = -forces[:, 4, :]
        self.gym.apply_rigid_body_force_at_pos_tensors(self.sim, gymtorch.unwrap_tensor(forces), 
                                                  gymtorch.unwrap_tensor(force_positions), 
                                                  gymapi.LOCAL_SPACE)

        # self.gym.apply_rigid_body_force_at_pos_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(force_positions), gymapi.ENV_SPACE)
        
        self.actions = actions.to(self.device).clone()

        if (self._pd_control):
            pd_tar = self._action_to_pd_targets(self.actions)
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            # print("here", pd_tar)
            # for idx in self.toe_joints:
            #     pd_tar[:,idx] =0 # disable toe joints
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        else:
            forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
            force_tensor = gymtorch.unwrap_tensor(forces)
            self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

        return

    def post_physics_step(self):
        self.progress_buf += 1

        self._refresh_sim_tensors()
        self._compute_observations()
        self._compute_reward(self.actions)
        self._compute_reset()
        
        self.extras["terminate"] = self._terminate_buf

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return

    def render(self):
        if self.viewer and self.camera_follow:
            self._update_camera()

        super().render()
        return

    def _build_key_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in KEY_BODY_NAMES:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in self._contact_bodies:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _action_to_pd_targets(self, action):
        # action *=0
        # # hip abduction
        # action[:,0] = 0.33
        # action[:,7] = -0.33
        # # hip flexion
        # action[:,2] = -0.0
        # action[:,9] = 0.0
        # # shoulder pitch
        # action[:,15] = 0.9
        # action[:,19] = -0.9
        # # toe pitch
        # action[:,5] = 0.2*0
        # action[:,12] = -0.2*0
        # # elbow
        # action[:,17] = 0.5
        # action[:,21] = -0.5
        # # shin to tarsus
        # action[:,4] = -0.3*0
        # action[:,11] = 0.3*0
        # #knee
        # action[:,3] = 0.0
        # action[:,10] = -0.0
        # action[:,6] = -1.2
        # action[:,14] = 1.2
        # action[:,15] = 1.5
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._root_states[0, 0:3].cpu().numpy()
        
        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], 
                              self._cam_prev_char_pos[1] - 3.0, 
                              1.0)
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0],
                                 self._cam_prev_char_pos[1],
                                 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._root_states[0, 0:3].cpu().numpy()
        
        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0], 
                                  char_root_pos[1] + cam_delta[1], 
                                  cam_pos[2])

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)
        return

#####################################################################
###=========================jit functions=========================###
#####################################################################

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

@torch.jit.script
def compute_digit_reward(obs_buf):
    # type: (Tensor) -> Tensor
    reward = torch.ones_like(obs_buf[:, 0])
    return reward

@torch.jit.script
def compute_digit_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos, rigid_body_rot,
                           max_episode_length, enable_early_termination, termination_height):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, float) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(masked_contact_buf > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_height
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)
        has_fallen = torch.logical_or(has_fallen, rigid_body_pos[:,0, 2] < termination_height)
        has_fallen = torch.logical_or(has_fallen, rigid_body_pos[:,0, 2]>1.5)
        torso_orientation = get_euler_xyz(rigid_body_rot[:, 0, :])
        has_fallen = torch.logical_or(has_fallen, torch.logical_and(5.28>torch.abs(torso_orientation[0]), torch.abs(torso_orientation[0]) > 1.0))
        has_fallen = torch.logical_or(has_fallen, torch.logical_and(5.28>torch.abs(torso_orientation[1]), torch.abs(torso_orientation[1]) > 1.0))
        
        # has_fallen = torch.logical_or(has_fallen, torch.abs(torso_orientation[1]) > 0.3)
        # has_fallen = torch.logical_or(has_fallen, torch.abs(torso_orientation[2]) > 1.0)
        
        # print(torso_orientation,rigid_body_rot[:, 0, :] , rigid_body_pos[:,0, 2] < termination_height, rigid_body_pos[:,0, 2]>1.5)
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
        # print(terminated[0], body_height[0], fall_height[0])
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated