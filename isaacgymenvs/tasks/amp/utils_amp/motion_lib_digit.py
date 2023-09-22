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
import yaml

from ..poselib.poselib.skeleton.skeleton3d import SkeletonMotion
from ..poselib.poselib.core.rotation3d import *
from isaacgym.torch_utils import *
from isaacgymenvs.utils.torch_jit_utils import *

from isaacgymenvs.tasks.amp.digit_amp_base import DOF_BODY_IDS, DOF_OFFSETS, joint_frame_rot, shuffle_idxs, KEY_BODY_NAMES


class MotionLib():
    def __init__(self, motion_file, num_dofs, key_body_ids, device):
        self._num_dof = num_dofs
        self._key_body_ids = None
        self._device = device
        joint_r = torch.stack(list(joint_frame_rot.values()), dim=0).to(self._device)
        self.joint_r_q = quat_conjugate(quat_from_euler_xyz(joint_r[:, 0], joint_r[:, 1], joint_r[:, 2]))
        self._load_motions(motion_file)

        self.motion_ids = torch.arange(len(self._motions), dtype=torch.long, device=self._device)

        return

    def num_motions(self):
        return len(self._motions)

    def get_total_length(self):
        return sum(self._motion_lengths)

    def get_motion(self, motion_id):
        return self._motions[motion_id]

    def sample_motions(self, n):
        m = self.num_motions()
        motion_ids = np.random.choice(m, size=n, replace=True, p=self._motion_weights)

        return motion_ids

    def sample_time(self, motion_ids, truncate_time=None):
        n = len(motion_ids)
        phase = np.random.uniform(low=0.0, high=1.0, size=motion_ids.shape)
        
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert(truncate_time >= 0.0)
            motion_len -= truncate_time

        motion_time = phase * motion_len

        return motion_time

    def get_motion_length(self, motion_ids):
        return self._motion_lengths[motion_ids]

    def get_motion_state(self, motion_ids, motion_times):
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()
        num_key_bodies = len(KEY_BODY_NAMES)

        root_pos0 = np.empty([n, 3])
        root_pos1 = np.empty([n, 3])
        root_rot = np.empty([n, 4])
        root_rot0 = np.empty([n, 4])
        root_rot1 = np.empty([n, 4])
        root_vel = np.empty([n, 3])
        root_ang_vel = np.empty([n, 3])
        local_rot0 = np.empty([n, num_bodies, 4])
        local_rot1 = np.empty([n, num_bodies, 4])
        dof_vel = np.empty([n, self._num_dof])
        key_pos0 = np.empty([n, num_key_bodies, 3])
        key_pos1 = np.empty([n, num_key_bodies, 3])

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

        unique_ids = np.unique(motion_ids)
        self._key_body_ids = torch.tensor([self._motions[unique_ids[0]].skeleton_tree.node_names.index(id) for id in KEY_BODY_NAMES], device=self._device)
        for uid in unique_ids:
            ids = np.where(motion_ids == uid)
            curr_motion = self._motions[uid]

            root_pos0[ids, :]  = curr_motion.global_translation[frame_idx0[ids], 0].numpy()
            root_pos1[ids, :]  = curr_motion.global_translation[frame_idx1[ids], 0].numpy()
            # print( curr_motion.global_translation[0])
            root_rot0[ids, :] = curr_motion.global_rotation[frame_idx0[ids], 0].numpy()
            root_rot1[ids, :]  = curr_motion.global_rotation[frame_idx1[ids], 0].numpy()

            local_rot0[ids, :, :]= curr_motion.local_rotation[frame_idx0[ids]].numpy()
            local_rot1[ids, :, :] = curr_motion.local_rotation[frame_idx1[ids]].numpy()

            root_vel[ids, :] = curr_motion.global_root_velocity[frame_idx0[ids]].numpy()
            root_ang_vel[ids, :] = curr_motion.global_root_angular_velocity[frame_idx0[ids]].numpy()
            
            key_pos0[ids, :, :] = curr_motion.global_translation[frame_idx0[ids][:, np.newaxis], self._key_body_ids[np.newaxis, :]].numpy()
            key_pos1[ids, :, :] = curr_motion.global_translation[frame_idx1[ids][:, np.newaxis], self._key_body_ids[np.newaxis, :]].numpy()

            dof_vel[ids, :] = curr_motion.dof_vels[frame_idx0[ids]]

        blend = to_torch(np.expand_dims(blend, axis=-1), device=self._device)

        root_pos0 = to_torch(root_pos0, device=self._device)
        root_pos1 = to_torch(root_pos1, device=self._device)
        root_rot0 = to_torch(root_rot0, device=self._device)
        root_rot1 = to_torch(root_rot1, device=self._device)
        root_vel = to_torch(root_vel, device=self._device)
        root_ang_vel = to_torch(root_ang_vel, device=self._device)
        local_rot0 = to_torch(local_rot0, device=self._device)
        local_rot1 = to_torch(local_rot1, device=self._device)
        key_pos0 = to_torch(key_pos0, device=self._device)
        key_pos1 = to_torch(key_pos1, device=self._device)
        dof_vel = to_torch(dof_vel, device=self._device)

        root_pos = (1.0 - blend) * root_pos0 + blend * root_pos1

        root_rot = slerp(root_rot0, root_rot1, blend)

        blend_exp = blend.unsqueeze(-1)
        key_pos = (1.0 - blend_exp) * key_pos0 + blend_exp * key_pos1
        
        local_rot = slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
        dof_pos = self._local_rotation_to_dof(local_rot)
        # print(root_pos[0],root_vel[0])
        # root_pos[:,2] = 0.87
        # root_vel[:,0] = 1.0
        # print(root_vel)
        # root_pos[:,2] += 0.9
        # root_vel[:,0] = 1.0
        return root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos#, local_rot

    def _load_motions(self, motion_file):
        self._motions = []
        self._motion_lengths = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_files = []

        total_len = 0.0

        motion_files, motion_weights = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)
        for f in range(num_motion_files):
            curr_file = motion_files[f]
            print("Loading {:d}/{:d} motion files: {:s}".format(f + 1, num_motion_files, curr_file))
            curr_motion = SkeletonMotion.from_file(curr_file)
            # print(curr_motion.global_translation[0])
            # assert False
            print("fps-",curr_motion.fps)
            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps

            num_frames = curr_motion.tensor.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)

            self._motion_fps.append(motion_fps)
            self._motion_dt.append(curr_dt)
            self._motion_num_frames.append(num_frames)
 
            curr_dof_vels = self._compute_motion_dof_vels(curr_motion)
            curr_motion.dof_vels = curr_dof_vels

            self._motions.append(curr_motion)
            self._motion_lengths.append(curr_len)
            
            curr_weight = motion_weights[f]
            self._motion_weights.append(curr_weight)
            self._motion_files.append(curr_file)


        self._motion_lengths = np.array(self._motion_lengths)
        self._motion_weights = np.array(self._motion_weights)
        self._motion_weights /= np.sum(self._motion_weights)

        self._motion_fps = np.array(self._motion_fps)
        self._motion_dt = np.array(self._motion_dt)
        self._motion_num_frames = np.array(self._motion_num_frames)

        num_motions = self.num_motions()
        total_len = self.get_total_length()

        print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))

        return

    def _fetch_motion_files(self, motion_file):
        ext = os.path.splitext(motion_file)[1]
        if (ext == ".yaml"):
            dir_name = os.path.dirname(motion_file)
            motion_files = []
            motion_weights = []

            with open(os.path.join(os.getcwd(), motion_file), 'r') as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)

            motion_list = motion_config['motions']
            for motion_entry in motion_list:
                curr_file = motion_entry['file']
                curr_weight = motion_entry['weight']
                assert(curr_weight >= 0)

                curr_file = os.path.join(dir_name, curr_file)
                motion_weights.append(curr_weight)
                motion_files.append(curr_file)
        else:
            motion_files = [motion_file]
            motion_weights = [1.0]

        return motion_files, motion_weights

    def _calc_frame_blend(self, time, len, num_frames, dt):
        phase = time / len
        phase = np.clip(phase, 0.0, 1.0)

        frame_idx0 = (phase * (num_frames - 1)).astype(int)
        frame_idx1 = np.minimum(frame_idx0 + 1, num_frames - 1)
        blend = (time - frame_idx0 * dt) / dt

        return frame_idx0, frame_idx1, blend

    def _get_num_bodies(self):
        motion = self.get_motion(0)
        num_bodies = motion.num_joints
        return num_bodies

    def _compute_motion_dof_vels(self, motion):
        num_frames = motion.tensor.shape[0]
        dt = 1.0 / motion.fps
        dof_vels = []

        for f in range(num_frames - 1):
            local_rot0 = motion.local_rotation[f]
            local_rot1 = motion.local_rotation[f + 1]
            frame_dof_vel = self._local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
            # frame_dof_vel = ((self._local_rotation_to_dof(local_rot1.unsqueeze(0).to("cuda:0"))-
            #                   self._local_rotation_to_dof(local_rot0.unsqueeze(0).to("cuda:0")))/dt)[0]
            dof_vels.append(frame_dof_vel)
        
        dof_vels.append(dof_vels[-1])
        dof_vels = np.array(dof_vels)

        return dof_vels
    
    def _local_rotation_to_dof(self, local_rot):
        body_ids = DOF_BODY_IDS
        dof_offsets = DOF_OFFSETS
        n = local_rot.shape[0]

        dof_pos = torch.zeros((n, 32), dtype=torch.float, device=self._device)

        # for j in range(len(body_ids)):
        for j in range(30):
            # body_id = body_ids[j]
            body_id = j+1
            joint_q = local_rot[:, body_id]
            joint_theta, joint_axis = quat_to_angle_axis(quat_mul(self.joint_r_q[j].repeat(joint_q.size(0),1),joint_q))
            joint_theta = joint_theta * joint_axis[..., 2] # assume joint is always along y axis
            joint_theta = normalize_angle(joint_theta)
            dof_pos[:, j] = joint_theta

        shuffled_dof = torch.zeros_like(dof_pos)
        shuffled_dof = dof_pos[:, shuffle_idxs]
        shuffled_dof[:,2] *= -1 # invert hip flexion left
        shuffled_dof[:,9] *= -1 # invert hip flexion right
        shuffled_dof[:,15] *= -1 # invert shoulder pitch left
        shuffled_dof[:,19] *= -1 # invert shoulder pitch right
        # shuffled_dof[:,14] *= -1 # invert shoulder pitch left
        # shuffled_dof[:,18] *= 0 # invert shoulder pitch right

        return shuffled_dof

    def _local_rotation_to_dof_vel(self, local_rot0, local_rot1, dt):
        body_ids = DOF_BODY_IDS
        dof_offsets = DOF_OFFSETS

        dof_vel = np.zeros([32])
        shuffled_dof_vel = np.zeros_like(dof_vel)
        diff_quat_data = quat_mul_norm(quat_inverse(local_rot0), local_rot1)
        diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
        local_vel = diff_axis * diff_angle.unsqueeze(-1) / dt
        local_vel = local_vel.numpy()

        for j in range(30):
            body_id = j+1
            joint_vel = local_vel[body_id]
            dof_vel[j] = joint_vel[2] # assume joint is always along y axis
        shuffled_dof_vel = dof_vel[shuffle_idxs]
        shuffled_dof_vel[2] *= -1 # invert hip flexion left
        shuffled_dof_vel[9] *= -1 # invert hip flexion right
        shuffled_dof_vel[15] *= -1 # invert shoulder pitch left
        shuffled_dof_vel[19] *= -1 # invert shoulder pitch right
        return shuffled_dof_vel