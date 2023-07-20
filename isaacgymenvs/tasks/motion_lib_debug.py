from isaacgym.torch_utils import *
from isaacgymenvs.tasks.amp.utils_amp.motion_lib_digit import MotionLib
import torch
import numpy as np

motion_file = "/home/dan/Projects/dynamic_motion_imitation/IsaacGymEnvs/isaacgymenvs/tasks/../../assets/amp/digit_motions/digit_neutral.npy" 
num_dof = 22 
_key_body_ids = np.array([28, 23,  8, 16])
motion_lib = MotionLib(motion_file=motion_file, 
                         num_dofs=num_dof,
                         key_body_ids=_key_body_ids, 
                         device=torch.device("cuda:0"))

motion_ids = motion_lib.sample_motions(1)

root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, local_rot = motion_lib.get_motion_state(motion_ids,motion_ids)

dof_pos = motion_lib._local_rotation_to_dof(local_rot)