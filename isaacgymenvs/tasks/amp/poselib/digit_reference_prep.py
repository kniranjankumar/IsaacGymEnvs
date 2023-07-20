#!/usr/bin/env python
# coding: utf-8

# In[19]:


from isaacgym.torch_utils import *
import torch
import json
import numpy as np

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive, plot_skeleton_states


# In[2]:


import xml.etree.ElementTree as ET
path = "/home/dan/Projects/dynamic_motion_imitation/IsaacGymEnvs/assets/urdf/DigitRobot/DigitRobot/urdf/digit_model.urdf"


# In[3]:


tree = ET.parse(path)
xml_doc_root = tree.getroot()


# In[4]:


node_names = []
parent_indices = [-1]
local_translation = [np.array([0,0,0])]
for joint in xml_doc_root.findall("joint"):
    # print(joint.attrib.get("name"), joint.attrib.get("type"), joint.find("parent").attrib.get("link"), joint.find("child").attrib.get("link"))
    # if joint.attrib.get("type") == "revolute":

        if not joint.find("parent").attrib.get("link") in node_names:
            node_names.append(joint.find("parent").attrib.get("link"))
        parent_indices.append(node_names.index(joint.find("parent").attrib.get("link")))
        if not joint.find("child").attrib.get("link") in node_names:
            node_names.append(joint.find("child").attrib.get("link"))
            local_translation.append(np.fromstring(joint.find("origin").attrib.get("xyz"), dtype=float, sep=" "))
        else:
            print("duplicate")
            assert False
# print(len(node_names), len(parent_indices), len(local_translation))
print(local_translation)
print(node_names)
skel_tree = SkeletonTree(node_names, torch.tensor(parent_indices), torch.tensor(torch.from_numpy(np.array(local_translation))))


# In[9]:


zero_pose = SkeletonState.zero_pose(skel_tree)


# In[10]:


# plot_skeleton_state(zero_pose)


# In[11]:


x_np = np.load("/home/dan/Projects/dynamic_motion_imitation/IsaacGymEnvs/isaacgymenvs/retargetted_digit_body_orientations.npy", allow_pickle=True)
tensor_backend = {
        "arr": x_np,
        "context": {
            "dtype": x_np.dtype.name
        }}
idx = 0
rot = []
for x_np_ in x_np:
    rot_dict = x_np_
    rot_dict["torso"] = np.array([0,0,0,1])
    rot_list = np.array([rot_dict[name] for name in node_names])
    rot.append(torch.from_numpy(rot_list))
motion_rot = torch.stack(rot,dim=0)
    

translation = torch.zeros([x_np.shape[0],3])
translation[:,0]=torch.linspace(0,2,x_np.shape[0])
translation[:,1]=0
translation[:,2]=1.0

walk_pose =  SkeletonState.from_rotation_and_root_translation(
                         skeleton_tree=skel_tree,
                         r=motion_rot,
                         t=translation,
                         is_local=False
                     )
walk_motion = SkeletonMotion.from_skeleton_state(walk_pose,200)
walk_motion_cropped = walk_motion.crop(0, 350)
walk_motion_cropped.to_file("/home/dan/Projects/dynamic_motion_imitation/IsaacGymEnvs/assets/amp/digit_motions/digit_walk.npy")
# plot_skeleton_motion_interactive(walk_motion)

