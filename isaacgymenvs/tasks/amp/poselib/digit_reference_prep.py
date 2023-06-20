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


x_np = np.load("/home/dan/Projects/dynamic_motion_imitation/IsaacGymEnvs/isaacgymenvs/retargetted_digit_body_orientations.npy",  allow_pickle=True)
tensor_backend = {
        "arr": x_np,
        "context": {
            "dtype": x_np.dtype.name
        }}
idx = 0


# In[ ]:


# walk_pose = SkeletonState(SkeletonState._to_state_vector(torch.from_numpy(x_np), torch.Tensor([0,0,0])), skel_tree, is_local=False)
# rot = zero_pose.global_rotation.clone()
idx +=10
# rot = torch.from_numpy(x_np[idx])
# rot = torch.cat((torch.tensor([[0,0,0,1]]), rot))
skeleton_states = []
for i in range(0,len(x_np)):
    rot_dict = x_np[i]
    rot_dict["torso"] = np.array([0,0,0,1])
    rot_list = np.array([rot_dict[name] for name in node_names])
    rot = torch.from_numpy(rot_list)
    # rot = torch.cat((torch.tensor([[0,0,0,1]]), rot))
    walk_pose =  SkeletonState.from_rotation_and_root_translation(
                         skeleton_tree=skel_tree,
                         r=rot,
                         t=zero_pose.root_translation,
                         is_local=False
                     )
    skeleton_states.append(walk_pose)
    


# In[21]:


plot_skeleton_states(np.array(skeleton_states))


# In[ ]:





# In[140]:


rot.size()


# In[50]:


x_np.shape

