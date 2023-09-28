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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("unprocessed_motion_file", help="path to the file", type=str, default="digit_state_hop.npy")
args = parser.parse_args()
# args.unprocessed_motion_file = ' '.join(args.unprocessed_motion_file)

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

filename = args.unprocessed_motion_file
x_np = np.load("/home/dan/Projects/dynamic_motion_imitation/T2M-GPT/digit_motion/"+filename, allow_pickle=True)

idx = 0
rot = []
translation_array = torch.zeros([x_np.shape[0],3])

for (x_np_,translation, rotation,_) in x_np:
    rot_dict = x_np_
    rot_dict["torso"] = rotation
    rot_list = np.array([rot_dict[name] for name in node_names])
    rot.append(torch.from_numpy(rot_list))
    translation_array[idx,:] = torch.tensor(translation+np.array([0,0,-0.0]))
    print(translation_array[idx,:])
    idx +=1
motion_rot = torch.stack(rot,dim=0)
    

# translation[:,0]=torch.linspace(0,2,x_np.shape[0])
# translation[:,1]=0
# translation[:,2]=1.0

walk_pose =  SkeletonState.from_rotation_and_root_translation(
                         skeleton_tree=skel_tree,
                         r=motion_rot,
                         t=translation_array,
                         is_local=False
                     )
walk_motion = SkeletonMotion.from_skeleton_state(walk_pose,200)
filename = filename.split(" ")
filename = "_".join(filename)
# walk_motion_cropped = walk_motion.crop(0, 350)
walk_motion_cropped = walk_motion
walk_motion_cropped.to_file("/home/dan/Projects/dynamic_motion_imitation/IsaacGymEnvs/assets/amp/digit_motions/"+filename)
# plot_skeleton_motion_interactive(walk_motion)
print(walk_motion_cropped.global_translation[0])

