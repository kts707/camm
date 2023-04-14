import numpy as np
import pickle
from nnutils.geom_utils import load_skeleton
from nnutils.vis_utils import draw_skeleton_2d, get_skeleton

# draw the given kinematic chain in 2D planes (xy, xz, yz) and export a 3D mesh of the kinematic chain

skeleton_file = 'iiwa_joints.pkl'

with open(skeleton_file, 'rb') as f:
    skeleton = pickle.load(f)

skeleton_object = load_skeleton(skeleton_file,'cpu')
skel = get_skeleton(skeleton_object.joint_centers, skeleton_object.joint_connections)
skel.export('iiwa_skeleton.obj')
draw_skeleton_2d(skeleton_object.joint_centers,skeleton_object.joint_connections,'iiwa_kinematic_chain.png',title='iiwa kinematic chain')

# print out the sturcture of the kinematic chain
for key in skeleton.keys():
    print(key,'is mapped to',skeleton[key][2])