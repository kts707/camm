from absl import flags, app
import os
import sys
sys.path.insert(0,'')
import numpy as np
import torch
import trimesh

from nnutils.train_utils import v2s_trainer
from nnutils.geom_utils import  correct_bones, get_interpolated_skinning_weights, joint_transform, \
                                get_refined_bones_transforms, gauss_mlp_skinning, lbs

from nnutils.vis_utils import get_skeleton_vis_v1
opts = flags.FLAGS

# script specific ones
flags.DEFINE_string('canonical_mesh_path', 'example/mesh-rest.obj', 'path to the canonical mesh')
flags.DEFINE_string('output_dir', 'example/output_sequence', 'output directory')


def main(_):
    trainer = v2s_trainer(opts, is_eval=True)
    data_info = trainer.init_dataset()    
    trainer.define_model(data_info)

    model = trainer.model
    model.eval()

    assert opts.skeleton_file != ''

    bones_rst = trainer.model.bones
    bones_rst, _ = correct_bones(trainer.model, bones_rst)


    if opts.skeleton_bone_residual > 0:
        print('residual updated')
        clipped_residuals = torch.tanh(trainer.model.skel_bone_residuals) * opts.skeleton_bone_residual
        trainer.model.skeleton.update_skeleton_with_residuals(clipped_residuals)

    num_skeleton_bone = trainer.model.skeleton.joint_centers.shape[0]

    bone_to_skeleton_pairs = get_interpolated_skinning_weights(trainer.model.skeleton, bones_rst)

    if not os.path.exists(opts.output_dir):
        os.makedirs(opts.output_dir, exist_ok=True)

    canonical_mesh = trimesh.load(opts.canonical_mesh_path)
    pts_can = canonical_mesh.vertices

    pts_can = torch.from_numpy(pts_can).float().to(trainer.model.device)

    for i in range(50):
        relative_transform = torch.Tensor([1,0,0,0,1,0,0,0,1,0,0,0]).view(1,12).repeat(1,num_skeleton_bone,1)

        ######################## Define rigid transformation of kinematic chain links here ###################################
        theta1 = np.radians(-30/49 * i)
        c1, s1 = np.cos(theta1), np.sin(theta1)
        rotation1 = torch.Tensor([c1,-s1,0,s1,c1,0,0,0,1,0,0,0])
        relative_transform[0,17,:] = rotation1

        theta2 = np.radians(-30/49 * i)
        c2, s2 = np.cos(theta2), np.sin(theta2)
        rotation2 = torch.Tensor([c2,-s2,0,s2,c2,0,0,0,1,0,0,0])
        relative_transform[0,15,:] = rotation2

        theta3 = np.radians(30 / 49 * i)
        c3, s3 = np.cos(theta3), np.sin(theta3)
        rotation3 = torch.Tensor([c3,-s3,0,s3,c3,0,0,0,1,0,0,0])
        relative_transform[0,3,:] = rotation3

        theta4 = np.radians(-20 / 49 * i)
        c4, s4 = np.cos(theta4), np.sin(theta4)
        rotation4 = torch.Tensor([c4,-s4,0,s4,c4,0,0,0,1,0,0,0])
        relative_transform[0,6,:] = rotation4

        #########################################################################################################################

        relative_transform = relative_transform.view(-1,num_skeleton_bone*12)

        rest_pose_code = trainer.model.rest_pose_code
        rest_pose_code = rest_pose_code(torch.Tensor([0]).long().to(trainer.model.device))
        nerf_skin = trainer.model.nerf_skin

        skeleton_joint_transforms = trainer.model.skeleton.forward_kinematics(relative_transform)

        regulated_locations = joint_transform(trainer.model.skeleton.joint_centers, skeleton_joint_transforms[0], is_vec=True)
        regulated_locations = regulated_locations[0]


        skel = get_skeleton_vis_v1(regulated_locations, trainer.model.skeleton.joint_connections)
        skel.export(os.path.join(opts.output_dir, 'skel-'+str(i)+'.obj'))

        refinement_transform = get_refined_bones_transforms(bones_rst, regulated_locations, bone_to_skeleton_pairs, canonical_mesh.vertices.shape[0], trainer.model.device)

        skin_forward = gauss_mlp_skinning(pts_can[:,None], trainer.model.embedding_xyz, bones_rst, 
                            rest_pose_code, nerf_skin, use_hs=opts.use_hs, skin_aux=trainer.model.skin_aux)

        pts_dfm, _ = lbs(bones_rst, refinement_transform,
                        skin_forward, pts_can[:,None], backward=False)       

        canonical_mesh.vertices = pts_dfm.squeeze(1).detach().cpu().numpy()
        canonical_mesh.export(os.path.join(opts.output_dir,'mesh-'+str(i)+'.obj'))

    for i in range(50):
        relative_transform = torch.Tensor([1,0,0,0,1,0,0,0,1,0,0,0]).view(1,12).repeat(1,num_skeleton_bone,1)

        ######################## Define rigid transformation of kinematic chain links here ###################################
        theta1 = np.radians(-30/49 * (49-i))
        c1, s1 = np.cos(theta1), np.sin(theta1)
        rotation1 = torch.Tensor([c1,-s1,0,s1,c1,0,0,0,1,0,0,0])
        relative_transform[0,17,:] = rotation1

        theta2 = np.radians(-30/49 * (49-i))
        c2, s2 = np.cos(theta2), np.sin(theta2)
        rotation2 = torch.Tensor([c2,-s2,0,s2,c2,0,0,0,1,0,0,0])
        relative_transform[0,15,:] = rotation2

        theta3 = np.radians(30 / 49 * (49-i))
        c3, s3 = np.cos(theta3), np.sin(theta3)
        rotation3 = torch.Tensor([c3,-s3,0,s3,c3,0,0,0,1,0,0,0])
        relative_transform[0,3,:] = rotation3

        theta4 = np.radians(-20 / 49 * (49-i))
        c4, s4 = np.cos(theta4), np.sin(theta4)
        rotation4 = torch.Tensor([c4,-s4,0,s4,c4,0,0,0,1,0,0,0])
        relative_transform[0,6,:] = rotation4

        #########################################################################################################################

        relative_transform = relative_transform.view(-1,num_skeleton_bone*12)

        rest_pose_code = trainer.model.rest_pose_code
        rest_pose_code = rest_pose_code(torch.Tensor([0]).long().to(trainer.model.device))
        nerf_skin = trainer.model.nerf_skin

        skeleton_joint_transforms = trainer.model.skeleton.forward_kinematics(relative_transform)


        regulated_locations = joint_transform(trainer.model.skeleton.joint_centers, skeleton_joint_transforms[0], is_vec=True)
        regulated_locations = regulated_locations[0]


        skel = get_skeleton_vis_v1(regulated_locations, trainer.model.skeleton.joint_connections)
        skel.export(os.path.join(opts.output_dir,'skel-'+str(50+i)+'.obj'))

        refinement_transform = get_refined_bones_transforms(bones_rst, regulated_locations, bone_to_skeleton_pairs, canonical_mesh.vertices.shape[0], trainer.model.device)
        skin_forward = gauss_mlp_skinning(pts_can[:,None], trainer.model.embedding_xyz, bones_rst, 
                            rest_pose_code, nerf_skin, use_hs=opts.use_hs, skin_aux=trainer.model.skin_aux)

        pts_dfm, _ = lbs(bones_rst, refinement_transform,
                        skin_forward, pts_can[:,None], backward=False)       

        canonical_mesh.vertices = pts_dfm.squeeze(1).detach().cpu().numpy()
        canonical_mesh.export(os.path.join(opts.output_dir, 'mesh-'+str(50+i)+'.obj'))



if __name__ == '__main__':
    app.run(main)