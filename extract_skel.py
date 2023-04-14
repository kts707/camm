from absl import flags, app
import sys
from nnutils.vis_utils import draw_skeleton_2d, get_bone_skeleton_association, get_skeleton, get_skeleton_numpy, get_skeleton_numpy_vis_v1
sys.path.insert(0,'third_party')
import numpy as np
import cv2
import trimesh
import os
import torch

from utils.io import get_bones_mesh, save_vid, str_to_frame, save_bones
from nnutils.train_utils import v2s_trainer
from nnutils.geom_utils import correct_bones, get_interpolated_skinning_weights, load_skeleton, obj_to_cam, tensor2array, vec_to_sim3, obj_to_cam
from ext_utils.flowlib import cat_imgflo 
opts = flags.FLAGS
                
def save_output(rendered_seq, aux_seq, seqname, skeleton, save_flo, bone_to_skeleton_pairs):
    save_dir = '%s/'%(opts.model_path.rsplit('/',1)[0]+'-rendering')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    length = len(aux_seq['mesh'])
    mesh_rest = aux_seq['mesh_rest']
    len_max = (mesh_rest.vertices.max(0) - mesh_rest.vertices.min(0)).max()
    mesh_rest.export('%s/mesh-rest.obj'%save_dir)
    if 'mesh_rest_skin' in aux_seq.keys():
        aux_seq['mesh_rest_skin'].export('%s/mesh-rest-skin.obj'%save_dir)

    # save canonical skeleton
    canonical_skeleton = get_skeleton_numpy(skeleton.joint_centers.detach().cpu().numpy(), skeleton.joint_connections)
    canonical_skeleton.export('%s/skeleton-rest.obj'%save_dir)

    flo_gt_vid = []
    flo_p_vid = []
    for i in range(length):
        impath = aux_seq['impath'][i]
        seqname = impath.split('/')[-2]
        save_prefix = '%s/%s'%(save_dir,seqname)
        idx = int(impath.split('/')[-1].split('.')[-2])
        mesh = aux_seq['mesh'][i]
        rtk = aux_seq['rtk'][i]

        if 'skeleton' in aux_seq.keys() and len(aux_seq['skeleton'])>0:
            skeleton_path = '%s-skeleton-%05d.obj'%(save_prefix, idx)
            skel_as_mesh = get_skeleton_numpy_vis_v1(aux_seq['skeleton'][i], skeleton.joint_connections)
            skel_as_mesh.export(skeleton_path)

            if 'bone' in aux_seq.keys() and len(aux_seq['bone'])>0:
                bones = aux_seq['bone'][i]
                bone_path = '%s-bone-%05d.obj'%(save_prefix, idx)
                save_bones(bones, len_max/5, bone_path)

                ends = np.zeros((bones.shape[0],3))
                for bone_idx in range(bones.shape[0]):
                    pair_stats = bone_to_skeleton_pairs[bone_idx]
                    parent_joint_idx, child_joint_idx = pair_stats[0][0], pair_stats[0][1]
                    ends[bone_idx,:] = aux_seq['skeleton'][i][parent_joint_idx] + pair_stats[1][1] * (aux_seq['skeleton'][i][child_joint_idx] - aux_seq['skeleton'][i][parent_joint_idx])
                bone_skeleton_association = get_bone_skeleton_association(bones[:,:3], ends)
                bone_skeleton_association = trimesh.util.concatenate([skel_as_mesh, bone_skeleton_association])
                bone_skeleton_association.export('%s-bone_skeleton_association-%05d.obj'%(save_prefix, idx))

        mesh.export('%s-mesh-%05d.obj'%(save_prefix, idx))
        np.savetxt('%s-cam-%05d.txt'  %(save_prefix, idx), rtk)
            
        img_gt = rendered_seq['img'][i]
        flo_gt = rendered_seq['flo'][i]
        mask_gt = rendered_seq['sil'][i][...,0]
        flo_gt[mask_gt<=0] = 0
        img_gt[mask_gt<=0] = 1
        if save_flo: img_gt = cat_imgflo(img_gt, flo_gt)
        else: img_gt*=255
        cv2.imwrite('%s-img-gt-%05d.jpg'%(save_prefix, idx), img_gt[...,::-1])
        flo_gt_vid.append(img_gt)
        
        img_p = rendered_seq['img_coarse'][i]
        flo_p = rendered_seq['flo_coarse'][i]
        mask_gt = cv2.resize(mask_gt, flo_p.shape[:2][::-1]).astype(bool)
        flo_p[mask_gt<=0] = 0
        img_p[mask_gt<=0] = 1
        if save_flo: img_p = cat_imgflo(img_p, flo_p)
        else: img_p*=255
        cv2.imwrite('%s-img-p-%05d.jpg'%(save_prefix, idx), img_p[...,::-1])
        flo_p_vid.append(img_p)

        flo_gt = cv2.resize(flo_gt, flo_p.shape[:2])
        flo_err = np.linalg.norm( flo_p - flo_gt ,2,-1)
        flo_err_med = np.median(flo_err[mask_gt])
        flo_err[~mask_gt] = 0.
        cv2.imwrite('%s-flo-err-%05d.jpg'%(save_prefix, idx), 
                128*flo_err/flo_err_med)

        img_gt = rendered_seq['img'][i]
        img_p = rendered_seq['img_coarse'][i]
        img_gt = cv2.resize(img_gt, img_p.shape[:2][::-1])
        img_err = np.power(img_gt - img_p,2).sum(-1)
        img_err_med = np.median(img_err[mask_gt])
        img_err[~mask_gt] = 0.
        cv2.imwrite('%s-img-err-%05d.jpg'%(save_prefix, idx), 
                128*img_err/img_err_med)

    upsample_frame = min(30, len(flo_p_vid))
    save_vid('%s-img-p' %(save_prefix), flo_p_vid, upsample_frame=upsample_frame)
    save_vid('%s-img-gt' %(save_prefix),flo_gt_vid,upsample_frame=upsample_frame)

def main(_):
    trainer = v2s_trainer(opts, is_eval=True)
    data_info = trainer.init_dataset()    
    trainer.define_model(data_info)
    seqname=opts.seqname

    dynamic_mesh = opts.flowbw or opts.lbs
    idx_render = str_to_frame(opts.test_frames, data_info)

    trainer.model.img_size = opts.render_size
    chunk = opts.frame_chunk

    bones_rst = trainer.model.bones
    bones_rst, _ = correct_bones(trainer.model, bones_rst)

    assert opts.skeleton_file != ''
    if opts.skeleton_bone_residual > 0:
        constructed_skeleton = load_skeleton(opts.skeleton_file, trainer.model.device, residual_update=True)
        unchanged_skel = get_skeleton(constructed_skeleton.joint_centers, constructed_skeleton.joint_connections)
        unchanged_skel.export((opts.model_path[:-17]+'canonical_skel_unchanged.obj'))
        clipped_residuals = torch.tanh(trainer.model.skel_bone_residuals) * opts.skeleton_bone_residual
        
        trainer.model.skeleton.update_skeleton_with_residuals(clipped_residuals)
        learned_skel = get_skeleton(trainer.model.skeleton.joint_centers, trainer.model.skeleton.joint_connections)
        learned_skel.export((opts.model_path[:-17]+'canonical_skel_learned.obj'))

    bone_to_skeleton_pairs = get_interpolated_skinning_weights(trainer.model.skeleton, bones_rst)

    draw_skeleton_2d(trainer.model.skeleton.joint_centers, trainer.model.skeleton.joint_connections, (opts.model_path[:-17]+'skeleton_2d.png'))
    ends = np.zeros((bones_rst.shape[0],3))
    joint_centers = trainer.model.skeleton.joint_centers.detach().cpu().numpy()
    for bone_idx in range(bones_rst.shape[0]):
        pair_stats = bone_to_skeleton_pairs[bone_idx]
        parent_joint_idx, child_joint_idx = pair_stats[0][0], pair_stats[0][1]
        ends[bone_idx,:] = joint_centers[parent_joint_idx] + pair_stats[1][1] * (joint_centers[child_joint_idx] - joint_centers[parent_joint_idx])
    skel_as_mesh = get_skeleton_numpy(joint_centers, trainer.model.skeleton.joint_connections)
    bone_skeleton_association = get_bone_skeleton_association(bones_rst[:,:3].detach().cpu().numpy(), ends)
    bones_rst_mesh = get_bones_mesh(bones_rst.detach().cpu().numpy(),0.02)
    bone_skeleton_association = trimesh.util.concatenate([skel_as_mesh, bone_skeleton_association, bones_rst_mesh])
    bone_skeleton_association.export((opts.model_path[:-17]+'canonical_skel_bone_association.obj'))

    for i in range(0, len(idx_render), chunk):
        rendered_seq, aux_seq = trainer.eval_skel(idx_render=idx_render[i:i+chunk],
                                             dynamic_mesh=dynamic_mesh, skeleton=trainer.model.skeleton, bone_to_skeleton_pairs=bone_to_skeleton_pairs) 
        rendered_seq = tensor2array(rendered_seq)
        save_output(rendered_seq, aux_seq, seqname, trainer.model.skeleton, save_flo=opts.use_corresp, bone_to_skeleton_pairs=bone_to_skeleton_pairs)

if __name__ == '__main__':
    app.run(main)