import pdb
import cv2
import numpy as np
import trimesh
from pytorch3d import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
import pytorch3d
import pickle


import sys
sys.path.insert(0, 'third_party')
from ext_utils.flowlib import warp_flow, cat_imgflo 

def evaluate_mlp(model, xyz_embedded, embed_xyz=None, dir_embedded=None,
                chunk=32*1024, 
                xyz=None,
                code=None, sigma_only=False):
    """
    embed_xyz: embedding function
    chunk is the point-level chunk divided by number of bins
    """
    B,nbins,_ = xyz_embedded.shape
    out_chunks = []
    for i in range(0, B, chunk):
        embedded = xyz_embedded[i:i+chunk]
        if embed_xyz is not None:
            embedded = embed_xyz(embedded)
        if dir_embedded is not None:
            embedded = torch.cat([embedded,
                       dir_embedded[i:i+chunk]], -1)
        if code is not None:
            code_chunk = code[i:i+chunk]
            if code_chunk.dim() == 2: 
                code_chunk = code_chunk[:,None]
            code_chunk = code_chunk.repeat(1,nbins,1)
            embedded = torch.cat([embedded,code_chunk], -1)
        if xyz is not None:
            xyz_chunk = xyz[i:i+chunk]
        else: xyz_chunk = None
        out_chunks += [model(embedded, sigma_only=sigma_only, xyz=xyz_chunk)]

    out = torch.cat(out_chunks, 0)
    return out


def bone_transform(bones_in, rts, is_vec=False):
    """ 
    bones_in: 1,B,10  - B gaussian ellipsoids of bone coordinates
    rts: ...,B,3,4    - B ririd transforms
    rts are applied to bone coordinate transforms (left multiply)
    is_vec:     whether rts are stored as r1...9,t1...3 vector form
    """
    B = bones_in.shape[-2]
    bones = bones_in.view(-1,B,10).clone()
    if is_vec:
        rts = rts.view(-1,B,12)
    else:
        rts = rts.view(-1,B,3,4)
    bs = rts.shape[0] 

    center = bones[:,:,:3]
    orient = bones[:,:,3:7] # real first
    scale =  bones[:,:,7:10]
    if is_vec:
        Rmat = rts[:,:,:9].view(-1,B,3,3)
        Tmat = rts[:,:,9:12].view(-1,B,3,1)
    else:
        Rmat = rts[:,:,:3,:3]   
        Tmat = rts[:,:,:3,3:4]   

    # move bone coordinates (left multiply)
    center = Rmat.matmul(center[...,None])[...,0]+Tmat[...,0]
    Rquat = transforms.matrix_to_quaternion(Rmat)
    orient = transforms.quaternion_multiply(Rquat, orient)

    scale = scale.repeat(bs,1,1)
    bones = torch.cat([center,orient,scale],-1)
    return bones 

def joint_transform(joints_in, rts, is_vec=False):
    """ 
    joints_in: 1,B,10  - B joint coordinates
    rts: ...,B,3,4    - B ririd transforms
    rts are applied to bone coordinate transforms (left multiply)
    is_vec:     whether rts are stored as r1...9,t1...3 vector form
    """
    B = joints_in.shape[-2]
    joints_in = joints_in.view(-1,B,3).clone()
    if is_vec:
        rts = rts.view(-1,B,12)
    else:
        rts = rts.view(-1,B,3,4)

    center = joints_in[:,:,:3]

    if is_vec:
        Rmat = rts[:,:,:9].view(-1,B,3,3)
        Tmat = rts[:,:,9:12].view(-1,B,3,1)
    else:
        Rmat = rts[:,:,:3,:3]   
        Tmat = rts[:,:,:3,3:4]   

    # move joint coordinates (left multiply)
    center = Rmat.matmul(center[...,None])[...,0]+Tmat[...,0]
    center[1:,:] -= center[0,:]
    return center

def rtmat_invert(Rmat, Tmat):
    """
    Rmat: ...,3,3   - rotations
    Tmat: ...,3   - translations
    """
    rts = torch.cat([Rmat, Tmat[...,None]],-1)
    rts_i = rts_invert(rts)
    Rmat_i = rts_i[...,:3,:3] # bs, B, 3,3
    Tmat_i = rts_i[...,:3,3]
    return Rmat_i, Tmat_i

def rtk_invert(rtk_in, B):
    """
    rtk_in: ... (rot 1...9, trans 1...3)
    """
    rtk_shape = rtk_in.shape
    rtk_in = rtk_in.view(-1,B,12)# B,12
    rmat=rtk_in[:,:,:9]
    rmat=rmat.view(-1,B,3,3)
    tmat= rtk_in[:,:,9:12]
    rts_fw = torch.cat([rmat,tmat[...,None]],-1)
    rts_fw = rts_fw.view(-1,B,3,4)
    rts_bw = rts_invert(rts_fw)

    rvec = rts_bw[...,:3,:3].reshape(-1,9)
    tvec = rts_bw[...,:3,3] .reshape(-1,3)
    rtk = torch.cat([rvec,tvec],-1).view(rtk_shape)
    return rtk

def rts_invert(rts_in):
    """
    rts: ...,3,4   - B ririd transforms
    """
    rts = rts_in.view(-1,3,4).clone()
    Rmat = rts[:,:3,:3] # bs, B, 3,3
    Tmat = rts[:,:3,3:]
    Rmat_i=Rmat.permute(0,2,1)
    Tmat_i=-Rmat_i.matmul(Tmat)
    rts_i = torch.cat([Rmat_i, Tmat_i],-1)
    rts_i = rts_i.view(rts_in.shape)
    return rts_i

def rtk_to_4x4(rtk):
    """
    rtk: ...,12
    """
    device = rtk.device
    bs = rtk.shape[0]
    zero_one = torch.Tensor([[0,0,0,1]]).to(device).repeat(bs,1)

    rmat=rtk[:,:9]
    rmat=rmat.view(-1,3,3)
    tmat=rtk[:,9:12]
    rts = torch.cat([rmat,tmat[...,None]],-1)
    rts = torch.cat([rts,zero_one[:,None]],1)
    return rts

def rtk_compose(rtk1, rtk2):
    """
    rtk ...
    """
    rtk_shape = rtk1.shape
    rtk1 = rtk1.view(-1,12)# ...,12
    rtk2 = rtk2.view(-1,12)# ...,12

    rts1 = rtk_to_4x4(rtk1)
    rts2 = rtk_to_4x4(rtk2)

    rts = rts1.matmul(rts2)
    rvec = rts[...,:3,:3].reshape(-1,9)
    tvec = rts[...,:3,3].reshape(-1,3)
    rtk = torch.cat([rvec,tvec],-1).view(rtk_shape)
    return rtk

def vec_to_sim3(vec):
    """
    vec:      ...,10
    center:   ...,3
    orient:   ...,3,3
    scale:    ...,3
    """
    center = vec[...,:3]
    orient = vec[...,3:7] # real first
    orient = F.normalize(orient, 2,-1)
    orient = transforms.quaternion_to_matrix(orient) # real first
    scale =  vec[...,7:10].exp()
    return center, orient, scale


def gauss_mlp_skinning(xyz, embedding_xyz, bones, 
                    pose_code,  nerf_skin, use_hs=False, skin_aux=None, joints_only=False):
    """
    xyz:        N_rays, ndepth, 3
    bones:      ... nbones, 10
    pose_code:  ...,1, nchannel
    """
    N_rays = xyz.shape[0]
    #TODO hacky way to make code compaitible with noqueryfw
    if pose_code.dim() == 2 and pose_code.shape[0]!=N_rays: 
        pose_code = pose_code[None].repeat(N_rays, 1,1)

    xyz_embedded = embedding_xyz(xyz)
    dskin = mlp_skinning(nerf_skin, pose_code, xyz_embedded)
    skin = skinning(bones, xyz, use_hs=use_hs, dskin=dskin, skin_aux=skin_aux, joints_only=joints_only) # bs, N, B
    return skin

def skeleton_skinning_v1(xyz, embedding_xyz, joints, 
                    pose_code,  nerf_skin, skin_aux=None, top_k=None):
    """
    xyz:        N_rays, ndepth, 3
    joints:      ... njoints, 3
    pose_code:  ...,1, nchannel
    """
    N_rays = xyz.shape[0]
    #TODO hacky way to make code compaitible with noqueryfw
    if pose_code is not None and pose_code.dim() == 2 and pose_code.shape[0]!=N_rays: 
        pose_code = pose_code[None].repeat(N_rays, 1,1)

    if nerf_skin is not None:
        xyz_embedded = embedding_xyz(xyz)
        dskin = mlp_skinning(nerf_skin, pose_code, xyz_embedded)
    else:
        dskin = None

    skin = joints_skinning_v1(joints, xyz, dskin=dskin, skin_aux=skin_aux, top_k=top_k) # bs, N, B
    return skin

def skeleton_skinning_v2(xyz, joints, skeleton):
    """
    xyz:        N_rays, ndepth, 3
    joints:      ... njoints, 3
    pose_code:  ...,1, nchannel
    """
    skin = joints_skinning_v2(joints, xyz, skeleton) # bs, N, B
    return skin

def mlp_skinning(mlp, code, pts_embed):
    """
    code: bs, D          - N D-dimensional pose code
    pts_embed: bs,N,x    - N point positional embeddings
    dskin: bs,N,B        - delta skinning matrix
    """
    if mlp is None:
        dskin = None
    else:
        dskin = evaluate_mlp(mlp, pts_embed, code=code, chunk=8*1024)
    return dskin

def joint_connection_greedy_estimation(joint_centers, y_priority=0):
    """iteratively optimize structure and joint location
    no distinction between self joint and child joint

    Args:
        joint_centers (torch.Tensor):   # (BS, num_parts, 3, 1)

    Returns:
        joint_connection: (num_parts - 1, 2)
        connected_joints_distance_squared: (num_parts - 1)
    """
    with torch.no_grad():
        batchsize = joint_centers.shape[0]
        num_child = joint_centers.shape[3]

        num_parts = joint_centers.shape[1]
        joints_3d = joint_centers.permute(0, 1, 3, 2).reshape(batchsize, -1, 3)
        device = joint_centers.device

        # l2 distance between self and self's parent's children
        relative_cand_position = joints_3d[:, :, None] - joints_3d[:, None]  # (BS, n_all_child, n_all_child, 3)
        cand_distance = torch.sum(relative_cand_position ** 2, dim=-1).mean(dim=0)  # (n_all_child, n_all_child)

        # choose best child pairs
        best_distance_euclidean, best_idx = F.max_pool2d(-cand_distance[None, None], num_child,
                                               return_indices=True)
        best_distance_euclidean = -best_distance_euclidean[0, 0]  # (n_parts, n_parts)

        # # greedy estimation
        connectivity = torch.eye(num_parts, device=device, dtype=torch.long)  # (n_parts, n_parts)

        joint_connection = torch.zeros(num_parts - 1, 2, device=device, dtype=torch.long)
        connected_joints_distance_squared = torch.zeros((num_parts-1,1), device=device)
        connected_component = 0
        if y_priority > 0:
            joints_xz = joint_centers[:,:,(0,2),:].permute(0, 1, 3, 2).reshape(batchsize, -1, 2)
            relative_cand_position_xy = joints_xz[:, :, None] - joints_xz[:, None]  # (BS, n_all_child, n_all_child, 3)
            cand_distance_xz = torch.sum(relative_cand_position_xy ** 2, dim=-1).mean(dim=0)  # (n_all_child, n_all_child)

            # # structure step
            # # choose best child pairs
            best_distance_xz, best_idx = F.max_pool2d(-cand_distance_xz[None, None], num_child,
                                                return_indices=True)
            best_distance_xz = -best_distance_xz[0, 0]  # (n_parts, n_parts)
            best_distance = best_distance_euclidean + best_distance_xz * y_priority
        
        else:
            best_distance = best_distance_euclidean

        while True:  # there are n_parts-1 connection
            # find minimum distance
            invalid_connection_bias = connectivity * 1e10
            connected = torch.argmin(best_distance + invalid_connection_bias)
            connected_idx_0 = torch.div(connected, num_parts, rounding_mode='trunc')
            connected_idx_1 = connected % num_parts

            # update connectivity
            connectivity[connected_idx_0] = torch.maximum(connectivity[connected_idx_0].clone(),
                                                          connectivity[connected_idx_1].clone())
            connectivity[torch.where(connectivity[connected_idx_0] == 1)] = connectivity[connected_idx_0].clone()

            joint_connection[connected_component, 0] = connected_idx_0
            joint_connection[connected_component, 1] = connected_idx_1
            connected_joints_distance_squared[connected_component,0] = best_distance_euclidean[connected_idx_0, connected_idx_1]
            if connected_component == num_parts - 2:
                break
            connected_component += 1
    return joint_connection, connected_joints_distance_squared

def compute_joints_distance(joints, joint_connections):
    joints_list_1 = joints[joint_connections[:,0]]
    joints_list_2 = joints[joint_connections[:,1]]
    computed_dist_squared = torch.sum((joints_list_1 - joints_list_2)** 2, dim=-1, keepdim=True)
    return computed_dist_squared

def get_joints_dist_matrix(joints):
    """
    compute distance matrix between joints
    """
    joints_xyz = joints[:,:3]
    distance = torch.sum((joints_xyz[:, None] - joints_xyz[None, :]) ** 2, dim=-1)
    return distance

def axis_rotate(orient, mdis):
    bs,N,B,_,_ = mdis.shape
    mdis = (orient * mdis.view(bs,N,B,1,3)).sum(4)[...,None] # faster 
    #mdis = orient.matmul(mdis) # bs,N,B,3,1 # slower
    return mdis

def masked_softmax(vec, mask, dim=-1, mode='softmax', soft_blend=1):
    if mode == 'softmax':

        vec = torch.distributions.Bernoulli(logits=vec).probs

        masked_exps = torch.exp(soft_blend*vec) * mask.float()
        masked_exps_sum = masked_exps.sum(dim)

        output = torch.zeros_like(vec)
        output[masked_exps_sum>0,:] = masked_exps[masked_exps_sum>0,:]/ masked_exps_sum[masked_exps_sum>0].unsqueeze(-1)

        output = (output * vec).sum(dim, keepdim=True)

        output = torch.distributions.Bernoulli(probs=output).logits

    elif mode == 'max':
        vec[~mask] = -math.inf
        output = torch.max(vec, dim, keepdim=True)[0]

    return output

def get_root_id(joints):
    mean = torch.mean(joints[:,:3], axis = 0)
    # mean = torch.Tensor(list(mean)*(bones.shape[0]))
    cost = joints[:,:3] - mean
    center_distance= torch.sum(cost** 2, dim = 1)
    idx_min = torch.argmin(center_distance)
    return int(idx_min)

def get_bones_2d(bones, rtk, kaug_vec):
    device = bones.device

    # print('bones',bones.shape)
    center, _, _ = vec_to_sim3(bones)

    Rmat = rtk[:3,:3].to(device)
    Tmat = rtk[:3,3].to(device)

    vertices = obj_to_cam(center, Rmat.float(), Tmat.float())
    Kmat = K2mat(rtk[3,:]).to(device)
    # vertices_2d = Kmat.matmul(vertices.permute(1,0))
    # vertices_2d_n = torch.divide(vertices_2d[0,:,:], vertices_2d[0,2,:])

    Kaug = K2inv(kaug_vec) # p = Kaug Kmat P
    Kaug1 = Kaug.matmul(Kmat)
    vertices_2d_kaug = Kaug1.matmul(vertices.permute(1,0))
    vertices_2d_kaug_n = torch.divide(vertices_2d_kaug[0,:2,:], vertices_2d_kaug[0,2,:])
    return vertices_2d_kaug_n.T

def get_projections(bones, rtk, kaug_vec, len_max):
    device = bones.device
    len_max = len_max
    elips = trimesh.creation.uv_sphere(radius=len_max/20,count=[16, 16])
    # remove identical vertices
    # elips = trimesh.Trimesh(vertices=elips.vertices, faces=elips.faces)

    ellip_verts = []
    # print('bones',bones.shape)
    for bone in bones:
        center = bone[:3]
        # print(center)
        orient = bone[3:7] # real first
        orient = F.normalize(orient, 2,-1)

        orient = transforms.quaternion_to_matrix(orient) # real first
        # orient = torch.from_numpy(orient).float()
        # print('orient',orient.shape)
        orient = orient.permute(1,0) # transpose R
        scale = bone[7:10].exp()

        elips_verts = elips.vertices
        elips_verts = torch.Tensor(elips_verts).to(device)
        elips_verts = elips_verts / scale
        elips_verts = elips_verts.matmul(orient)
        elips_verts = elips_verts + center
        mesh_rest = pytorch3d.structures.meshes.Meshes(
                verts=elips_verts[None],
                faces=torch.Tensor(elips.faces[None]).to(device))
        shape_samp = pytorch3d.ops.sample_points_from_meshes(mesh_rest,
                                1000, return_normals=False)
        shape_samp = shape_samp[0].to(device)
        ellip_verts.append(shape_samp)

    ellip_verts = torch.cat(ellip_verts).to(device)

    Rmat = rtk[:3,:3].to(device)
    Tmat = rtk[:3,3].to(device)

    vertices = obj_to_cam(ellip_verts, Rmat.float(), Tmat.float())
    Kmat = K2mat(rtk[3,:]).to(device)
    # vertices_2d = Kmat.matmul(vertices.permute(1,0))
    # vertices_2d_n = torch.divide(vertices_2d[0,:,:], vertices_2d[0,2,:])

    Kaug = K2inv(kaug_vec) # p = Kaug Kmat P
    Kaug1 = Kaug.matmul(Kmat)
    vertices_2d_kaug = Kaug1.matmul(vertices.permute(1,0))
    vertices_2d_kaug_n = torch.divide(vertices_2d_kaug[0,:,:], vertices_2d_kaug[0,2,:])
    return vertices_2d_kaug_n.T

''' Hierarchical softmax following the kinematic tree of the human body. Imporves convergence speed'''
def hierarchical_softmax(x):
    def softmax(x):
        return torch.nn.functional.softmax(x, dim=-2)

    def sigmoid(x):
        return torch.sigmoid(x)

    n_batch, n_point, n_dim = x.shape
    x = x.flatten(0,1)

    prob_all = torch.ones(n_batch * n_point, n_dim, device=x.device)

    prob_all[:, [1, 2, 3]] = prob_all[:, [0]] * sigmoid(x[:, [0]]) * softmax(x[:, [1, 2, 3]])
    prob_all[:, [0]] = prob_all[:, [0]] * (1 - sigmoid(x[:, [0]]))

    prob_all[:, [4, 5, 6]] = prob_all[:, [1, 2, 3]] * (sigmoid(x[:, [4, 5, 6]]))
    prob_all[:, [1, 2, 3]] = prob_all[:, [1, 2, 3]] * (1 - sigmoid(x[:, [4, 5, 6]]))

    prob_all[:, [7, 8, 9]] = prob_all[:, [4, 5, 6]] * (sigmoid(x[:, [7, 8, 9]]))
    prob_all[:, [4, 5, 6]] = prob_all[:, [4, 5, 6]] * (1 - sigmoid(x[:, [7, 8, 9]]))

    prob_all[:, [10, 11]] = prob_all[:, [7, 8]] * (sigmoid(x[:, [10, 11]]))
    prob_all[:, [7, 8]] = prob_all[:, [7, 8]] * (1 - sigmoid(x[:, [10, 11]]))

    prob_all[:, [12, 13, 14]] = prob_all[:, [9]] * sigmoid(x[:, [24]]) * softmax(x[:, [12, 13, 14]])
    prob_all[:, [9]] = prob_all[:, [9]] * (1 - sigmoid(x[:, [24]]))

    prob_all[:, [15]] = prob_all[:, [12]] * (sigmoid(x[:, [15]]))
    prob_all[:, [12]] = prob_all[:, [12]] * (1 - sigmoid(x[:, [15]]))

    prob_all[:, [16, 17]] = prob_all[:, [13, 14]] * (sigmoid(x[:, [16, 17]]))
    prob_all[:, [13, 14]] = prob_all[:, [13, 14]] * (1 - sigmoid(x[:, [16, 17]]))

    prob_all[:, [18, 19]] = prob_all[:, [16, 17]] * (sigmoid(x[:, [18, 19]]))
    prob_all[:, [16, 17]] = prob_all[:, [16, 17]] * (1 - sigmoid(x[:, [18, 19]]))

    prob_all[:, [20, 21]] = prob_all[:, [18, 19]] * (sigmoid(x[:, [20, 21]]))
    prob_all[:, [18, 19]] = prob_all[:, [18, 19]] * (1 - sigmoid(x[:, [20, 21]]))

    prob_all[:, [22, 23]] = prob_all[:, [20, 21]] * (sigmoid(x[:, [22, 23]]))
    prob_all[:, [20, 21]] = prob_all[:, [20, 21]] * (1 - sigmoid(x[:, [22, 23]]))

    prob_all = prob_all.reshape(n_batch, n_point, n_dim)
    return prob_all*0.1


def joints_skinning_chunk_v1(joints, pts, dskin=None, skin_aux=None, top_k=None):
    """
    joint: bs,B,10  - B joints
    pts: bs,N,3    - N 3d points, usually N=num points per ray, b~=2034
    skin: bs,N,B   - skinning matrix
    """
    log_scale= skin_aux[0]
    bs,N,_ = pts.shape
    B = joints.shape[-2]
    if joints.dim()==2: joints = joints[None].repeat(bs,1,1)

    # mahalanobis distance
    # transform a vector to the local coordinate
    mdis = joints.view(bs,1,B,3) - pts.view(bs,N,1,3) # bs,N,B,3
    mdis = mdis.pow(2)

    # log_scale (being optimized) controls temporature of the skinning weight softmax 
    # multiply 1000 to make the weights more concentrated initially
    inv_temperature = 2000 * log_scale.exp()
    # print('inv temperature',inv_temperature, mdis.shape,torch.min(mdis),torch.max(mdis))
    mdis = (-inv_temperature * mdis.sum(3)) # bs,N,B

    if top_k is not None:
        mdis_vals, mdis_idx = torch.topk(mdis, top_k, dim=2)
        mdis_sparse = (-torch.ones_like(mdis)*1e20).scatter_(2, mdis_idx, mdis_vals)

        if dskin is not None:
            dskin_vals, dskin_idx = torch.topk(dskin, top_k, dim=2)
            dskin_sparse = (-torch.ones_like(dskin)*1e20).scatter_(2, dskin_idx, dskin_vals)
            mdis = mdis_sparse+dskin_sparse
        else:
            mdis = mdis_sparse

    else:
        if dskin is not None:
            mdis = mdis+dskin

    skin = mdis.softmax(2)
    return skin

def joints_skinning_chunk_v2(joints, pts, skeleton):
    """
    joint: bs,B,10  - B joints
    pts: bs,N,3    - N 3d points, usually N=num points per ray, b~=2034
    skin: bs,N,B   - skinning matrix
    """
    bs,N,_ = pts.shape
    B = joints.shape[-2]
    if joints.dim()==2: joints = joints[None].repeat(bs,1,1)

    # geodesic distance
    # transform a vector to the local coordinate
    mdis = joints.view(bs,1,B,3) - pts.view(bs,N,1,3) # bs,N,B,3
    mdis = mdis.pow(2).sum(3).sqrt()
    mdis_vals, mdis_idx = torch.topk(-mdis, 1, dim=2)

    traversal_distance = torch.zeros_like(mdis).to(mdis.device)
    for i in range(mdis.shape[0]):
        # print(mdis_idx[i,:,:],mdis_vals[i,:,:])
        traversal_distance[i,:,:] = skeleton.traversal_distance_matrix[mdis_idx[i,:,:].item()].view(1,mdis.shape[2])
        # traversal_distance[i,:,:] += (-1*mdis_vals[i,:,:])
    
    # print(traversal_distance.shape)
    mdis += traversal_distance.pow(2)

    # hardcoded scale
    scale = -2000
    mdis = scale * mdis
    # print(traversal_distance)
    # mdis_closest = (-torch.ones_like(mdis)*1e20).scatter_(2, mdis_idx, mdis_vals)

    # log_scale (being optimized) controls temporature of the skinning weight softmax 
    # multiply 1000 to make the weights more concentrated initially
    # inv_temperature = 2000 * log_scale.exp()
    # # print('inv temperature',inv_temperature, mdis.shape,torch.min(mdis),torch.max(mdis))
    # mdis = (-inv_temperature * mdis.sum(3)) # bs,N,B

    # normalize by the sum instead of softmax??
    skin = mdis.softmax(2)
    return skin

def skinning_chunk(bones, pts, use_hs=False, dskin=None, skin_aux=None, joints_only=False):
#def skinning(bones, pts, dskin=None, skin_aux=None):
    """
    bone: bs,B,10  - B gaussian ellipsoids
    pts: bs,N,3    - N 3d points, usually N=num points per ray, b~=2034
    skin: bs,N,B   - skinning matrix
    """
    device = pts.device
    log_scale= skin_aux[0]
    w_const  = skin_aux[1]
    bs,N,_ = pts.shape
    B = bones.shape[-2]
    if bones.dim()==2: bones = bones[None].repeat(bs,1,1)
    bones = bones.view(-1,B,10)
   
    center, orient, scale = vec_to_sim3(bones)

    # mahalanobis distance [(p-v)^TR^T]S[R(p-v)]
    # transform a vector to the local coordinate
    mdis = center.view(bs,1,B,3) - pts.view(bs,N,1,3) # bs,N,B,3

    if not joints_only:
        orient = orient.permute(0,1,3,2) # transpose R
        mdis = axis_rotate(orient.view(bs,1,B,3,3), mdis[...,None])
        mdis = mdis[...,0]
        mdis = scale.view(bs,1,B,3) * mdis.pow(2)
    else:
        mdis = mdis.pow(2)

    # log_scale (being optimized) controls temporature of the skinning weight softmax 
    # multiply 1000 to make the weights more concentrated initially
    inv_temperature = 1000 * log_scale.exp()
    mdis = (-inv_temperature * mdis.sum(3)) # bs,N,B

    if dskin is not None:
        mdis = mdis+dskin

    if use_hs:
        skin = hierarchical_softmax(mdis)
    else:
        skin = mdis.softmax(2)
    return skin
    

def joints_skinning_v1(joints, pts, dskin=None, skin_aux=None, top_k=None):
    """
    joint: ...,B,3  - B joints
    pts: bs,N,3    - N 3d points
    skin: bs,N,B   - skinning matrix
    """
    chunk=4096
    bs,N,_ = pts.shape
    B = joints.shape[-2]
    if joints.dim()==2: joints = joints[None].repeat(bs,1,1)
    joints = joints.view(-1,B,3)

    skin = []
    for i in range(0,bs,chunk):
        if dskin is None:
            dskin_chunk = None
        else: 
            dskin_chunk = dskin[i:i+chunk]
        skin_chunk = joints_skinning_chunk_v1(joints[i:i+chunk], pts[i:i+chunk],\
                              dskin=dskin_chunk, skin_aux=skin_aux, top_k=top_k)
        skin.append(skin_chunk)
    skin = torch.cat(skin,0)
    return skin

def joints_skinning_v2(joints, pts, skeleton):
    """
    joint: ...,B,3  - B joints
    pts: bs,N,3    - N 3d points
    skin: bs,N,B   - skinning matrix
    """
    chunk=4096
    bs,N,_ = pts.shape
    B = joints.shape[-2]
    if joints.dim()==2: joints = joints[None].repeat(bs,1,1)
    joints = joints.view(-1,B,3)

    skin = []
    for i in range(0,bs,chunk):
        skin_chunk = joints_skinning_chunk_v2(joints[i:i+chunk], pts[i:i+chunk], skeleton)
        skin.append(skin_chunk)
    skin = torch.cat(skin,0)
    return skin

def skinning(bones, pts, use_hs=False, dskin=None, skin_aux=None, joints_only=False):
    """
    bone: ...,B,10  - B gaussian ellipsoids
    pts: bs,N,3    - N 3d points
    skin: bs,N,B   - skinning matrix
    """
    chunk=4096
    bs,N,_ = pts.shape
    B = bones.shape[-2]
    if bones.dim()==2: bones = bones[None].repeat(bs,1,1)
    bones = bones.view(-1,B,10)

    skin = []
    for i in range(0,bs,chunk):
        if dskin is None:
            dskin_chunk = None
        else: 
            dskin_chunk = dskin[i:i+chunk]
        skin_chunk = skinning_chunk(bones[i:i+chunk], pts[i:i+chunk], use_hs=use_hs,\
                              dskin=dskin_chunk, skin_aux=skin_aux, joints_only=joints_only)
        skin.append(skin_chunk)
    skin = torch.cat(skin,0)
    return skin

def blend_skinning_chunk(bones, rts, skin, pts):
#def blend_skinning(bones, rts, skin, pts):
    """
    bone: bs,B,10   - B gaussian ellipsoids
    rts: bs,B,3,4   - B ririd transforms, applied to bone coordinates (points attached to bones in world coords)
    pts: bs,N,3     - N 3d points
    skin: bs,N,B   - skinning matrix
    apply rts to bone coordinates, while computing blending globally
    """
    B = rts.shape[-3]
    N = pts.shape[-2]
    pts = pts.view(-1,N,3)
    rts = rts.view(-1,B,3,4)
    Rmat = rts[:,:,:3,:3] # bs, B, 3,3
    Tmat = rts[:,:,:3,3]
    device = Tmat.device

    ## convert from bone to root transforms
    #bones = bones.view(-1,B,10)
    #bs = Rmat.shape[0]
    #center = bones[:,:,:3]
    #orient = bones[:,:,3:7] # real first
    #orient = F.normalize(orient, 2,-1)
    #orient = transforms.quaternion_to_matrix(orient) # real first
    #gmat = torch.eye(4)[None,None].repeat(bs, B, 1, 1).to(device)
    #
    ## root to bone
    #gmat_r2b = gmat.clone()
    #gmat_r2b[:,:,:3,:3] = orient.permute(0,1,3,2)
    #gmat_r2b[:,:,:3,3] = -orient.permute(0,1,3,2).matmul(center[...,None])[...,0]
   
    ## bone to root
    #gmat_b2r = gmat.clone()
    #gmat_b2r[:,:,:3,:3] = orient
    #gmat_b2r[:,:,:3,3] = center

    ## bone to bone  
    #gmat_b = gmat.clone()
    #gmat_b[:,:,:3,:3] = Rmat
    #gmat_b[:,:,:3,3] = Tmat
   
    #gmat = gmat_b2r.matmul(gmat_b.matmul(gmat_r2b))
    #Rmat = gmat[:,:,:3,:3]
    #Tmat = gmat[:,:,:3,3]

    # Gi=sum(wbGb), V=RV+T
    Rmat_w = (skin[...,None,None] * Rmat[:,None]).sum(2) # bs,N,B,3
    Tmat_w = (skin[...,None] * Tmat[:,None]).sum(2) # bs,N,B,3
    pts = Rmat_w.matmul(pts[...,None]) + Tmat_w[...,None] 
    pts = pts[...,0]
    return pts

def blend_skinning(bones, rts, skin, pts):
    """
    bone: bs,B,10   - B gaussian ellipsoids
    rts: bs,B,3,4   - B ririd transforms, applied to bone coordinates
    pts: bs,N,3     - N 3d points
    skin: bs,N,B   - skinning matrix
    apply rts to bone coordinates, while computing blending globally
    """
    chunk=4096
    B = rts.shape[-3]
    N = pts.shape[-2]
    bones = bones.view(-1,B,10)
    pts = pts.view(-1,N,3)
    rts = rts.view(-1,B,3,4)
    bs = pts.shape[0]

    pts_out = []
    for i in range(0,bs,chunk):
        pts_chunk = blend_skinning_chunk(bones[i:i+chunk], rts[i:i+chunk], 
                                          skin[i:i+chunk], pts[i:i+chunk])
        pts_out.append(pts_chunk)
    pts = torch.cat(pts_out,0)
    return pts

def lbs(bones, rts_fw, skin, xyz_in, backward=True):
    """
    bones: bs,B,10       - B gaussian ellipsoids indicating rest bone coordinates
    rts_fw: bs,B,12       - B rigid transforms, applied to the rest bones
    xyz_in: bs,N,3       - N 3d points after transforms in the root coordinates
    """
    B = bones.shape[-2]
    N = xyz_in.shape[-2]
    bs = rts_fw.shape[0]
    bones = bones.view(-1,B,10)
    xyz_in = xyz_in.view(-1,N,3)
    rts_fw = rts_fw.view(-1,B,12)# B,12
    rmat=rts_fw[:,:,:9]
    rmat=rmat.view(bs,B,3,3)
    tmat= rts_fw[:,:,9:12]
    rts_fw = torch.cat([rmat,tmat[...,None]],-1)
    rts_fw = rts_fw.view(-1,B,3,4)

    if backward:
        bones_dfm = bone_transform(bones, rts_fw) # bone coordinates after deform
        rts_bw = rts_invert(rts_fw)
        xyz = blend_skinning(bones_dfm, rts_bw, skin, xyz_in)
    else:
        xyz = blend_skinning(bones.repeat(bs,1,1), rts_fw, skin, xyz_in)
        bones_dfm = bone_transform(bones, rts_fw) # bone coordinates after deform
    return xyz, bones_dfm

def joints_blend_skinning_chunk(joints, rts, skin, pts):
    """
    joint: bs,B,10   - B joints
    rts: bs,B,3,4   - B ririd transforms, applied to bone coordinates (points attached to joints in world coords)
    pts: bs,N,3     - N 3d points
    skin: bs,N,B   - skinning matrix
    apply rts to bone coordinates, while computing blending globally
    """
    B = rts.shape[-3]
    N = pts.shape[-2]
    pts = pts.view(-1,N,3)
    rts = rts.view(-1,B,3,4)
    Rmat = rts[:,:,:3,:3] # bs, B, 3,3
    Tmat = rts[:,:,:3,3]

    # Gi=sum(wbGb), V=RV+T
    Rmat_w = (skin[...,None,None] * Rmat[:,None]).sum(2) # bs,N,B,3
    Tmat_w = (skin[...,None] * Tmat[:,None]).sum(2) # bs,N,B,3
    pts = Rmat_w.matmul(pts[...,None]) + Tmat_w[...,None] 
    pts = pts[...,0]
    return pts

def joints_blend_skinning(joints, rts, skin, pts):
    """
    joint: bs,B,3   - B joints
    rts: bs,B,3,4   - B ririd transforms, applied to bone coordinates
    pts: bs,N,3     - N 3d points
    skin: bs,N,B   - skinning matrix
    apply rts to bone coordinates, while computing blending globally
    """
    chunk=4096
    B = rts.shape[-3]
    N = pts.shape[-2]
    joints = joints.view(-1,B,3)
    pts = pts.view(-1,N,3)
    rts = rts.view(-1,B,3,4)
    bs = pts.shape[0]

    pts_out = []
    for i in range(0,bs,chunk):
        pts_chunk = joints_blend_skinning_chunk(joints[i:i+chunk], rts[i:i+chunk], 
                                          skin[i:i+chunk], pts[i:i+chunk])
        pts_out.append(pts_chunk)
    pts = torch.cat(pts_out,0)
    return pts

def lbs_skeleton(joints, rts_fw, skin, xyz_in, backward=True):
    """
    joints: bs,B,3       - B canonical joints coordinates
    rts_fw: bs,B,12       - B rigid transforms, applied to the canonical joints
    xyz_in: bs,N,3       - N 3d points after transforms in the root coordinates
    """
    B = joints.shape[-2]
    N = xyz_in.shape[-2]
    bs = rts_fw.shape[0]
    joints = joints.view(-1,B,3)
    xyz_in = xyz_in.view(-1,N,3)
    rts_fw = rts_fw.view(-1,B,12)# B,12
    rmat=rts_fw[:,:,:9]
    rmat=rmat.view(bs,B,3,3)
    tmat= rts_fw[:,:,9:12]
    rts_fw = torch.cat([rmat,tmat[...,None]],-1)
    rts_fw = rts_fw.view(-1,B,3,4)

    if backward:
        joints_dfm = joint_transform(joints, rts_fw) # joint coordinates after deform
        rts_bw = rts_invert(rts_fw)
        xyz = joints_blend_skinning(joints_dfm, rts_bw, skin, xyz_in)
    else:
        xyz = joints_blend_skinning(joints.repeat(bs,1,1), rts_fw, skin, xyz_in)
        joints_dfm = joint_transform(joints, rts_fw) # joint coordinates after deform
    return xyz, joints_dfm

def obj_to_cam(in_verts, Rmat, Tmat):
    """
    verts: ...,N,3
    Rmat:  ...,3,3
    Tmat:  ...,3 
    """
    verts = in_verts.clone()
    if verts.dim()==2: verts=verts[None]
    verts = verts.view(-1,verts.shape[1],3)
    Rmat = Rmat.view(-1,3,3).permute(0,2,1) # left multiply
    Tmat = Tmat.view(-1,1,3)
    
    verts =  verts.matmul(Rmat) + Tmat 
    verts = verts.reshape(in_verts.shape)
    return verts

def obj2cam_np(pts, Rmat, Tmat):
    """
    a wrapper for numpy array
    pts: ..., 3
    Rmat: 1,3,3
    Tmat: 1,3,3
    """
    pts_shape = pts.shape
    pts = torch.Tensor(pts).cuda().reshape(1,-1,3)
    pts = obj_to_cam(pts, Rmat,Tmat)
    return pts.view(pts_shape).cpu().numpy()

    
def K2mat(K):
    """
    K: ...,4
    """
    K = K.view(-1,4)
    device = K.device
    bs = K.shape[0]

    Kmat = torch.zeros(bs, 3, 3, device=device)
    Kmat[:,0,0] = K[:,0]
    Kmat[:,1,1] = K[:,1]
    Kmat[:,0,2] = K[:,2]
    Kmat[:,1,2] = K[:,3]
    Kmat[:,2,2] = 1
    return Kmat

def mat2K(Kmat):
    """
    Kmat: ...,3,3
    """
    shape=Kmat.shape[:-2]
    Kmat = Kmat.view(-1,3,3)
    device = Kmat.device
    bs = Kmat.shape[0]

    K = torch.zeros(bs, 4, device=device)
    K[:,0] = Kmat[:,0,0]
    K[:,1] = Kmat[:,1,1]
    K[:,2] = Kmat[:,0,2]
    K[:,3] = Kmat[:,1,2]
    K = K.view(shape+(4,))
    return K

def Kmatinv(Kmat):
    """
    Kmat: ...,3,3
    """
    K = mat2K(Kmat)
    Kmatinv = K2inv(K)
    Kmatinv = Kmatinv.view(Kmat.shape)
    return Kmatinv

def K2inv(K):
    """
    K: ...,4
    """
    K = K.view(-1,4)
    device = K.device
    bs = K.shape[0]

    Kmat = torch.zeros(bs, 3, 3, device=device)
    Kmat[:,0,0] = 1./K[:,0]
    Kmat[:,1,1] = 1./K[:,1]
    Kmat[:,0,2] = -K[:,2]/K[:,0]
    Kmat[:,1,2] = -K[:,3]/K[:,1]
    Kmat[:,2,2] = 1
    return Kmat

def pinhole_cam(in_verts, K):
    """
    in_verts: ...,N,3
    K:        ...,4
    verts:    ...,N,3 in (x,y,Z)
    """
    verts = in_verts.clone()
    verts = verts.view(-1,verts.shape[1],3)
    K = K.view(-1,4)

    Kmat = K2mat(K)
    Kmat = Kmat.permute(0,2,1)

    verts = verts.matmul(Kmat)
    verts_z = verts[:,:,2:3]
    verts_xy = verts[:,:,:2] / (1e-6+verts_z) # deal with neg z
    
    verts = torch.cat([verts_xy,verts_z],-1)
    verts = verts.reshape(in_verts.shape)
    return verts

def render_color(renderer, in_verts, faces, colors, texture_type='vertex'):
    """
    verts in ndc
    in_verts: ...,N,3/4
    faces: ...,N,3
    rendered: ...,4,...
    """
    import soft_renderer as sr
    verts = in_verts.clone()
    verts = verts.view(-1,verts.shape[-2],3)
    faces = faces.view(-1,faces.shape[-2],3)
    if texture_type=='vertex':  colors = colors.view(-1,colors.shape[-2],3)
    elif texture_type=='surface': colors = colors.view(-1,colors.shape[1],colors.shape[2],3)
    device=verts.device

    offset = torch.Tensor( renderer.transform.transformer._eye).to(device)[np.newaxis,np.newaxis]
    verts_pre = verts[:,:,:3]-offset
    verts_pre[:,:,1] = -1*verts_pre[:,:,1]  # pre-flip
    rendered = renderer.render_mesh(sr.Mesh(verts_pre,faces,textures=colors,texture_type=texture_type))
    return rendered

def render_flow(renderer, verts, faces, verts_n):
    """
    rasterization
    verts in ndc
    verts: ...,N,3/4
    verts_n: ...,N,3/4
    faces: ...,N,3
    """
    verts = verts.view(-1,verts.shape[1],3)
    verts_n = verts_n.view(-1,verts_n.shape[1],3)
    faces = faces.view(-1,faces.shape[1],3)
    device=verts.device

    rendered_ndc_n = render_color(renderer, verts, faces, verts_n)
    _,_,h,w = rendered_ndc_n.shape
    rendered_sil = rendered_ndc_n[:,-1]

    ndc = np.meshgrid(range(w), range(h))
    ndc = torch.Tensor(ndc).to(device)[None]
    ndc[:,0] = ndc[:,0]*2 / (w-1) - 1
    ndc[:,1] = ndc[:,1]*2 / (h-1) - 1

    flow = rendered_ndc_n[:,:2] - ndc
    flow = flow.permute(0,2,3,1) # x,h,w,2
    flow = torch.cat([flow, rendered_sil[...,None]],-1)

    flow[rendered_sil<1]=0.
    flow[...,-1]=0. # discard the last channel
    return flow

def force_type(varlist):
    for i in range(len(varlist)):
        varlist[i] = varlist[i].type(varlist[0].dtype)
    return varlist

def tensor2array(tdict):
    adict={}
    for k,v in tdict.items():
        adict[k] = v.detach().cpu().numpy()
    return adict

def array2tensor(adict, device='cpu'):
    tdict={}
    for k,v in adict.items():
        try: 
            tdict[k] = torch.Tensor(v)
            if device != 'cpu': tdict[k] = tdict[k].to(device)
        except: pass # trimesh object
    return tdict

def raycast(xys, Rmat, Tmat, Kinv, near_far):
    """
    assuming xys and Rmat have same num of bs
    xys: bs, N, 3
    Rmat:bs, ...,3,3 
    Tmat:bs, ...,3, camera to root coord transform 
    Kinv:bs, ...,3,3 
    near_far:bs,2
    """
    Rmat, Tmat, Kinv, xys = force_type([Rmat, Tmat, Kinv, xys])
    Rmat = Rmat.view(-1,3,3)
    Tmat = Tmat.view(-1,1,3)
    Kinv = Kinv.view(-1,3,3)
    bs,nsample,_ = xys.shape
    device = Rmat.device

    xy1s = torch.cat([xys, torch.ones_like(xys[:,:,:1])],2)
    xyz3d = xy1s.matmul(Kinv.permute(0,2,1))
    ray_directions = xyz3d.matmul(Rmat)  # transpose -> right multiply
    ray_origins = -Tmat.matmul(Rmat) # transpose -> right multiply

    if near_far is not None:
        znear= (torch.ones(bs,nsample,1).to(device) * near_far[:,0,None,None]) 
        zfar = (torch.ones(bs,nsample,1).to(device) * near_far[:,1,None,None]) 
    else:
        lbound, ubound=[-1.5,1.5]

        znear= Tmat[:,:,-1:].repeat(1,nsample,1)+lbound
        zfar = Tmat[:,:,-1:].repeat(1,nsample,1)+ubound
        znear[znear<1e-5]=1e-5

    ray_origins = ray_origins.repeat(1,nsample,1)

    rmat_vec = Rmat.reshape(-1,1,9)
    tmat_vec = Tmat.reshape(-1,1,3)
    kinv_vec = Kinv.reshape(-1,1,9)
    rtk_vec = torch.cat([rmat_vec, tmat_vec, kinv_vec],-1) # x,21
    rtk_vec = rtk_vec.repeat(1,nsample,1)

    rays={'rays_o': ray_origins, 
          'rays_d': ray_directions,
          'near': znear,
          'far': zfar,
          'rtk_vec': rtk_vec,
          'xys': xys,
          'nsample': nsample,
          'bs': bs,
          }
    return rays

def sample_xy(img_size, bs, nsample, device, return_all=False, lineid=None):
    """
    rand_inds:  bs, ns
    xys:        bs, ns, 2
    """
    xygrid = np.meshgrid(range(img_size), range(img_size))  # w,h->hxw
    xygrid = torch.Tensor(xygrid).to(device)  # (x,y)
    xygrid = xygrid.permute(1,2,0).reshape(1,-1,2)  # 1,..., 2
    
    if return_all:
        xygrid = xygrid.repeat(bs,1,1)                  # bs,..., 2
        nsample = xygrid.shape[1]
        rand_inds=torch.Tensor(range(nsample))
        rand_inds=rand_inds[None].repeat(bs,1)
        xys = xygrid
    else:
        if lineid is None:
            probs = torch.ones(img_size**2).to(device) # 512*512 vs 128*64
            rand_inds = torch.multinomial(probs, bs*nsample, replacement=False)
            rand_inds = rand_inds.view(bs,nsample)
            xys = torch.stack([xygrid[0][rand_inds[i]] for i in range(bs)],0) # bs,ns,2
        else:
            probs = torch.ones(img_size).to(device) # 512*512 vs 128*64
            rand_inds = torch.multinomial(probs, bs*nsample, replacement=True)
            rand_inds = rand_inds.view(bs,nsample)
            xys = torch.stack([xygrid[0][rand_inds[i]] for i in range(bs)],0) # bs,ns,2
            xys[...,1] = xys[...,1] + lineid[:,None]
   
    rand_inds = rand_inds.long()
    return rand_inds, xys

def chunk_rays(rays,start,delta):
    """
    rays: a dictionary
    """
    rays_chunk = {}
    for k,v in rays.items():
        if torch.is_tensor(v):
            v = v.view(-1, v.shape[-1])
            rays_chunk[k] = v[start:start+delta]
    return rays_chunk
        

def generate_bones(num_bones_x, num_bones, bound, device, use_bone_offset=False):
    """
    num_bones_x: bones along one direction
    bones: x**3,9
    """
    center =  torch.linspace(-bound, bound, num_bones_x).to(device)
    center =torch.meshgrid(center, center, center)
    center = torch.stack(center,0).permute(1,2,3,0).reshape(-1,3)
    center = center[:num_bones]
    
    orient =  torch.Tensor([[1,0,0,0]]).to(device)
    orient = orient.repeat(num_bones,1)
    scale = torch.zeros(num_bones,3).to(device)
    bones = torch.cat([center, orient, scale],-1)

    if use_bone_offset:
        dummy_offset = torch.Tensor(([245.431 ,292.436], 
                                    [300.229 ,272.734], 
                                    [262.991, 288.514],
                                    [272.813 ,364.842],
                                    [282.613, 433.516],
                                    [335.473 ,253.223],
                                    [380.526 ,339.457],
                                    [335.489 ,415.769],
                                    [372.707 ,319.839],
                                    [347.224 ,323.765],
                                    [321.744 ,441.322],
                                    [335.426, 554.986],
                                    [400.255 ,315.897],
                                    [382.52 , 435.462],
                                    [431.625 ,511.858],
                                    [239.489 ,288.502],
                                    [247.348 ,286.465],
                                    [  0.    ,  0.   ],
                                    [272.836, 257.107],
                                    [417.781, 555.006],
                                    [433.469 ,552.992],
                                    [445.241, 511.848],
                                    [290.533, 582.338],
                                    [292.385, 572.576],
                                    [341.431 ,568.571])).to(device)
        bones = torch.cat((dummy_offset*0.002*bound, torch.zeros(num_bones,8).to(device)), 1) + bones
    return bones

def center_skeleton(bone, skin_mesh, device):
    # Center the skeleton with respect to the skin
    # print(bone)
    bone_ = bone[:,:3]
    offset = torch.mean(bone_, axis=0) - torch.mean(skin_mesh, axis=0) 
    offset_stacked = torch.Tensor(list(offset)*(bone_.shape[0])).to(device)
    offset_stacked = offset_stacked.reshape(bone_.shape[0],3)
    bone_ = bone_ - offset_stacked
    bone[:,:3] = bone_
    return bone

def reinit_bones(model, mesh, num_bones, use_center_skeleton):
    """
    update the data of bones and nerf_body_rts[1].rgb without add new parameters
    num_bones: number of bones on the surface
    mesh: trimesh
    warning: ddp does not support adding/deleting parameters after construction
    """
    #TODO find another way to add/delete bones
    from kmeans_pytorch import kmeans
    device = model.device
    points = torch.Tensor(mesh.vertices).to(device)
    rthead = model.nerf_body_rts[1].rgb
    
    # reinit
    num_in = rthead[0].weight.shape[1]
    rthead = nn.Sequential(nn.Linear(num_in, 6*num_bones)).to(device)
    torch.nn.init.xavier_uniform_(rthead[0].weight, gain=0.5)
    torch.nn.init.zeros_(rthead[0].bias)

    if points.shape[0]<100:
        bound = model.latest_vars['obj_bound']
        bound = torch.Tensor(bound)[None]
        center = torch.rand(num_bones, 3) *  bound*2 - bound
    else:
        _, center = kmeans(X=points, num_clusters=num_bones, iter_limit=100,
                        tqdm_flag=False, distance='euclidean', device=device)
    center=center.to(device)
    orient =  torch.Tensor([[1,0,0,0]]).to(device)
    orient = orient.repeat(num_bones,1)
    scale = torch.zeros(num_bones,3).to(device)
    bones = torch.cat([center, orient, scale],-1)

    model.num_bones = num_bones
    num_output = model.nerf_body_rts[1].num_output
    bias_reinit =   rthead[0].bias.data
    weight_reinit=rthead[0].weight.data
    model.nerf_body_rts[1].rgb[0].bias.data[:num_bones*num_output] = bias_reinit
    model.nerf_body_rts[1].rgb[0].weight.data[:num_bones*num_output] = weight_reinit
    
    bones,_ = correct_bones(model, bones, inverse=True)
    if use_center_skeleton:
        #print("Bones:",bones.shape)
        #print(points.shape)
        bones = center_skeleton(bones, points, device)
    model.bones.data[:num_bones] = bones
    model.nerf_models['bones'] = model.bones
    return

def correct_bones(model, bones_rst, inverse=False):
    # bones=>bones_rst
    bones_rst = bones_rst.clone()
    rest_pose_code =  model.rest_pose_code
    rest_pose_code = rest_pose_code(torch.Tensor([0]).long().to(model.device))
    rts_head = model.nerf_body_rts[1]
    bone_rts_rst = rts_head(rest_pose_code)[0] # 1,B*12
    if inverse:
        bone_rts_rst = rtk_invert(bone_rts_rst, model.opts.num_bones)
    bones_rst = bone_transform(bones_rst, bone_rts_rst, is_vec=True)[0] 
    return bones_rst, bone_rts_rst

def correct_rest_pose(opts, bone_rts_fw, bone_rts_rst):
    # delta rts
    bone_rts_fw = bone_rts_fw.clone()
    rts_shape = bone_rts_fw.shape
    bone_rts_rst_inv = rtk_invert(bone_rts_rst, opts.num_bones)
    bone_rts_rst_inv = bone_rts_rst_inv.repeat(rts_shape[0],rts_shape[1],1)
    bone_rts_fw =     rtk_compose(bone_rts_rst_inv, bone_rts_fw)
    return bone_rts_fw

def warp_bw(opts, model, rt_dict, query_xyz_chunk, embedid):
    """
    only used in mesh extraction
    embedid: embedding id
    """
    chunk = query_xyz_chunk.shape[0]
    query_time = torch.ones(chunk,1).to(model.device)*embedid
    query_time = query_time.long()
    if opts.flowbw:
        # flowbw
        xyz_embedded = model.embedding_xyz(query_xyz_chunk)
        time_embedded = model.pose_code(query_time)[:,0]
        xyztime_embedded = torch.cat([xyz_embedded, time_embedded],1)

        flowbw_chunk = model.nerf_flowbw(xyztime_embedded, xyz=query_xyz_chunk)
        query_xyz_chunk += flowbw_chunk
    elif opts.lbs:
        # backward skinning
        bones_rst = model.bones
        bone_rts_fw = model.nerf_body_rts(query_time)
        # update bones
        bones_rst, bone_rts_rst = correct_bones(model, bones_rst)
        bone_rts_fw = correct_rest_pose(opts, bone_rts_fw, bone_rts_rst)

        query_xyz_chunk = query_xyz_chunk[:,None]

        if opts.nerf_skin:
            nerf_skin = model.nerf_skin
        else:
            nerf_skin = None
        time_embedded = model.pose_code(query_time)
        bones_dfm = bone_transform(bones_rst, bone_rts_fw, is_vec=True)

        skin_backward = gauss_mlp_skinning(query_xyz_chunk, model.embedding_xyz,
                   bones_dfm, time_embedded, nerf_skin, use_hs=model.use_hs, skin_aux=model.skin_aux, joints_only=opts.joints_only)

        query_xyz_chunk,bones_dfm = lbs(bones_rst, 
                                      bone_rts_fw,
                                      skin_backward,
                                      query_xyz_chunk)

        query_xyz_chunk = query_xyz_chunk[:,0]
        rt_dict['bones'] = bones_dfm 
    return query_xyz_chunk, rt_dict
        
def warp_fw(opts, model, rt_dict, vertices, embedid):
    """
    only used in mesh extraction
    """
    num_pts = vertices.shape[0]
    query_time = torch.ones(num_pts,1).long().to(model.device)*embedid
    pts_can=torch.Tensor(vertices).to(model.device)
    if opts.flowbw:
        # forward flow
        pts_can_embedded = model.embedding_xyz(pts_can)
        time_embedded = model.pose_code(query_time)[:,0]
        ptstime_embedded = torch.cat([pts_can_embedded, time_embedded],1)

        pts_dfm = pts_can + model.nerf_flowfw(ptstime_embedded, xyz=pts_can)
    elif opts.lbs:
        # forward skinning
        pts_can = pts_can[:,None]
        bones_rst = model.bones
        bone_rts_fw = model.nerf_body_rts(query_time)
        bones_rst, bone_rts_rst = correct_bones(model, bones_rst)
        bone_rts_fw = correct_rest_pose(opts, bone_rts_fw, bone_rts_rst)
        
        if opts.nerf_skin:
            nerf_skin = model.nerf_skin
        else:
            nerf_skin = None
        rest_pose_code =  model.rest_pose_code
        rest_pose_code = rest_pose_code(torch.Tensor([0]).long().to(bones_rst.device))
        skin_forward = gauss_mlp_skinning(pts_can, model.embedding_xyz, bones_rst, 
                            rest_pose_code, nerf_skin, use_hs=opts.use_hs, skin_aux=model.skin_aux, joints_only=opts.joints_only)

        pts_dfm,bones_dfm = lbs(bones_rst, bone_rts_fw, skin_forward, 
                pts_can,backward=False)
        pts_dfm = pts_dfm[:,0]
        rt_dict['bones'] = bones_dfm
    vertices = pts_dfm.cpu().numpy()
    return vertices, rt_dict

def warp_fw_skel(opts, model, rt_dict, vertices, embedid, skeleton, bone_to_skeleton_pairs):
    """
    only used in mesh extraction
    """

    if opts.lbs:
        num_pts = vertices.shape[0]
        query_time = torch.ones(skeleton.joint_centers.shape[0],1).long().to(model.device)*embedid
        pts_can=torch.Tensor(vertices).to(model.device)
        # forward skinning
        pts_can = pts_can[:,None]
        bones_rst = model.bones
        bone_rts_fw = model.nerf_body_rts(query_time)
        bones_rst, bone_rts_rst = correct_bones(model, bones_rst)
        bone_rts_fw = correct_rest_pose(opts, bone_rts_fw, bone_rts_rst)

        if opts.nerf_skin:
            nerf_skin = model.nerf_skin
        else:
            nerf_skin = None
        rest_pose_code =  model.rest_pose_code
        rest_pose_code = rest_pose_code(torch.Tensor([0]).long().to(bones_rst.device))

        # forward warp on joints
        skin_forward = gauss_mlp_skinning(skeleton.joint_centers[:,None], model.embedding_xyz, bones_rst, 
                            rest_pose_code, nerf_skin, use_hs=opts.use_hs, skin_aux=model.skin_aux, joints_only=opts.joints_only)
        pts_dfm, bones_dfm = lbs(bones_rst, bone_rts_fw,
                        skin_forward, skeleton.joint_centers[:,None], backward=False)

        _, regulated_locations = skeleton.get_proper_transforms(pts_dfm.squeeze(1),num_pts)

        # joint_transforms = regulated_joint_transforms[0]
        # joint_transforms = joint_transforms.view(skeleton.joint_centers.shape[0],12).detach().cpu().numpy()

        query_time = torch.ones(num_pts,1).long().to(model.device)*embedid
        bone_rts_fw = model.nerf_body_rts(query_time)
        bone_rts_fw = correct_rest_pose(opts, bone_rts_fw, bone_rts_rst)
        skin_forward = gauss_mlp_skinning(pts_can, model.embedding_xyz, bones_rst, 
                            rest_pose_code, nerf_skin, use_hs=opts.use_hs, skin_aux=model.skin_aux)

        # get refined bone transforms
        refinement_transform = get_refined_bones_transforms(bones_rst, regulated_locations, bone_to_skeleton_pairs, num_pts, model.device)
        pts_dfm, bones_dfm = lbs(bones_rst, refinement_transform,
                        skin_forward, pts_can, backward=False)
        pts_dfm = pts_dfm[:,0]
        rt_dict['skeleton'] = regulated_locations
        rt_dict['bones'] = bones_dfm
    else:
        raise NotImplementedError
    vertices = pts_dfm.cpu().numpy()
    return vertices, rt_dict

def canonical2ndc(model, dp_canonical_pts, rtk, kaug, embedid):
    """
    dp_canonical_pts: 5004,3, pts in the canonical space of each video
    dp_px: bs, 5004, 3
    """
    Rmat = rtk[:,:3,:3]
    Tmat = rtk[:,:3,3]
    Kmat = K2mat(rtk[:,3,:])
    Kaug = K2inv(kaug) # p = Kaug Kmat P
    Kinv = Kmatinv(Kaug.matmul(Kmat))
    K = mat2K(Kmatinv(Kinv))
    bs = Kinv.shape[0]
    npts = dp_canonical_pts.shape[0]

    # projection
    dp_canonical_pts = dp_canonical_pts[None]
    if model.opts.flowbw:
        time_embedded = model.pose_code(embedid)
        time_embedded = time_embedded.repeat(1,npts, 1)
        dp_canonical_embedded = model.embedding_xyz(dp_canonical_pts)[None]
        dp_canonical_embedded = dp_canonical_embedded.repeat(bs,1,1)
        dp_canonical_embedded = torch.cat([dp_canonical_embedded, time_embedded], -1)
        dp_deformed_flo = model.nerf_flowfw(dp_canonical_embedded, xyz=dp_canonical_pts)
        dp_deformed_pts = dp_canonical_pts + dp_deformed_flo
    else:
        dp_deformed_pts = dp_canonical_pts.repeat(bs,1,1)
    dp_cam_pts = obj_to_cam(dp_deformed_pts, Rmat, Tmat) 
    dp_px = pinhole_cam(dp_cam_pts,K)
    return dp_px 

def get_near_far(near_far, vars_np, tol_fac=1.2, pts=None):
    """
    pts:        point coordinate N,3
    near_far:   near and far plane M,2
    rtk:        object to camera transform, M,4,4
    idk:        indicator of obsered or not M
    tol_fac     tolerance factor
    """
    if pts is None:
        #pts = vars_np['mesh_rest'].vertices
        # turn points to bounding box
        pts = trimesh.bounds.corners(vars_np['mesh_rest'].bounds)

    device = near_far.device
    rtk = torch.Tensor(vars_np['rtk']).to(device)
    idk = torch.Tensor(vars_np['idk']).to(device)

    pts = pts_to_view(pts, rtk, device)

    pmax = pts[...,-1].max(-1)[0]
    pmin = pts[...,-1].min(-1)[0]
    delta = (pmax - pmin)*(tol_fac-1)

    near= pmin-delta
    far = pmax+delta

    near_far[idk==1,0] = torch.clamp(near[idk==1], min=1e-3)
    near_far[idk==1,1] = torch.clamp( far[idk==1], min=1e-3)
    return near_far

def pts_to_view(pts, rtk, device):
    """
    object to camera coordinates
    pts:        point coordinate N,3
    rtk:        object to camera transform, M,4,4
    idk:        indicator of obsered or not M
    """
    M = rtk.shape[0]
    out_pts = []
    chunk=100
    for i in range(0,M,chunk):
        rtk_sub = rtk[i:i+chunk]
        pts_sub = torch.Tensor(np.tile(pts[None],
                        (len(rtk_sub),1,1))).to(device) # M,N,3
        pts_sub = obj_to_cam(pts_sub,  rtk_sub[:,:3,:3], 
                                       rtk_sub[:,:3,3])
        pts_sub = pinhole_cam(pts_sub, rtk_sub[:,3])
        out_pts.append(pts_sub)
    out_pts = torch.cat(out_pts, 0)
    return out_pts

def compute_point_visibility(pts, vars_np, device):
    """
    pts:        point coordinate N,3
    rtk:        object to camera transform, M,4,4
    idk:        indicator of obsered or not M
    **deprecated** due to K vars_tensor['rtk'] may not be consistent
    """
    vars_tensor = array2tensor(vars_np, device=device)
    rtk = vars_tensor['rtk']
    idk = vars_tensor['idk']
    vis = vars_tensor['vis']
    
    pts = pts_to_view(pts, rtk, device) # T, N, 3
    h,w = vis.shape[1:]

    vis = vis[:,None]
    xy = pts[:,None,:,:2] 
    xy[...,0] = xy[...,0]/w*2 - 1
    xy[...,1] = xy[...,1]/h*2 - 1

    # grab the visibility value in the mask and sum over frames
    vis = F.grid_sample(vis, xy)[:,0,0]
    vis = (idk[:,None]*vis).sum(0)
    vis = (vis>0).float() # at least seen in one view
    return vis


def near_far_to_bound(near_far):
    """
    near_far: T, 2 on cuda
    bound: float
    this can only be used for a single video (and for approximation)
    """
    bound=(near_far[:,1]-near_far[:,0]).mean() / 2
    bound = bound.detach().cpu().numpy()
    return bound


def rot_angle(mat):
    """
    rotation angle of rotation matrix 
    rmat: ..., 3,3
    """
    eps=1e-4
    cos = (  mat[...,0,0] + mat[...,1,1] + mat[...,2,2] - 1 )/2
    cos = cos.clamp(-1+eps,1-eps)
    angle = torch.acos(cos)
    return angle

def match2coords(match, w_rszd):
    tar_coord = torch.cat([match[:,None]%w_rszd, match[:,None]//w_rszd],-1)
    tar_coord = tar_coord.float()
    return tar_coord
    
def match2flo(match, w_rszd, img_size, warp_r, warp_t, device):
    ref_coord = sample_xy(w_rszd, 1, 0, device, return_all=True)[1].view(-1,2)
    ref_coord = ref_coord.matmul(warp_r[:2,:2]) + warp_r[None,:2,2]
    tar_coord = match2coords(match, w_rszd)
    tar_coord = tar_coord.matmul(warp_t[:2,:2]) + warp_t[None,:2,2]

    flo_dp = (tar_coord - ref_coord) / img_size * 2 # [-2,2]
    flo_dp = flo_dp.view(w_rszd, w_rszd, 2)
    flo_dp = flo_dp.permute(2,0,1)

    xygrid = sample_xy(w_rszd, 1, 0, device, return_all=True)[1] # scale to img_size
    xygrid = xygrid * float(img_size/w_rszd)
    warp_r_inv = Kmatinv(warp_r)
    xygrid = xygrid.matmul(warp_r_inv[:2,:2]) + warp_r_inv[None,:2,2]
    xygrid = xygrid / w_rszd * 2 - 1 
    flo_dp = F.grid_sample(flo_dp[None], xygrid.view(1,w_rszd,w_rszd,2))[0]
    return flo_dp

def compute_flow_cse(cse_a,cse_b, warp_a, warp_b, img_size):
    """
    compute the flow between two frames under cse feature matching
    assuming two feature images have the same dimension (also rectangular)
    cse:        16,h,w, feature image
    flo_dp:     2,h,w
    """
    _,_,w_rszd = cse_a.shape
    hw_rszd = w_rszd*w_rszd
    device = cse_a.device

    cost = (cse_b[:,None,None] * cse_a[...,None,None]).sum(0)
    _,match_a = cost.view(hw_rszd, hw_rszd).max(1)
    _,match_b = cost.view(hw_rszd, hw_rszd).max(0)

    flo_a = match2flo(match_a, w_rszd, img_size, warp_a, warp_b, device)
    flo_b = match2flo(match_b, w_rszd, img_size, warp_b, warp_a, device)
    return flo_a, flo_b

def compute_flow_geodist(dp_refr,dp_targ, geodists):
    """
    compute the flow between two frames under geodesic distance matching
    dps:        h,w, canonical surface mapping index
    geodists    N,N, distance matrix
    flo_dp:     2,h,w
    """
    h_rszd,w_rszd = dp_refr.shape
    hw_rszd = h_rszd*w_rszd
    device = dp_refr.device
    chunk = 1024

    # match: hw**2
    match = torch.zeros(hw_rszd).to(device)
    for i in range(0,hw_rszd,chunk):
        chunk_size = len(dp_refr.view(-1,1)[i:i+chunk] )
        dp_refr_sub = dp_refr.view(-1,1)[i:i+chunk].repeat(1,hw_rszd).view(-1,1)
        dp_targ_sub = dp_targ.view(1,-1)        .repeat(chunk_size,1).view(-1,1)
        match_sub = geodists[dp_refr_sub, dp_targ_sub]
        dis_geo_sub,match_sub = match_sub.view(-1, hw_rszd).min(1)
        #match_sub[dis_geo_sub>0.1] = 0
        match[i:i+chunk] = match_sub

    # cx,cy
    tar_coord = match2coords(match, w_rszd)
    ref_coord = sample_xy(w_rszd, 1, 0, device, return_all=True)[1].view(-1,2)
    ref_coord = ref_coord.view(h_rszd, w_rszd, 2)
    tar_coord = tar_coord.view(h_rszd, w_rszd, 2)
    flo_dp = (tar_coord - ref_coord) / w_rszd * 2 # [-2,2]
    match = match.view(h_rszd, w_rszd)
    flo_dp[match==0] = 0
    flo_dp = flo_dp.permute(2,0,1)
    return flo_dp

def compute_flow_geodist_old(dp_refr,dp_targ, geodists):
    """
    compute the flow between two frames under geodesic distance matching
    dps:        h,w, canonical surface mapping index
    geodists    N,N, distance matrix
    flo_dp:     2,h,w
    """
    h_rszd,w_rszd = dp_refr.shape
    hw_rszd = h_rszd*w_rszd
    device = dp_refr.device
    dp_refr = dp_refr.view(-1,1).repeat(1,hw_rszd).view(-1,1)
    dp_targ = dp_targ.view(1,-1).repeat(hw_rszd,1).view(-1,1)

    match = geodists[dp_refr, dp_targ]
    dis_geo,match = match.view(hw_rszd, hw_rszd).min(1)
    #match[dis_geo>0.1] = 0

    # cx,cy
    tar_coord = match2coords(match, w_rszd)
    ref_coord = sample_xy(w_rszd, 1, 0, device, return_all=True)[1].view(-1,2)
    ref_coord = ref_coord.view(h_rszd, w_rszd, 2)
    tar_coord = tar_coord.view(h_rszd, w_rszd, 2)
    flo_dp = (tar_coord - ref_coord) / w_rszd * 2 # [-2,2]
    match = match.view(h_rszd, w_rszd)
    flo_dp[match==0] = 0
    flo_dp = flo_dp.permute(2,0,1)
    return flo_dp



def fb_flow_check(flo_refr, flo_targ, img_refr, img_targ, dp_thrd, 
                    save_path=None):
    """
    apply forward backward consistency check on flow fields
    flo_refr: 2,h,w forward flow
    flo_targ: 2,h,w backward flow
    fberr:    h,w forward backward error
    """
    h_rszd, w_rszd = flo_refr.shape[1:]
    # clean up flow
    flo_refr = flo_refr.permute(1,2,0).cpu().numpy()
    flo_targ = flo_targ.permute(1,2,0).cpu().numpy()
    flo_refr_mask = np.linalg.norm(flo_refr,2,-1)>0 # this also removes 0 flows
    flo_targ_mask = np.linalg.norm(flo_targ,2,-1)>0
    flo_refr_px = flo_refr * w_rszd / 2
    flo_targ_px = flo_targ * w_rszd / 2

    #fb check
    x0,y0  =np.meshgrid(range(w_rszd),range(h_rszd))
    hp0 = np.stack([x0,y0],-1) # screen coord

    flo_fb = warp_flow(hp0 + flo_targ_px, flo_refr_px) - hp0
    flo_fb = 2*flo_fb/w_rszd
    fberr_fw = np.linalg.norm(flo_fb, 2,-1)
    fberr_fw[~flo_refr_mask] = 0

    flo_bf = warp_flow(hp0 + flo_refr_px, flo_targ_px) - hp0
    flo_bf = 2*flo_bf/w_rszd
    fberr_bw = np.linalg.norm(flo_bf, 2,-1)
    fberr_bw[~flo_targ_mask] = 0

    if save_path is not None:
        # vis
        thrd_vis = 0.01
        img_refr = F.interpolate(img_refr, (h_rszd, w_rszd), mode='bilinear')[0]
        img_refr = img_refr.permute(1,2,0).cpu().numpy()[:,:,::-1]
        img_targ = F.interpolate(img_targ, (h_rszd, w_rszd), mode='bilinear')[0]
        img_targ = img_targ.permute(1,2,0).cpu().numpy()[:,:,::-1]
        flo_refr[:,:,0] = (flo_refr[:,:,0] + 2)/2
        flo_targ[:,:,0] = (flo_targ[:,:,0] - 2)/2
        flo_refr[fberr_fw>thrd_vis]=0.
        flo_targ[fberr_bw>thrd_vis]=0.
        flo_refr[~flo_refr_mask]=0.
        flo_targ[~flo_targ_mask]=0.
        img = np.concatenate([img_refr, img_targ], 1)
        flo = np.concatenate([flo_refr, flo_targ], 1)
        imgflo = cat_imgflo(img, flo)
        imgcnf = np.concatenate([fberr_fw, fberr_bw],1)
        imgcnf = np.clip(imgcnf, 0, dp_thrd)*(255/dp_thrd)
        imgcnf = np.repeat(imgcnf[...,None],3,-1)
        imgcnf = cv2.resize(imgcnf, imgflo.shape[::-1][1:])
        imgflo_cnf = np.concatenate([imgflo, imgcnf],0)
        cv2.imwrite(save_path, imgflo_cnf)
    return fberr_fw, fberr_bw


def mask_aug(rendered):
    lb = 0.1;    ub = 0.3
    _,h,w=rendered.shape
    if np.random.binomial(1,0.5):
        sx = int(np.random.uniform(lb*w,ub*w))
        sy = int(np.random.uniform(lb*h,ub*h))
        cx = int(np.random.uniform(sx,w-sx))
        cy = int(np.random.uniform(sy,h-sy))
        feat_mean = rendered.mean(-1).mean(-1)[:,None,None]
        rendered[:,cx-sx:cx+sx,cy-sy:cy+sy] = feat_mean
    return rendered

def process_so3_seq(rtk_seq, vis=False, smooth=True):
    """
    rtk_seq, bs, N, 13 including
    {scoresx1, rotationsx9, translationsx3}
    """
    from utils.io import draw_cams
    scores =rtk_seq[...,0]
    bs,N = scores.shape
    rmat =  rtk_seq[...,1:10]
    tmat = rtk_seq[:,0,10:13]
    rtk_raw = rtk_seq[:,0,13:29].reshape((-1,4,4))
   
    distribution = torch.Tensor(scores).softmax(1)
    entropy = (-distribution.log() * distribution).sum(1)

    if vis:
        # draw distribution
        obj_scale = 3
        cam_space = obj_scale * 0.2
        tmat_raw = np.tile(rtk_raw[:,None,:3,3], (1,N,1))
        scale_factor = obj_scale/tmat_raw[...,-1].mean()
        tmat_raw *= scale_factor
        tmat_raw = tmat_raw.reshape((bs,12,-1,3))
        tmat_raw[...,-1] += np.linspace(-cam_space, cam_space,12)[None,:,None]
        tmat_raw = tmat_raw.reshape((bs,-1,3))
        # bs, tiltxae
        all_rts = np.concatenate([rmat, tmat_raw],-1)
        all_rts = np.transpose(all_rts.reshape(bs,N,4,3), [0,1,3,2])
    
        for i in range(bs):
            top_idx = scores[i].argsort()[-30:]
            top_rt = all_rts[i][top_idx]
            top_score = scores[i][top_idx]
            top_score = (top_score - top_score.min())/(top_score.max()-top_score.min())
            mesh = draw_cams(top_rt, color_list = top_score)
            mesh.export('tmp/%d.obj'%(i))
   
    if smooth:
        # graph cut scores, bsxN
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
        graph = dcrf.DenseCRF2D(bs, 1, N)  # width, height, nlabels
        unary = unary_from_softmax(distribution.numpy().T.copy())
        graph.setUnaryEnergy(unary)
        grid = rmat[0].reshape((N,3,3))
        drot = np.matmul(grid[None], np.transpose(grid[:,None], (0,1,3,2)))
        drot = rot_angle(torch.Tensor(drot))
        compat = (-2*(drot).pow(2)).exp()*10
        compat = compat.numpy()
        graph.addPairwiseGaussian(sxy=10, compat=compat)

        Q = graph.inference(100)
        scores = np.asarray(Q).T

    # argmax
    idx_max = scores.argmax(-1)
    rmat = rmat[0][idx_max]

    rmat = rmat.reshape((-1,9))
    rts = np.concatenate([rmat, tmat],-1)
    rts = rts.reshape((bs,1,-1))

    # post-process se3
    root_rmat = rts[:,0,:9].reshape((-1,3,3))
    root_tmat = rts[:,0,9:12]
    
    rmat = rtk_raw[:,:3,:3]
    tmat = rtk_raw[:,:3,3]
    tmat = tmat + np.matmul(rmat, root_tmat[...,None])[...,0]
    rmat = np.matmul(rmat, root_rmat)
    rtk_raw[:,:3,:3] = rmat
    rtk_raw[:,:3,3] = tmat
   
    if vis:
        # draw again
        pdb.set_trace()
        rtk_vis = rtk_raw.copy()
        rtk_vis[:,:3,3] *= scale_factor
        mesh = draw_cams(rtk_vis)
        mesh.export('tmp/final.obj')
    return rtk_raw

def align_sim3(rootlist_a, rootlist_b, is_inlier=None, err_valid=None):
    """
    nx4x4 matrices
    is_inlier: n
    """
#    ta = np.matmul(-np.transpose(rootlist_a[:,:3,:3],[0,2,1]), 
#                                 rootlist_a[:,:3,3:4])
#    ta = ta[...,0].T
#    tb = np.matmul(-np.transpose(rootlist_b[:,:3,:3],[0,2,1]), 
#                                 rootlist_b[:,:3,3:4])
#    tb = tb[...,0].T
#    dso3,dtrn,dscale=umeyama_alignment(tb, ta,with_scale=False)
#    
#    dscale = np.linalg.norm(rootlist_a[0,:3,3],2,-1) /\
#             np.linalg.norm(rootlist_b[0,:3,3],2,-1)
#    rootlist_b[:,:3,:3] = np.matmul(rootlist_b[:,:3,:3], dso3.T[None])
#    rootlist_b[:,:3,3:4] = rootlist_b[:,:3,3:4] - \
#            np.matmul(rootlist_b[:,:3,:3], dtrn[None,:,None]) 

    dso3 = np.matmul(np.transpose(rootlist_b[:,:3,:3],(0,2,1)),
                        rootlist_a[:,:3,:3])
    dscale = np.linalg.norm(rootlist_a[:,:3,3],2,-1)/\
            np.linalg.norm(rootlist_b[:,:3,3],2,-1)

    # select inliers to fit 
    if is_inlier is not None:
        if is_inlier.sum() == 0:
            is_inlier[np.argmin(err_valid)] = True
        dso3 = dso3[is_inlier]
        dscale = dscale[is_inlier]

    dso3 = R.from_matrix(dso3).mean().as_matrix()
    rootlist_b[:,:3,:3] = np.matmul(rootlist_b[:,:3,:3], dso3[None])

    dscale = dscale.mean()
    rootlist_b[:,:3,3] = rootlist_b[:,:3,3] * dscale

    so3_err = np.matmul(rootlist_a[:,:3,:3], 
            np.transpose(rootlist_b[:,:3,:3],[0,2,1]))
    so3_err = rot_angle(torch.Tensor(so3_err))
    so3_err = so3_err / np.pi*180
    so3_err_max = so3_err.max()
    so3_err_mean = so3_err.mean()
    so3_err_med = np.median(so3_err)
    so3_err_std = np.asarray(so3_err.std())
    print(so3_err)
    print('max  so3 error (deg): %.1f'%(so3_err_max))
    print('med  so3 error (deg): %.1f'%(so3_err_med))
    print('mean so3 error (deg): %.1f'%(so3_err_mean))
    print('std  so3 error (deg): %.1f'%(so3_err_std))

    return rootlist_b

def align_sfm_sim3(aux_seq, datasets):
    from utils.io import draw_cams, load_root
    for dataset in datasets:
        seqname = dataset.imglist[0].split('/')[-2]

        # only process dataset with rtk_path input
        if dataset.has_prior_cam:
            root_dir = dataset.rtklist[0][:-9]
            root_sfm = load_root(root_dir, 0)[:-1] # excluding the last

            # split predicted root into multiple sequences
            seq_idx = [seqname == i.split('/')[-2] for i in aux_seq['impath']]
            root_pred = aux_seq['rtk'][seq_idx]
            is_inlier = aux_seq['is_valid'][seq_idx]
            err_valid = aux_seq['err_valid'][seq_idx]
            # only use certain ones to match
            #pdb.set_trace()
            #mesh = draw_cams(root_sfm, color='gray')
            #mesh.export('0.obj')
            
            # pre-align the center according to cat mask
            root_sfm = visual_hull_align(root_sfm, 
                    aux_seq['kaug'][seq_idx],
                    aux_seq['masks'][seq_idx])

            root_sfm = align_sim3(root_pred, root_sfm, 
                    is_inlier=is_inlier, err_valid=err_valid)
            # only modify rotation
            #root_pred[:,:3,:3] = root_sfm[:,:3,:3]
            root_pred = root_sfm
            
            aux_seq['rtk'][seq_idx] = root_pred
            aux_seq['is_valid'][seq_idx] = True
        else:
            print('not aligning %s, no rtk path in config file'%seqname)

def visual_hull_align(rtk, kaug, masks):
    """
    input: array
    output: array
    """
    rtk = torch.Tensor(rtk)
    kaug = torch.Tensor(kaug)
    masks = torch.Tensor(masks)
    num_view,h,w = masks.shape
    grid_size = 64
   
    if rtk.shape[0]!=num_view:
        print('rtk size mismtach: %d vs %d'%(rtk.shape[0], num_view))
        rtk = rtk[:num_view]
        
    rmat = rtk[:,:3,:3]
    tmat = rtk[:,:3,3:]

    Kmat = K2mat(rtk[:,3])
    Kaug = K2inv(kaug) # p = Kaug Kmat P
    kmat = mat2K(Kaug.matmul(Kmat))

    rmatc = rmat.permute((0,2,1))
    tmatc = -rmatc.matmul(tmat)

    bound = tmatc.norm(2,-1).mean()
    pts = np.linspace(-bound, bound, grid_size).astype(np.float32)
    query_yxz = np.stack(np.meshgrid(pts, pts, pts), -1)  # (y,x,z)
    query_yxz = torch.Tensor(query_yxz).view(-1, 3)
    query_xyz = torch.cat([query_yxz[:,1:2], query_yxz[:,0:1], query_yxz[:,2:3]],-1)

    score_xyz = []
    chunk = 1000
    for i in range(0,len(query_xyz),chunk):
        query_xyz_chunk = query_xyz[None, i:i+chunk].repeat(num_view, 1,1)
        query_xyz_chunk = obj_to_cam(query_xyz_chunk, rmat, tmat)
        query_xyz_chunk = pinhole_cam(query_xyz_chunk, kmat)

        query_xy = query_xyz_chunk[...,:2]
        query_xy[...,0] = query_xy[...,0]/w*2-1
        query_xy[...,1] = query_xy[...,1]/h*2-1

        # sum over time
        score = F.grid_sample(masks[:,None], query_xy[:,None])[:,0,0]
        score = score.sum(0)
        score_xyz.append(score)

    # align the center
    score_xyz = torch.cat(score_xyz)
    center = query_xyz[score_xyz>0.8*num_view]
    print('%d points used to align center'% (len(center)) )
    center = center.mean(0)
    tmatc = tmatc - center[None,:,None]
    tmat = np.matmul(-rmat, tmatc)
    rtk[:,:3,3:] = tmat

    return rtk

def ood_check_cse(dp_feats, dp_embed, dp_idx):
    """
    dp_feats: bs,16,h,w
    dp_idx:   bs, h,w
    dp_embed: N,16
    valid_list bs
    """
    bs,_,h,w = dp_feats.shape
    N,_ = dp_embed.shape
    device = dp_feats.device
    dp_idx = F.interpolate(dp_idx.float()[None], (h,w), mode='nearest').long()[0]
    
    ## dot product 
    #pdb.set_trace()
    #err_list = []
    #err_threshold = 0.05
    #for i in range(bs):
    #    err = 1- (dp_embed[dp_idx[i]]*dp_feats[i].permute(1,2,0)).sum(-1)
    #    err_list.append(err)

    # fb check
    err_list = []
    err_threshold = 12
    # TODO no fb check
    #err_threshold = 100
    for i in range(bs):
        # use chunk
        chunk = 5000
        max_idx = torch.zeros(N).to(device)
        for j in range(0,N,chunk):
            costmap = (dp_embed.view(N,16,1)[j:j+chunk]*\
                    dp_feats[i].view(1,16,h*w)).sum(-2)
            max_idx[j:j+chunk] = costmap.argmax(-1)  #  N
    
        rpj_idx = max_idx[dp_idx[i]]
        rpj_coord = torch.stack([rpj_idx % w, rpj_idx//w],-1)
        ref_coord = sample_xy(w, 1, 0, device, return_all=True)[1].view(h,w,2)
        err = (rpj_coord - ref_coord).norm(2,-1) 
        err_list.append(err)

    valid_list = []
    error_list = []
    for i in range(bs):
        err = err_list[i]
        mean_error = err[dp_idx[i]!=0].mean()
        is_valid = mean_error < err_threshold
        error_list.append( mean_error)
        valid_list.append( is_valid  )
        #cv2.imwrite('tmp/%05d.png'%i, (err/mean_error).cpu().numpy()*100)
        #print(i); print(mean_error)
    error_list = torch.stack(error_list,0)
    valid_list = torch.stack(valid_list,0)

    return valid_list, error_list

def bbox_dp2rnd(bbox, kaug):
    """
    bbox: bs, 4
    kaug: bs, 4
    cropab2: bs, 3,3, transformation from dp bbox to rendered bbox coords
    """
    cropa2im = torch.cat([(bbox[:,2:] - bbox[:,:2]) / 112., 
                           bbox[:,:2]],-1)
    cropa2im = K2mat(cropa2im)
    im2cropb = K2inv(kaug) 
    cropa2b = im2cropb.matmul(cropa2im)
    return cropa2b
            



def resample_dp(dp_feats, dp_bbox, kaug, target_size):
    """
    dp_feats: bs, 16, h,w
    dp_bbox:  bs, 4
    kaug:     bs, 4
    """
    # if dp_bbox are all zeros, just do the resizing
    if dp_bbox.abs().sum()==0:
        dp_feats_rsmp = F.interpolate(dp_feats, (target_size, target_size),
                                                            mode='bilinear')
    else:
        dp_size = dp_feats.shape[-1]
        device = dp_feats.device

        dp2rnd = bbox_dp2rnd(dp_bbox, kaug)
        rnd2dp = Kmatinv(dp2rnd)
        xygrid = sample_xy(target_size, 1, 0, device, return_all=True)[1] 
        xygrid = xygrid.matmul(rnd2dp[:,:2,:2]) + rnd2dp[:,None,:2,2]
        xygrid = xygrid / dp_size * 2 - 1 
        dp_feats_rsmp = F.grid_sample(dp_feats, xygrid.view(-1,target_size,target_size,2))
    return dp_feats_rsmp


def vrender_flo(weights_coarse, xyz_coarse_target, xys, img_size):
    """
    weights_coarse:     ..., ndepth
    xyz_coarse_target:  ..., ndepth, 3
    flo_coarse:         ..., 2
    flo_valid:          ..., 1
    """
    # render flow 
    weights_coarse = weights_coarse.clone()
    xyz_coarse_target = xyz_coarse_target.clone()

    # bs, nsamp, -1, x
    weights_shape = weights_coarse.shape
    xyz_coarse_target = xyz_coarse_target.view(weights_shape+(3,))
    xy_coarse_target = xyz_coarse_target[...,:2]

    # deal with negative z
    invalid_ind = torch.logical_or(xyz_coarse_target[...,-1]<1e-5,
                           xy_coarse_target.norm(2,-1).abs()>2*img_size)
    weights_coarse[invalid_ind] = 0.
    xy_coarse_target[invalid_ind] = 0.

    # renormalize
    weights_coarse = weights_coarse/(1e-9+weights_coarse.sum(-1)[...,None])

    # candidate motion vector
    xys_unsq = xys.view(weights_shape[:-1]+(1,2))
    flo_coarse = xy_coarse_target - xys_unsq
    flo_coarse =  weights_coarse[...,None] * flo_coarse
    flo_coarse = flo_coarse.sum(-2)

    ## candidate target point
    #xys_unsq = xys.view(weights_shape[:-1]+(2,))
    #xy_coarse_target = weights_coarse[...,None] * xy_coarse_target
    #xy_coarse_target = xy_coarse_target.sum(-2)
    #flo_coarse = xy_coarse_target - xys_unsq

    flo_coarse = flo_coarse/img_size * 2
    flo_valid = (invalid_ind.sum(-1)==0).float()[...,None]
    return flo_coarse, flo_valid

def diff_flo(pts_target, xys, img_size):
    """
    pts_target:         ..., 1, 2
    xys:                ..., 2
    flo_coarse:         ..., 2
    flo_valid:          ..., 1
    """

    # candidate motion vector
    pts_target = pts_target.view(xys.shape)
    flo_coarse = pts_target - xys
    flo_coarse = flo_coarse/img_size * 2
    return flo_coarse

def fid_reindex(fid, num_vids, vid_offset):
    """
    re-index absolute frameid {0,....N} to subsets of video id and relative frameid
    fid: N absolution id
    vid: N video id
    tid: N relative id
    """
    tid = torch.zeros_like(fid).float()
    vid = torch.zeros_like(fid)
    max_ts = (vid_offset[1:] - vid_offset[:-1]).max()
    for i in range(num_vids):
        assign = torch.logical_and(fid>=vid_offset[i],
                                    fid<vid_offset[i+1])
        vid[assign] = i
        tid[assign] = fid[assign].float() - vid_offset[i]
        doffset = vid_offset[i+1] - vid_offset[i]
        tid[assign] = (tid[assign] - doffset/2)/max_ts*2
        #tid[assign] = 2*(tid[assign] / doffset)-1
        #tid[assign] = (tid[assign] - doffset/2)/1000.
    return vid, tid

def pts2line(pts, lines):
    '''
    Calculate points-to-bone distance. Point to line segment distance refer to
    https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    :param pts: N*3
    :param lines: N*6, where [N,0:3] is the starting position and [N, 3:6] is the ending position
    :return: origins are the neatest projected position of the point on the line.
             ends are the points themselves.
             dist is the distance in between, which is the distance from points to lines.
             Origins and ends will be used for generate rays.
    '''
    l2 = np.sum((lines[:, 3:6] - lines[:, 0:3]) ** 2, axis=1)
    origins = np.zeros((len(pts) * len(lines), 3))
    ends = np.zeros((len(pts) * len(lines), 3))
    dist = np.zeros((len(pts) * len(lines)))
    segments = np.zeros((len(pts),len(lines)))
    for l in range(len(lines)):
        if np.abs(l2[l]) < 1e-8:  # for zero-length edges
            origins[l * len(pts):(l + 1) * len(pts)] = lines[l][0:3]
        else:  # for other edges
            t = np.sum((pts - lines[l][0:3][np.newaxis, :]) * (lines[l][3:6] - lines[l][0:3])[np.newaxis, :], axis=1) / \
                l2[l]
            t = np.clip(t, 0, 1)
            segments[:,l] = t
            t_pos = lines[l][0:3][np.newaxis, :] + t[:, np.newaxis] * (lines[l][3:6] - lines[l][0:3])[np.newaxis, :]
            origins[l * len(pts):(l + 1) * len(pts)] = t_pos
        ends[l * len(pts):(l + 1) * len(pts)] = pts
        dist[l * len(pts):(l + 1) * len(pts)] = np.linalg.norm(
            origins[l * len(pts):(l + 1) * len(pts)] - ends[l * len(pts):(l + 1) * len(pts)], axis=1)

    return origins, ends, dist, segments

def pts2line_torch(pts, lines):
    '''
    Calculate points-to-bone distance. Point to line segment distance refer to
    https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    :param pts: N*3
    :param lines: N*6, where [N,0:3] is the starting position and [N, 3:6] is the ending position
    :return: origins are the neatest projected position of the point on the line.
             ends are the points themselves.
             dist is the distance in between, which is the distance from points to lines.
             Origins and ends will be used for generate rays.
    '''
    l2 = torch.sum((lines[:, 3:6] - lines[:, 0:3]) ** 2, dim=1)
    origins = torch.zeros((len(pts) * len(lines), 3))
    ends = torch.zeros((len(pts) * len(lines), 3))
    dist = torch.zeros((len(pts) * len(lines)))
    segments = torch.zeros((len(pts),len(lines)))

    for l in range(len(lines)):
        if torch.abs(l2[l]) < 1e-8:  # for zero-length edges
            origins[l * len(pts):(l + 1) * len(pts)] = lines[l][0:3]
        else:  # for other edges
            t = torch.sum((pts - lines[l][0:3][None, :]) * (lines[l][3:6] - lines[l][0:3])[None, :], dim=1) / \
                l2[l]
            t = torch.clip(t, 0, 1)
            segments[:,l] = t
            t_pos = lines[l][0:3][None, :] + t[:, None] * (lines[l][3:6] - lines[l][0:3])[None, :]
            origins[l * len(pts):(l + 1) * len(pts)] = t_pos
        ends[l * len(pts):(l + 1) * len(pts)] = pts
        dist[l * len(pts):(l + 1) * len(pts)] = torch.linalg.norm(
            origins[l * len(pts):(l + 1) * len(pts)] - ends[l * len(pts):(l + 1) * len(pts)], dim=1)

    return origins, ends, dist, segments

def calc_pts2bone_visible_mat(mesh, origins, ends):
    '''
    Check whether the surface point is visible by the internal bone.
    Visible is defined as no occlusion on the path between.
    :param mesh:
    :param surface_pts: points on the surface (n*3)
    :param origins: origins of rays
    :param ends: ends of the rays, together with origins, we can decide the direction of the ray.
    :return: binary visibility matrix (n*m), where 1 indicate the n-th surface point is visible to the m-th ray
    '''
    ray_dir = ends - origins
    RayMeshIntersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    locations, index_ray, index_tri = RayMeshIntersector.intersects_location(origins, ray_dir + 1e-15)
    locations_per_ray = [locations[index_ray == i] for i in range(len(ray_dir))]
    min_hit_distance = []
    for i in range(len(locations_per_ray)):
        if len(locations_per_ray[i]) == 0:
            min_hit_distance.append(np.linalg.norm(ray_dir[i]))
        else:
            min_hit_distance.append(np.min(np.linalg.norm(locations_per_ray[i] - origins[i], axis=1)))
    min_hit_distance = np.array(min_hit_distance)
    distance = np.linalg.norm(ray_dir, axis=1)
    vis_mat = (np.abs(min_hit_distance - distance) < 1e-10)

    return vis_mat

def volumetric_geodesic_dist_pts_to_bones(pts_bone_visibility, surface_geodesic, pts_bone_dist, skeleton_bones_names, return_freq=False):
    visible_matrix = np.zeros(pts_bone_visibility.shape)
    visible_matrix[np.where(pts_bone_visibility == 1)] = pts_bone_dist[np.where(pts_bone_visibility == 1)]

    for c in range(visible_matrix.shape[1]):
        unvisible_pts = np.argwhere(pts_bone_visibility[:, c] == 0).squeeze(1)
        visible_pts = np.argwhere(pts_bone_visibility[:, c] == 1).squeeze(1)
        if len(visible_pts) == 0:
            visible_matrix[:, c] = pts_bone_dist[:, c]
            print('no visible bones')
            continue
        for r in unvisible_pts:
            dist1 = np.min(surface_geodesic[r, visible_pts])
            nn_visible = visible_pts[np.argmin(surface_geodesic[r, visible_pts])]
            if np.isinf(dist1):
                print('inf geodesic distance')
                visible_matrix[r, c] = 100.0 + pts_bone_dist[r, c]
            else:
                visible_matrix[r, c] = dist1 + visible_matrix[nn_visible, c]
    if return_freq:
        joints_freq = np.zeros((1,visible_matrix.shape[1]+1))
        for bone_idx in range(visible_matrix.shape[1]):
            current_bone = skeleton_bones_names[bone_idx]
            # print(current_bone)
            # idx 0 is parent, idx 1 is child
            joints_freq[0,current_bone[0]] += 1
            joints_freq[0,current_bone[1]] += 1
    else:
        joints_freq = None
    return visible_matrix, joints_freq

def load_skeleton(pickle_file_path, device, canonical_mesh=None, need_traversal_info=False, residual_update=False):
    with open(pickle_file_path, 'rb') as f:
        extracted_joints = pickle.load(f)

    joints_center = np.zeros((len(extracted_joints.keys()), 3))
    joint_connections = np.zeros((joints_center.shape[0]-1,2)).astype(np.int64)

    idx_mapping = {}

    row_idx = 0
    connection_idx = 0
    for joint in extracted_joints.keys():
        joints_center[row_idx, :] = extracted_joints[joint][0]
        idx_mapping[joint] = row_idx
        row_idx += 1

    for joint in extracted_joints.keys():
        for node in extracted_joints[joint][1]:
            # print([idx_mapping[joint], idx_mapping[node]])
            joint_connections[connection_idx,:] = [idx_mapping[joint], idx_mapping[node]]
            connection_idx += 1


        # print(joints)
    joints_center = torch.from_numpy(joints_center).float().to(device)
    joint_connections = torch.from_numpy(joint_connections)
    constructed_skeleton = Skeleton(joints_center, joint_connections, device, canonical_mesh=canonical_mesh, need_traversal_info=need_traversal_info, residual_update=residual_update)
    return constructed_skeleton

def get_interpolated_skinning_weights(constructed_skeleton, bones_rst):
    skeleton_bones, skeleton_bones_names = constructed_skeleton.get_bones()

    canonical_bones = bones_rst.detach().cpu().numpy()
    canonical_bones = canonical_bones[:,:3]
    _, _, pts_bone_dist, segments = pts2line(canonical_bones, skeleton_bones)

    pts_bone_dist = pts_bone_dist.reshape(len(skeleton_bones), len(canonical_bones)).transpose()

    closest_bones = np.argmin(pts_bone_dist, axis=1)

    refine_camm_transform = []
    for camm_bone in range(segments.shape[0]):
        closest_bone = closest_bones[camm_bone]
        closest_bone_parent_idx, closest_bone_child_idx = skeleton_bones_names[closest_bone][0], skeleton_bones_names[closest_bone][1]
        t = segments[camm_bone, closest_bone]
        parent_transform = 1 - t
        child_transform = t
        skeleton_to_bone_dist = pts_bone_dist[camm_bone, closest_bone]
        bone_dir_vec = constructed_skeleton.joint_centers[closest_bone_child_idx] - constructed_skeleton.joint_centers[closest_bone_parent_idx]
        offset_from_skeleton_to_bone = bones_rst[camm_bone,:3] - (constructed_skeleton.joint_centers[closest_bone_parent_idx] + t * bone_dir_vec)
        offset_from_skeleton_to_bone = offset_from_skeleton_to_bone.detach().cpu().numpy().reshape(3,)
        bone_dir_vec = bone_dir_vec.detach().cpu().numpy().reshape(3,)
        rotation_from_skeleton_to_bone = rotation_matrix_from_vectors(bone_dir_vec, offset_from_skeleton_to_bone)
        saved_result = [[closest_bone_parent_idx, closest_bone_child_idx], [parent_transform, child_transform], rotation_from_skeleton_to_bone, skeleton_to_bone_dist]
        refine_camm_transform.append(saved_result)
    
    return refine_camm_transform

def get_interpolated_skinning_weights_torch(constructed_skeleton, bones_rst):
    skeleton_bones, skeleton_bones_names = constructed_skeleton.get_bones_torch()

    # canonical_bones = bones_rst.detach().cpu().numpy()
    canonical_bones = bones_rst[:,:3]
    _, _, pts_bone_dist, segments = pts2line_torch(canonical_bones, skeleton_bones)

    pts_bone_dist = pts_bone_dist.view(len(skeleton_bones), len(canonical_bones)).permute(1,0)

    closest_bones = torch.argmin(pts_bone_dist, dim=1)

    refine_camm_transform = []
    for camm_bone in range(segments.shape[0]):
        closest_bone = closest_bones[camm_bone]
        closest_bone_parent_idx, closest_bone_child_idx = skeleton_bones_names[closest_bone][0], skeleton_bones_names[closest_bone][1]
        t = segments[camm_bone, closest_bone]
        parent_transform = 1 - t
        child_transform = t
        skeleton_to_bone_dist = pts_bone_dist[camm_bone, closest_bone]
        bone_dir_vec = constructed_skeleton.joint_centers[closest_bone_child_idx] - constructed_skeleton.joint_centers[closest_bone_parent_idx]
        offset_from_skeleton_to_bone = bones_rst[camm_bone,:3] - (constructed_skeleton.joint_centers[closest_bone_parent_idx] + t * bone_dir_vec)
        offset_from_skeleton_to_bone = offset_from_skeleton_to_bone.view(3,)
        bone_dir_vec = bone_dir_vec.view(3,)
        rotation_from_skeleton_to_bone = rotation_matrix_from_vectors_torch(bone_dir_vec, offset_from_skeleton_to_bone)
        saved_result = [[closest_bone_parent_idx, closest_bone_child_idx], [parent_transform, child_transform], rotation_from_skeleton_to_bone, skeleton_to_bone_dist]
        refine_camm_transform.append(saved_result)
    
    return refine_camm_transform

def get_refined_bones_transforms(bones_rst, regulated_locations, refine_camm_transform, num_pts, device):
    refinement_transform = np.zeros((bones_rst.shape[0], 12))
    identity_rotation = np.eye(3).reshape(-1,9)
    refinement_transform[:,:9] = identity_rotation
    new_joints_loc = regulated_locations.detach().cpu().numpy()
    canonical_bones = bones_rst.detach().cpu().numpy()
    for camm_bone in range(bones_rst.shape[0]):
        refine_stats = refine_camm_transform[camm_bone]
        parent_joint_idx, child_joint_idx = refine_stats[0][0], refine_stats[0][1]
        parent_weight, child_weight = refine_stats[1][0], refine_stats[1][1]
        rmat_from_skeleton_to_bone = refine_stats[2]
        skeleton_to_bone_dist = refine_stats[3]
        bone_vec = (new_joints_loc[child_joint_idx] - new_joints_loc[parent_joint_idx]).reshape(3,)
        bone_vec = bone_vec / (np.linalg.norm(bone_vec) + 1e-10)

        skeleton_to_bone_vec = rmat_from_skeleton_to_bone.dot(bone_vec) * skeleton_to_bone_dist
        new_camm_bone_loc = skeleton_to_bone_vec + new_joints_loc[parent_joint_idx] + (new_joints_loc[child_joint_idx] - new_joints_loc[parent_joint_idx]) * child_weight
        refinement_transform[camm_bone,9:] = new_camm_bone_loc - canonical_bones[camm_bone,:3]
    
    refinement_transform = torch.from_numpy(refinement_transform).float().to(device).view(-1,1,bones_rst.shape[0]*12)

    refinement_transform = refinement_transform.repeat(num_pts,1,1)
    return refinement_transform

def get_refined_bones_transforms_torch(bones_rst, regulated_locations, refine_camm_transform, num_pts, device):
    refinement_transform = torch.zeros((bones_rst.shape[0], 12))
    identity_rotation = torch.eye(3).reshape(-1,9)
    refinement_transform[:,:9] = identity_rotation
    refinement_transform = refinement_transform.to(device)

    new_joints_loc = regulated_locations
    canonical_bones = bones_rst
    for camm_bone in range(bones_rst.shape[0]):
        refine_stats = refine_camm_transform[camm_bone]
        parent_joint_idx, child_joint_idx = refine_stats[0][0], refine_stats[0][1]
        parent_weight, child_weight = refine_stats[1][0], refine_stats[1][1]
        rmat_from_skeleton_to_bone = refine_stats[2]
        skeleton_to_bone_dist = refine_stats[3]
        bone_vec = (new_joints_loc[child_joint_idx] - new_joints_loc[parent_joint_idx]).reshape(3,)
        bone_vec = bone_vec / (torch.linalg.norm(bone_vec) + 1e-10)

        skeleton_to_bone_vec = rmat_from_skeleton_to_bone.matmul(bone_vec) * skeleton_to_bone_dist
        new_camm_bone_loc = skeleton_to_bone_vec + new_joints_loc[parent_joint_idx] + (new_joints_loc[child_joint_idx] - new_joints_loc[parent_joint_idx]) * child_weight
        refinement_transform[camm_bone,9:] = new_camm_bone_loc - canonical_bones[camm_bone,:3]
    
    refinement_transform = refinement_transform.view(-1,1,bones_rst.shape[0]*12)

    refinement_transform = refinement_transform.repeat(num_pts,1,1)
    return refinement_transform

def get_refined_bones_transforms_torch_batch(bones_rst, regulated_locations, refine_camm_transform, num_pts, device):
    assert len(regulated_locations.shape) == 3, 'regulated locations must contain all locations'
    refinement_transform = torch.zeros((regulated_locations.shape[0], bones_rst.shape[0], 12))
    identity_rotation = torch.eye(3).reshape(-1,9)
    refinement_transform[:,:,:9] = identity_rotation
    refinement_transform = refinement_transform.to(device)

    new_joints_loc = regulated_locations
    canonical_bones = bones_rst
    for camm_bone in range(bones_rst.shape[0]):
        refine_stats = refine_camm_transform[camm_bone]
        parent_joint_idx, child_joint_idx = refine_stats[0][0], refine_stats[0][1]
        parent_weight, child_weight = refine_stats[1][0], refine_stats[1][1]
        rmat_from_skeleton_to_bone = refine_stats[2]
        skeleton_to_bone_dist = refine_stats[3]
        bone_vec = (new_joints_loc[:,child_joint_idx,:] - new_joints_loc[:,parent_joint_idx,:])

        bone_vec_norm = bone_vec.pow(2).sum(dim=-1).sqrt()

        bone_vec = bone_vec / (bone_vec_norm[:,None] + 1e-10)
        bone_vec = bone_vec.permute(1,0)

        skeleton_to_bone_vec = (rmat_from_skeleton_to_bone.matmul(bone_vec)).permute(1,0) * skeleton_to_bone_dist

        new_camm_bone_loc = skeleton_to_bone_vec + new_joints_loc[:,parent_joint_idx,:] + (new_joints_loc[:,child_joint_idx,:] - new_joints_loc[:,parent_joint_idx,:]) * child_weight
        refinement_transform[:,camm_bone,9:] = new_camm_bone_loc - canonical_bones[camm_bone,:3]
    
    return refinement_transform

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)

    if any(v):
      c = np.dot(a, b)
      s = np.linalg.norm(v)
      kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
      rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
      return rotation_matrix
    
    else:
      assert not np.allclose(vec1, (-1*vec2)), "two vectors' directions are exactly in the opposite direction"
      return np.eye(3)

def rotation_matrix_from_vectors_torch(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / torch.linalg.norm(vec1)).view(3,), (vec2 / torch.linalg.norm(vec2)).view(3,)
    v = torch.cross(a, b)

    if any(v):
      c = torch.dot(a, b)
      s = torch.linalg.norm(v)
      kmat = torch.Tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]).to(v.device)
      rotation_matrix = torch.eye(3).to(v.device) + kmat + kmat.matmul(kmat) * ((1 - c) / (s ** 2))
      return rotation_matrix
    
    else:
      assert not torch.allclose(vec1, (-1*vec2)), "two vectors' directions are exactly in the opposite direction"
      return torch.eye(3)

# kinematic chain joint definietion
class Joint(object):
    def __init__(self, idx, canonical_pose, parent=None, children=None):
        self.idx = idx

        if children is None:
            # leaf joint
            self.children = []
        else:
            self.children = children

        # root joint has parent = None
        self.parent = parent
        
        # canonical pose is defined w.r.t to parent joint
        # for root joint, it is defined w.r.t global canonical space coordinate
        self.canonical_pose = canonical_pose
        self.canonical_pose.requires_grad_(False)

        self.length = self.canonical_pose[:3,3].pow(2).sum().sqrt().requires_grad_(False)

# kinematic chain definition
class Skeleton(object):
    def __init__(self, joint_centers, joint_connections, device, canonical_mesh=None, need_traversal_info=False, residual_update=False):
        self.joint_centers = joint_centers
        self.joint_connections = joint_connections
        self.joint_centers.requires_grad_(False)
        self.joint_connections.requires_grad_(False)
        self.device = device
        self.residual_update = residual_update

        self.children_mapping, self.parents_mapping = {}, {}
        for i in range(joint_centers.shape[0]):
            self.children_mapping[i] = []
        
        for i in range(joint_connections.shape[0]):
            self.children_mapping[joint_connections[i,0].item()] += [joint_connections[i,1].item()]
            self.parents_mapping[joint_connections[i,1].item()] = joint_connections[i,0].item()

        # first joint is the root joint
        self.root_canonical_pose = torch.eye(4)
        self.root_canonical_pose[:3,3] = joint_centers[0,:].view(-1,3)
        self.root_canonical_pose = self.root_canonical_pose.to(self.device)
        # print(self.root_canonical_pose)
        self.root = Joint(0, self.root_canonical_pose, parent=None, children=self.children_mapping[0])

        # add rest of the joints in bfs order
        self.joint_list = [self.root]
        for i in range(1,joint_centers.shape[0]):
            relative_canonical_pose = torch.eye(4).to(self.device)
            relative_canonical_pose[:3,3] = joint_centers[i,:].view(-1,3) - joint_centers[self.parents_mapping[i],:].view(-1,3)
            self.joint_list.append(Joint(i, relative_canonical_pose, parent=self.parents_mapping[i], children=self.children_mapping[i]))
        
        if canonical_mesh is not None:
            self.canonical_mesh = canonical_mesh
        
        if need_traversal_info:
            self.traversal_distance_matrix = self.traversal_dist()
            # print(self.traversal_distance_matrix)
            self.traversal_children_parent_list = self.get_bone_children_parent_list()
        
        if self.residual_update:
            self.bone_length_list = torch.zeros(joint_centers.shape[0]-1,)
            for i in range(1,joint_centers.shape[0]):
                self.bone_length_list[i-1] = self.joint_list[i].length.clone()
            self.bone_length_list = self.bone_length_list.to(self.device)
            self.fixed_bone_length_list = self.bone_length_list.clone()
            self.fixed_bone_length_list.requires_grad_(False)
            self.fixed_joint_centers = self.joint_centers.clone()

    def reset_skeleton(self):
        self.joint_centers = self.fixed_joint_centers.clone().requires_grad_(False)
        self.bone_length_list = self.fixed_bone_length_list.clone().requires_grad_(False)


    def update_skeleton_with_residuals(self, residuals):
        assert residuals.shape[0] == (self.joint_centers.shape[0] - 1)

        this_level = self.root.children
        while this_level:
            next_level = []
            for current_joint_idx in this_level:
                # get the children joints that needed to be updated as well
                update_list = []
                this_level_children = self.joint_list[current_joint_idx].children
                while this_level_children:
                    next_level_children = []
                    for current_joint_idx_children in this_level_children:
                        update_list.append(current_joint_idx_children)
                        next_level_children+=self.joint_list[current_joint_idx_children].children
                    this_level_children = next_level_children

                # get the correct location of current joint
                self.bone_length_list[current_joint_idx-1] += residuals[current_joint_idx-1]


                parent_idx = self.parents_mapping[current_joint_idx]
                parent_to_current = self.fixed_joint_centers[current_joint_idx,:] - self.fixed_joint_centers[parent_idx,:]

                length_ratio = self.bone_length_list[current_joint_idx-1] / self.joint_list[current_joint_idx].length

                self.joint_list[current_joint_idx].canonical_pose[:3,3] *= length_ratio
                current_joint_new_location = self.joint_centers[parent_idx,:] + length_ratio * parent_to_current
                current_children_offset = self.fixed_joint_centers[current_joint_idx,:] - current_joint_new_location
                self.joint_centers[current_joint_idx,:] = current_joint_new_location

                # update the children joints
                for update_idx in update_list:
                    self.joint_centers[update_idx, :] -= current_children_offset

                next_level+=self.joint_list[current_joint_idx].children
            this_level = next_level

    def traversal_dist(self):
        '''
        get a list of list where each row is the traversal distance from current joint to all of the joints in the kinematic chain
        '''
        # compute the traversal distance from every joint to every other joint
        distance_matrix = []
        for current_joint_idx in range(len(self.joint_list)):

            current_dist_matrix = torch.zeros((len(self.joint_list),1)).to(self.device)
            visited_list = [current_joint_idx]

            # handle current joint's children
            this_level_children = self.joint_list[current_joint_idx].children
            while this_level_children:
                next_level_children = []
                for current_joint_idx_children in this_level_children:
                    current_dist_matrix[current_joint_idx_children] = current_dist_matrix[self.joint_list[current_joint_idx_children].parent] + self.joint_list[current_joint_idx_children].length
                    next_level_children+=self.joint_list[current_joint_idx_children].children
                    visited_list.append(current_joint_idx_children)
                this_level_children = next_level_children

            # handle joints between root and current joint
            current_joint_idx_parent = current_joint_idx
            while current_joint_idx_parent != 0:
                next_level_children = []
                current_dist_matrix[self.joint_list[current_joint_idx_parent].parent] = current_dist_matrix[current_joint_idx_parent] + self.joint_list[current_joint_idx_parent].length
                visited_list.append(current_joint_idx_parent)
                current_joint_idx_sub_parent = current_joint_idx_parent
                root_starting_children = self.joint_list[current_joint_idx_sub_parent].children
                # print('root has',root_starting_children)
                while root_starting_children:
                    next_level_root_starting = []
                    for current_joint_idx_root_starting_children in root_starting_children:
                        if current_joint_idx_root_starting_children not in visited_list:
                            current_dist_matrix[current_joint_idx_root_starting_children] = current_dist_matrix[self.joint_list[current_joint_idx_root_starting_children].parent] + self.joint_list[current_joint_idx_root_starting_children].length
                            next_level_root_starting+=self.joint_list[current_joint_idx_root_starting_children].children
                            visited_list.append(current_joint_idx_root_starting_children)
                    root_starting_children = next_level_root_starting
                current_joint_idx_parent = self.joint_list[current_joint_idx_parent].parent

            # handle rest of the joints
            assert len(visited_list) <= len(self.joint_list)
            if len(visited_list) < len(self.joint_list):
                root_starting_children = self.joint_list[0].children

                while root_starting_children:
                    next_level_root_starting = []
                    for current_joint_idx_root_starting_children in root_starting_children:
                        if current_joint_idx_root_starting_children not in visited_list:
                            current_dist_matrix[current_joint_idx_root_starting_children] = current_dist_matrix[self.joint_list[current_joint_idx_root_starting_children].parent] + self.joint_list[current_joint_idx_root_starting_children].length
                            next_level_root_starting+=self.joint_list[current_joint_idx_root_starting_children].children
                            visited_list.append(current_joint_idx_root_starting_children)
                    root_starting_children = next_level_root_starting
            
            distance_matrix.append(current_dist_matrix)
        return distance_matrix

    def get_bones(self):
        """
        extract bones from skeleton struction
        :param skel: input skeleton
        :return: bones are B*6 array where each row consists starting and ending points of a bone
                bone_name are a list of B elements, where each element consists starting and ending joint name
                leaf_bones indicate if this bone is a virtual "leaf" bone.
                We add virtual "leaf" bones to the leaf joints since they always have skinning weights as well
        """
        bones = []
        bone_name = []
        # leaf_bones = []
        this_level = [0]
        while this_level:
            next_level = []
            for current_joint in this_level:
                p_pos = self.joint_centers[current_joint,:].detach().cpu().numpy()
                next_level += self.joint_list[current_joint].children
                for current_children_joint in self.joint_list[current_joint].children:
                    c_pos = self.joint_centers[current_children_joint].detach().cpu().numpy()
                    bones.append(np.concatenate((p_pos, c_pos))[np.newaxis, :])
                    bone_name.append([current_joint, current_children_joint])
            this_level = next_level
        bones = np.concatenate(bones, axis=0)
        return bones, bone_name

    def get_bones_torch(self):
        """
        extract bones from skeleton struction
        :param skel: input skeleton
        :return: bones are B*6 array where each row consists starting and ending points of a bone
                bone_name are a list of B elements, where each element consists starting and ending joint name
                leaf_bones indicate if this bone is a virtual "leaf" bone.
                We add virtual "leaf" bones to the leaf joints since they always have skinning weights as well
        """
        bones = []
        bone_name = []
        # leaf_bones = []
        this_level = [0]
        while this_level:
            next_level = []
            for current_joint in this_level:
                p_pos = self.joint_centers[current_joint,:]
                next_level += self.joint_list[current_joint].children
                for current_children_joint in self.joint_list[current_joint].children:
                    c_pos = self.joint_centers[current_children_joint]
                    bones.append(torch.cat((p_pos, c_pos))[None, :])
                    bone_name.append([current_joint, current_children_joint])
            this_level = next_level
        bones = torch.cat(bones, dim=0)
        return bones, bone_name

    def get_bone_children_parent_list(self):
        traversal_parent_child_list = []
        for joint_idx in range(len(self.joint_list)):
            child_list = []
            this_level = self.joint_list[joint_idx].children
            while this_level:
                next_level = []
                for current_joint in this_level:
                    child_list.append(current_joint)
                    next_level += self.joint_list[current_joint].children                   
                this_level = next_level
            parent_list = []
            for i in range(len(self.joint_list)):
                if i not in child_list:
                    parent_list.append(i)
            traversal_parent_child_list.append([parent_list, child_list])
        return traversal_parent_child_list

    def get_proper_transforms(self, deformed_joints, num_pts):
        '''
        given joints at unregulated locations, return the transforms needed for each joint and regulated 
        joint locations to satisfy kinematic chain constraints enforced
        '''
        if len(deformed_joints.shape) == 3:
            unregulated_joints = deformed_joints.squeeze(1)
        else:
            unregulated_joints = deformed_joints
        regulated_locations = self.enforce_kinematic_chains_constraints(unregulated_joints)
        translation = regulated_locations - self.joint_centers
        # print('translation',translation)
        identity_rotations = torch.eye(3).view(-1,9).repeat(self.joint_centers.shape[0],1).to(self.device)
        new_transforms = torch.cat([identity_rotations, translation],axis=-1).view(1,1,self.joint_centers.shape[0]*12)
        new_transforms = new_transforms.repeat(num_pts,1,1)
        return new_transforms, regulated_locations

    def enforce_kinematic_chains_constraints(self, deformed_joints):
        '''
        enforce kinematic chain constraints
        '''
        regulated_locations = self.regulate_lengths(deformed_joints)
        overall_offset = regulated_locations[0,:] - self.joint_centers[0,:]
        regulated_locations[1:,:] -= overall_offset
        regulated_locations[0,:] = self.joint_centers[0,:]

        return regulated_locations

    def regulate_lengths(self, deformed_skeleton):

        regulated_joint_centers = deformed_skeleton.clone()
        root_offset = regulated_joint_centers[0,:] - self.joint_centers[0,:]
        regulated_joint_centers -= root_offset

        this_level = self.root.children
        while this_level:
            next_level = []
            for current_joint_idx in this_level:
                # get the children joints that needed to be updated as well
                update_list = []
                this_level_children = self.joint_list[current_joint_idx].children
                while this_level_children:
                    next_level_children = []
                    for current_joint_idx_children in this_level_children:
                        update_list.append(current_joint_idx_children)
                        next_level_children+=self.joint_list[current_joint_idx_children].children
                    this_level_children = next_level_children

                # get the correct location of current joint
                parent_idx = self.parents_mapping[current_joint_idx]
                parent_to_current = deformed_skeleton[current_joint_idx,:] - regulated_joint_centers[parent_idx,:]

                if self.residual_update:
                    length_ratio = self.bone_length_list[current_joint_idx-1] / parent_to_current.pow(2).sum().sqrt()
                else:
                    length_ratio = self.joint_list[current_joint_idx].length / parent_to_current.pow(2).sum().sqrt()

                current_joint_new_location = regulated_joint_centers[parent_idx,:] + length_ratio * parent_to_current
                current_children_offset = deformed_skeleton[current_joint_idx,:] - current_joint_new_location
                regulated_joint_centers[current_joint_idx,:] = current_joint_new_location

                # update the children joints
                for update_idx in update_list:
                    regulated_joint_centers[update_idx, :] -= current_children_offset

                next_level+=self.joint_list[current_joint_idx].children
            this_level = next_level

        return regulated_joint_centers

    def get_proper_transforms_batch(self, deformed_joints, num_repeat):
        '''
        given joints at unregulated locations, return the transforms needed for each joint and regulated 
        joint locations to satisfy kinematic chain constraints enforced
        '''
        if len(deformed_joints.shape) == 3:
            unregulated_joints = deformed_joints.squeeze(1).view(-1,self.joint_centers.shape[0],3)
        else:
            unregulated_joints = deformed_joints.view(-1,self.joint_centers.shape[0],3)

        regulated_locations = self.enforce_kinematic_chains_constraints_batch(unregulated_joints, num_repeat)

        return regulated_locations

    def enforce_kinematic_chains_constraints_batch(self, deformed_joints, num_repeat):
        '''
        enforce kinematic chain constraints
        '''
        regulated_locations = self.regulate_lengths_batch(deformed_joints, num_repeat)
        overall_offset = regulated_locations[:,0,:] - self.joint_centers[0,:]

        regulated_locations[:,1:,:] -= overall_offset.view(regulated_locations.shape[0],1,3)
        regulated_locations[:,0,:] = self.joint_centers[0,:]

        return regulated_locations

    def regulate_lengths_batch(self, deformed_skeleton, num_repeat):

        regulated_joint_centers = deformed_skeleton.clone()
        root_offset = regulated_joint_centers[:,0,:] - self.joint_centers[0,:]
        regulated_joint_centers -= root_offset.view(regulated_joint_centers.shape[0],1,3)

        this_level = self.root.children
        while this_level:
            next_level = []
            for current_joint_idx in this_level:
                # get the children joints that needed to be updated as well
                update_list = []
                this_level_children = self.joint_list[current_joint_idx].children
                while this_level_children:
                    next_level_children = []
                    for current_joint_idx_children in this_level_children:
                        update_list.append(current_joint_idx_children)
                        next_level_children+=self.joint_list[current_joint_idx_children].children
                    this_level_children = next_level_children

                # get the correct location of current joint
                parent_idx = self.parents_mapping[current_joint_idx]
                parent_to_current = deformed_skeleton[:,current_joint_idx,:] - regulated_joint_centers[:,parent_idx,:]


                if self.residual_update:
                    length_ratio = self.bone_length_list[current_joint_idx-1] / torch.sum(parent_to_current.pow(2),dim=-1).sqrt()
                else:
                    length_ratio = self.joint_list[current_joint_idx].length / torch.sum(parent_to_current.pow(2),dim=-1).sqrt()


                current_joint_new_location = regulated_joint_centers[:,parent_idx,:] + (parent_to_current * length_ratio[:,None])

                current_children_offset = deformed_skeleton[:,current_joint_idx,:] - current_joint_new_location

                regulated_joint_centers[:,current_joint_idx,:] = current_joint_new_location

                # update the children joints
                for update_idx in update_list:
                    regulated_joint_centers[:,update_idx, :] -= current_children_offset

                next_level+=self.joint_list[current_joint_idx].children
            this_level = next_level

        return regulated_joint_centers

    def forward_kinematics(self, relative_transform):
        '''
        given a set of relative rotations for each bone in skeleton, infer the transform in global frame for each joint
        '''

        # assert joints_transform.shape[1] == len(self.joint_list)
        num_pts = relative_transform.shape[0]
        rts_fw = relative_transform.view(-1,len(self.joint_list),12)[0:1]

        rmat = rts_fw[:,:,:9]
        rmat = rmat.view(-1,len(self.joint_list),3,3).to(self.device)
        zero_tmat = torch.zeros(rts_fw[:,:,9:12].shape).to(self.device)

        rts_fw = torch.cat([rmat, zero_tmat[...,None]], -1)

        rts_fw = rts_fw.view(-1,len(self.joint_list),3,4)

        # root joint is assumed to in the same place
        identity_transform = torch.cat([torch.eye(3),torch.Tensor([0,0,0]).view(3,1)], axis=-1).to(self.device)


        rts_fw[:,0,:,:] = identity_transform

        this_level = self.root.children

        while this_level:
            next_level = []
            for current_joint_idx in this_level:
                parent_idx = self.parents_mapping[current_joint_idx]

                rts_fw[:,current_joint_idx,:,:] = rts_fw[:,parent_idx,:,:].clone().matmul(torch.cat([rts_fw[:,current_joint_idx,:,:].clone(),torch.Tensor([0,0,0,1]).view(1,1,4).to(self.device)],axis=1)\
                                                    .matmul((self.joint_list[current_joint_idx].canonical_pose)))
                next_level+=self.joint_list[current_joint_idx].children
            this_level = next_level
        

        rts_fw[:,:,:3,:3] = torch.eye(3).to(self.device)
        rvec = rts_fw[...,:3,:3].reshape(-1,9)
        tvec = rts_fw[...,:3,3].reshape(-1,3)

        tvec[1:,:] = tvec[1:,:] - self.joint_centers[1:,:] + self.joint_centers[0,:]
        rts_fw = torch.cat([rvec,tvec],-1).view(-1,len(self.joint_list)*12)

        rts_fw = rts_fw.repeat(num_pts,1)
        return rts_fw
