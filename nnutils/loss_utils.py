import pdb
import trimesh
import cv2
import numpy as np
import torch
from nnutils.geom_utils import rot_angle, mat2K, Kmatinv, obj_to_cam, \
                                pinhole_cam, lbs, gauss_mlp_skinning, evaluate_mlp
import torch.nn.functional as F

def nerf_gradient(mlp, embed, pts, use_xyz=False,code=None, sigma_only=False):
    """
    gradient of mlp params wrt pts
    """
    pts.requires_grad_(True)
    pts_embedded = embed(pts)
    if use_xyz: xyz=pts
    else: xyz=None
    y = evaluate_mlp(mlp, pts_embedded, chunk=pts.shape[0], 
            xyz=xyz,code=code,sigma_only=sigma_only)
    
    sdf = -y
    ibetas = 1/(mlp.beta.abs()+1e-9)
    sigmas = (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() * ibetas))
        
    # get gradient for each size-1 output
    gradients = []
    for i in range(y.shape[-1]):
        y_sub = y [...,i:i+1]
        d_output = torch.ones_like(y_sub, requires_grad=False, device=y.device)
        gradient = torch.autograd.grad(
            outputs=y_sub,
            inputs=pts,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        gradients.append( gradient[...,None] )
    gradients = torch.cat(gradients,-1) # ...,input-dim, output-dim
    return gradients, sigmas

def eikonal_loss(mlp, embed, pts_exp, bound):
    """
    pts_exp: X* backward warped points
    """
    pts_exp = pts_exp.view(1,-1,3).detach()
    nsample = pts_exp.shape[1]
    device = next(mlp.parameters()).device

    # Sample points for the eikonal loss
    bound = torch.Tensor(bound)[None,None]
    pts = torch.rand(1,nsample,3)*2*bound-bound
    pts= pts.to(device)
    pts = torch.cat([pts,pts_exp],1)

    g,sigmas_unit = nerf_gradient(mlp, embed, pts, sigma_only=True)
    g = g[...,0]
    #sigmas_unit = sigmas_unit[...,0].detach()
    sigmas_unit = ((pts.abs() < bound.to(device)).float().sum(-1)==3).float()

    #need to weight by occupancy score
    eikonal_loss = (g.norm(2, dim=-1) - 1) ** 2
    eikonal_loss = (sigmas_unit*eikonal_loss).sum() / sigmas_unit.sum()
    return eikonal_loss

def elastic_loss(mlp, embed, xyz, time_embedded):
    xyz = xyz.detach().clone()
    time_embedded = time_embedded.detach().clone()
    g,_ = nerf_gradient(mlp, embed, xyz, use_xyz=mlp.use_xyz,code=time_embedded)
    jacobian = g+torch.eye(3)[None,None].to(g.device)

    sign, log_svals = jacobian.slogdet()
    log_svals = log_svals.clone()
    log_svals[sign<=0] = 0.
    elastic_loss = log_svals**2
    return elastic_loss
    

def bone_density_loss(mlp, embed, bones):
    pts = bones[:,:3] 
    pts_embedded = embed(pts)
    y = evaluate_mlp(mlp, pts_embedded, pts.shape[0], sigma_only=True)
    return bone_density_loss

def visibility_loss(mlp, embed, xyz_pos, w_pos, bound, chunk):
    """
    w_pos: num_points x num_samples, visibility returns from nerf
    bound: scalar, used to sample negative samples
    """
    device = next(mlp.parameters()).device
    xyz_pos = xyz_pos.detach().clone()
    w_pos = w_pos.detach().clone()
    
    # negative examples
    nsample = w_pos.shape[0]*w_pos.shape[1]
    bound = torch.Tensor(bound)[None,None]
    xyz_neg = torch.rand(1,nsample,3)*2*bound-bound
    xyz_neg = xyz_neg.to(device)
    xyz_neg_embedded = embed(xyz_neg)
    vis_neg_pred = evaluate_mlp(mlp, xyz_neg_embedded, chunk=chunk)[...,0]
    vis_loss_neg = -F.logsigmoid(-vis_neg_pred).sum()*0.1/nsample
      
    # positive examples
    xyz_pos_embedded = embed(xyz_pos)
    vis_pos_pred = evaluate_mlp(mlp, xyz_pos_embedded, chunk=chunk)[...,0]
    vis_loss_pos = -(F.logsigmoid(vis_pos_pred) * w_pos).sum()/nsample

    vis_loss = vis_loss_pos + vis_loss_neg
    return vis_loss

def rtk_loss(rtk, rtk_raw, aux_out):
    rot_pred = rtk[:,:3,:3]
    rot_gt = rtk_raw[:,:3,:3]
    rot_loss = rot_angle(rot_pred.matmul(rot_gt.permute(0,2,1))).mean()
    rot_loss = 0.01*rot_loss

    trn_pred = rtk[:,:3,3]
    trn_gt = rtk_raw[:,:3,3]
    trn_loss = (trn_pred - trn_gt).pow(2).sum(-1).mean()
    total_loss = rot_loss + trn_loss
    aux_out['rot_loss'] = rot_loss
    aux_out['trn_loss'] = trn_loss
    return total_loss

def compute_pts_exp(pts_prob, pts):
    """
    pts:      ..., ndepth, 3
    pts_prob: ..., ndepth
    """
    ndepth = pts_prob.shape[-1]
    pts_prob = pts_prob.clone()
    pts_prob = pts_prob.view(-1, ndepth,1)
    pts_prob = pts_prob/(1e-9+pts_prob.sum(1)[:,None])
    pts_exp = (pts * pts_prob).sum(1)
    return pts_exp

def feat_match_loss(nerf_feat, embedding_xyz, feats, pts, pts_prob, bound, 
        is_training=True):
    """
    feats:    ..., num_feat
    pts:      ..., ndepth, 3
    pts_prob: ..., ndepth
    loss:     ..., 1
    """
    pts = pts.clone()

    base_shape = feats.shape[:-1] # bs, ns
    nfeat =     feats.shape[-1]
    ndepth = pts_prob.shape[-1]
    feats=        feats.view(-1, nfeat)
    pts =           pts.view(-1, ndepth,3)
    
    # part1: compute expected pts
    pts_exp = compute_pts_exp(pts_prob, pts)

    ## part2: matching
    pts_pred = feat_match(nerf_feat, embedding_xyz, feats, 
            bound,grid_size=20,is_training=is_training)

    # part3: compute loss
    feat_err = (pts_pred - pts_exp).norm(2,-1) # n,ndepth

    # rearrange outputs
    pts_pred  = pts_pred.view(base_shape+(3,))
    pts_exp   = pts_exp .view(base_shape+(3,))
    feat_err = feat_err .view(base_shape+(1,))
    return pts_pred, pts_exp, feat_err

def kp_reproj_loss(pts_pred, xys, models, embedding_xyz, rays, use_hs=False, joints_only=False):
    """
    pts_pred,   ...,3
    xys,        ...,2
    out,        ...,1 same as pts_pred
    gcc loss is only used to update root/body pose and skinning weights
    """
    xys = xys.view(-1,1,2)
    xy_reproj = kp_reproj(pts_pred, models, embedding_xyz, rays, use_hs=use_hs, joints_only=joints_only) 
    proj_err = (xys - xy_reproj[...,:2]).norm(2,-1)
    proj_err = proj_err.view(pts_pred.shape[:-1]+(1,))
    return proj_err

def kp_reproj(pts_pred, models, embedding_xyz, rays, use_hs=False, to_target=False, joints_only=False):
    """
    pts_pred,   ...,3
    out,        ...,1,3 same as pts_pred
    to_target   whether reproject to target frame
    """
    N = pts_pred.view(-1,3).shape[0]
    xyz_coarse_sampled = pts_pred.view(-1,1,3)
    # detach grad since reproj-loss would not benefit feature learning 
    # (due to ambiguity)
    #xyz_coarse_sampled = xyz_coarse_sampled.detach() 

    # TODO wrap flowbw and lbs into the same module
    # TODO include loss for flowbw
    if to_target:  rtk_vec = rays['rtk_vec_target']
    else:          rtk_vec = rays['rtk_vec']
    rtk_vec = rtk_vec.view(N,-1) # bs, ns, 21
    if 'bones' in models.keys():
        if to_target:    bone_rts_fw = rays['bone_rts_target']
        else:            bone_rts_fw = rays['bone_rts']
        bone_rts_fw = bone_rts_fw.view(N,-1) # bs, ns,-1
        if 'nerf_skin' in models.keys():
            nerf_skin = models['nerf_skin']
        else: nerf_skin = None
        bones = models['bones_rst']
        skin_aux = models['skin_aux']
        rest_pose_code = models['rest_pose_code']
        rest_pose_code = rest_pose_code(torch.Tensor([0]).long().to(bones.device))
        skin_forward = gauss_mlp_skinning(xyz_coarse_sampled, embedding_xyz, bones,
                                  rest_pose_code, nerf_skin, use_hs=use_hs, skin_aux=skin_aux, joints_only=joints_only)
        
        xyz_coarse_sampled,_ = lbs(bones, bone_rts_fw,
                          skin_forward, xyz_coarse_sampled, backward=False)

    Rmat = rtk_vec[:,0:9]  .view(N,1,3,3)
    Tmat = rtk_vec[:,9:12] .view(N,1,3)
    Kinv = rtk_vec[:,12:21].view(N,1,3,3)
    K = mat2K(Kmatinv(Kinv))

    xyz_coarse_sampled = obj_to_cam( xyz_coarse_sampled, Rmat, Tmat) 
    xyz_coarse_sampled = pinhole_cam(xyz_coarse_sampled,K)

    xy_coarse_sampled = xyz_coarse_sampled[...,:2]
    return xy_coarse_sampled
    
    
def feat_match(nerf_feat, embedding_xyz, feats, bound, 
        grid_size=20,is_training=True, init_pts=None, rt_entropy=False):
    """
    feats:    -1, num_feat
    """
    if is_training: 
        chunk_pts = 8*1024
    else:
        chunk_pts = 1024
    chunk_pix = 4096
    nsample,_ = feats.shape
    device = feats.device
    feats = F.normalize(feats,2,-1)
    
    # sample model on a regular 3d grid, and correlate with feature, nkxkxk
    #p1d = np.linspace(-bound, bound, grid_size).astype(np.float32)
    #query_yxz = np.stack(np.meshgrid(p1d, p1d, p1d), -1)  # (y,x,z)
    pxd = np.linspace(-bound[0], bound[0], grid_size).astype(np.float32)
    pyd = np.linspace(-bound[1], bound[1], grid_size).astype(np.float32)
    pzd = np.linspace(-bound[2], bound[2], grid_size).astype(np.float32)
    query_yxz = np.stack(np.meshgrid(pyd, pxd, pzd), -1)  # (y,x,z)
    query_yxz = torch.Tensor(query_yxz).to(device).view(-1, 3)
    query_xyz = torch.cat([query_yxz[:,1:2], query_yxz[:,0:1], query_yxz[:,2:3]],-1)

    if init_pts is not None:
        query_xyz = query_xyz[None] + init_pts[:,None]
    else:
        # N x Ns x 3
        query_xyz = query_xyz[None]

    # inject some noise at training time
    if is_training and init_pts is None:
        bound = torch.Tensor(bound)[None,None].to(device)
        query_xyz = query_xyz + torch.randn_like(query_xyz) * bound * 0.05

    cost_vol = []
    for i in range(0,grid_size**3,chunk_pts):
        if init_pts is None:
            query_xyz_chunk = query_xyz[0,i:i+chunk_pts]
            xyz_embedded = embedding_xyz(query_xyz_chunk)[:,None] # (N,1,...)
            vol_feat_subchunk = evaluate_mlp(nerf_feat, xyz_embedded)[:,0] # (chunk, num_feat)
            # normalize vol feat
            vol_feat_subchunk = F.normalize(vol_feat_subchunk,2,-1)[None]

        cost_chunk = []
        for j in range(0,nsample,chunk_pix):
            feats_chunk = feats[j:j+chunk_pix] # (chunk pix, num_feat)
     
            if init_pts is not None:
                # only query 3d grid according to each px when they are diff
                # vol feature
                query_xyz_chunk = query_xyz[j:j+chunk_pix,i:i+chunk_pts].clone()
                xyz_embedded = embedding_xyz(query_xyz_chunk)
                vol_feat_subchunk = evaluate_mlp(nerf_feat, xyz_embedded)
                # normalize vol feat
                vol_feat_subchunk = F.normalize(vol_feat_subchunk,2,-1)

            # cpix, cpts
            # distance metric
            cost_subchunk = (vol_feat_subchunk * \
                    feats_chunk[:,None]).sum(-1) * (nerf_feat.beta.abs()+1e-9)
            cost_chunk.append(cost_subchunk)
        cost_chunk = torch.cat(cost_chunk,0) # (nsample, cpts)
        cost_vol.append(cost_chunk)
    cost_vol = torch.cat(cost_vol,-1) # (nsample, k**3)
    prob_vol = cost_vol.softmax(-1)

    # regress to the true location, n,3
    if not is_training: torch.cuda.empty_cache()
    # n, ns, 1 * n, ns, 3
    pts_pred = (prob_vol[...,None] * query_xyz).sum(1)

    if rt_entropy:
        # compute normalized entropy
        match_unc = (-prob_vol * prob_vol.clamp(1e-9,1-1e-9).log()).sum(1)[:,None]
        match_unc = match_unc/np.log(grid_size**3)
        return pts_pred, match_unc
    else:
        return pts_pred

def grad_update_bone(bones,embedding_xyz, nerf_vis, learning_rate):
    """
    #TODO need to update bones locally
    """
    device = bones.device
    bones_data = bones.data.detach()
    bones_data.requires_grad_(True)
    bone_xyz_embed = embedding_xyz(bones_data[:,None,:3])
    sdf_at_bone = evaluate_mlp(nerf_vis, bone_xyz_embed)
    bone_loc_loss = F.relu(-sdf_at_bone).mean()
    
    # compute gradient wrt bones
    d_output = torch.ones_like(bone_loc_loss, requires_grad=False, device=device)
    gradient = torch.autograd.grad(
        outputs=bone_loc_loss,
        inputs=bones_data,
        grad_outputs=d_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    bones.data = bones.data-gradient*learning_rate

    return bone_loc_loss

def bone_separation_loss_3d(bones_rst_xyz, num_bones, sigma=0.05):
    distance = torch.sum((bones_rst_xyz[:, :, None] - bones_rst_xyz[:, None]) ** 2, dim=-1)
    exp_dist = torch.exp(-distance / (2 * sigma ** 2))  # (B, num_bones, num_bones)
    exp_dist = exp_dist * (1 - torch.eye(num_bones, device=bones_rst_xyz.device))
    loss = exp_dist.sum() / (num_bones - 1)
    return loss

def surface_2d_loss(surface_points_2d, gt_mask, pixel_radius_threshold= 3):
    """surface distance in image
    Args:
        surface_points_2d (torch.Tensor):   # (B, num_parts, 2, num_points)
        gt_mask (torch.Tensor): # (B, size, size)
        pixel_radius_threshold (float): Ignore distances smaller than this.
    Returns:
        torch.Tensor: [description]
    """
    device = surface_points_2d.device
    gt_mask = torch.from_numpy(gt_mask).to(device)
    batchsize, img_size, _ = gt_mask.shape
    valid_mask = (gt_mask.sum(dim=(1, 2)) > 1).float()

    def _foreground_sampler(img_size, num_ray, mask):
        """uniformly sample from foreground mask
        Args:
            img_size (int): image size
            num_ray (int): number of points to sample
            mask (int): shape: (B, img_size, img_size)
        Returns:
            coordinates: sampled coordinates, shape: (B, num_ray, 2)
            mask_value: mask value at sampled location
        """
        mask = mask.reshape(-1, img_size ** 2)
        noised_mask = (mask > 0.5) + torch.empty_like(mask, dtype=torch.float).uniform_(0, 0.1)
        _, coordinates = torch.topk(noised_mask, num_ray, dim=1, sorted=False)
        mask_value = torch.gather(mask, dim=1, index=coordinates)
        coordinates = torch.stack([coordinates % img_size,
                                    torch.div(coordinates, img_size, rounding_mode='trunc')], dim=-1) + 0.5

        return coordinates, mask_value

    surface_points_2d = surface_points_2d.permute(0, 1, 3, 2).reshape(batchsize, -1, 2)
    # chamfer
    with torch.no_grad():
        # randomly sample foreground pixels
        grid, gt_mask = _foreground_sampler(img_size, img_size ** 2 // 16, gt_mask)
        reliable_gt_mask = gt_mask == 1
        reliable_gt_background = gt_mask == 0

        # distance
        torch.cuda.empty_cache()
        distance = torch.sum((grid[:, :, None].half() - surface_points_2d[:, None].half()) ** 2, dim=-1)
        distance = distance + reliable_gt_background[:, :, None] * 1e10  # mask outside

        # nearest gt mask
        nearest_gt_mask_idx = torch.argmin(distance, dim=1)  # (B, num_parts * num_points)

        # nearest surface points
        nearest_surface_idx = torch.argmin(distance, dim=2)  # (B, num_grid_points)

    # surface -> mask
    nearest_gt_mask = torch.gather(grid.expand(batchsize, -1, 2), dim=1,
                                    index=nearest_gt_mask_idx[:, :, None].expand(-1, -1, 2))
    nearest_gt_mask_reliability = torch.gather(gt_mask, dim=1, index=nearest_gt_mask_idx) == 1
    surface_to_mask = torch.sum((surface_points_2d - nearest_gt_mask) ** 2, dim=-1) * valid_mask[:, None]
    surface_to_mask = surface_to_mask * nearest_gt_mask_reliability
    surface_to_mask = surface_to_mask.masked_fill(surface_to_mask < pixel_radius_threshold ** 2, 0)
    loss = surface_to_mask.mean()

    # mask -> surface
    nearest_surface = torch.gather(surface_points_2d, dim=1,
                                    index=nearest_surface_idx[:, :, None].expand(-1, -1, 2))
    mask_to_surface = torch.sum((grid - nearest_surface) ** 2, dim=-1)
    mask_to_surface = mask_to_surface.masked_fill(mask_to_surface < pixel_radius_threshold ** 2, 0)
    loss += torch.sum(mask_to_surface * reliable_gt_mask) / (reliable_gt_mask.sum() + 1e-8)

    return loss / img_size ** 2

def grid_coordinates(size, device, scale=2):

    grid = torch.meshgrid(torch.arange(size, device=device),
                          torch.arange(size, device=device), indexing='ij')[::-1]
    grid = torch.stack(grid, dim=-1) * scale + 0.5
    grid = grid.reshape(1, size ** 2, 2)
    return grid

def sdf_bone_loss(bones,embedding_xyz, nerf_vis, use_embed_bones_all):
    """
    #TODO need to update bones locally
    """
    #device = bones.device
    # bones_data = bones.data#.detach()
    #bones_data.requires_grad_(True)
    if use_embed_bones_all:
        bone_xyz_embed = embedding_xyz(bones[:,None,:]) #Pass all the bones params insead of centres
    else:
        bone_xyz_embed = embedding_xyz(bones[:,None,:3])
    sdf_at_bone = evaluate_mlp(nerf_vis, bone_xyz_embed).sigmoid()
    bone_loc_loss = F.relu(-sdf_at_bone).mean()
    
    # compute gradient wrt bones
    # d_output = torch.ones_like(bone_loc_loss, requires_grad=False, device=device)
    # gradient = torch.autograd.grad(
    #     outputs=bone_loc_loss,
    #     inputs=bones_data,
    #     grad_outputs=d_output,
    #     create_graph=True,
    #     retain_graph=True,
    #     only_inputs=True)[0]

    # bones.data = bones.data-gradient*learning_rate

    return bone_loc_loss

def loss_filter_line(sil_err, errid, frameid, sil_loss_samp, img_size, scale_factor=10):
    """
    sil_err: Tx512
    errid: N
    """
    sil_loss_samp = sil_loss_samp.detach().cpu().numpy().reshape(-1)
    sil_err[errid] = sil_loss_samp
    sil_err = sil_err.reshape(-1,img_size)
    sil_err = sil_err.sum(-1) / (1e-9+(sil_err>0).astype(float).sum(-1))
    sil_err_med = np.median(sil_err[sil_err>0])
    invalid_frame = sil_err > sil_err_med*scale_factor
    invalid_idx = invalid_frame[frameid]
    sil_err[:] = 0
    return invalid_idx

def loss_filter(g_floerr, flo_loss_samp, sil_at_samp_flo, scale_factor=10):
    """
    g_floerr:       T,
    flo_loss_samp:  bs,N,1
    sil_at_samp_flo:bs,N,1
    """
    bs = sil_at_samp_flo.shape[0] 
    # find history meidan
    g_floerr = g_floerr[g_floerr>0]

    # tb updated as history value
    #flo_err = []
    #for i in range(bs):
    #    flo_err_sub =flo_loss_samp[i][sil_at_samp_flo[i]]
    #    if len(flo_err_sub) >0:
    #        #flo_err_sub = flo_err_sub.median().detach().cpu().numpy()
    #        flo_err_sub = flo_err_sub.mean().detach().cpu().numpy()
    #    else: 
    #        flo_err_sub = 0
    #    flo_err.append(flo_err_sub)
    #flo_err = np.stack(flo_err)
    
    # vectorized version but uses mean to update
    flo_err = (flo_loss_samp * sil_at_samp_flo).sum(1) /\
              (1e-9+sil_at_samp_flo.sum(1)) # bs, N, 1
    flo_err = flo_err.detach().cpu().numpy()[...,0]

    # find invalid idx
    invalid_idx = flo_err > np.median(g_floerr)*scale_factor
    return flo_err, invalid_idx


def compute_xyz_wt_loss(gt_list, curr_list):
    loss = []
    for i in range(len(gt_list)):
        loss.append( (gt_list[i].detach() - curr_list[i]).pow(2).mean() )
    loss = torch.stack(loss).mean()
    return loss

def compute_root_sm_2nd_loss(rtk_all, data_offset):
    """
    2nd order loss
    """
    rot_sm_loss = []
    trn_sm_loss = []
    for didx in range(len(data_offset)-1):
        stt_idx = data_offset[didx]
        end_idx = data_offset[didx+1]

        stt_rtk = rtk_all[stt_idx:end_idx-2]
        mid_rtk = rtk_all[stt_idx+1:end_idx-1]
        end_rtk = rtk_all[stt_idx+2:end_idx]

        rot_sub1 = stt_rtk[:,:3,:3].matmul(mid_rtk[:,:3,:3].permute(0,2,1))
        rot_sub2 = mid_rtk[:,:3,:3].matmul(end_rtk[:,:3,:3].permute(0,2,1))

        trn_sub1 = stt_rtk[:,:3,3] - mid_rtk[:,:3,3]
        trn_sub2 = mid_rtk[:,:3,3] - end_rtk[:,:3,3]

        rot_sm_sub = rot_sub1.matmul(rot_sub2.permute(0,2,1))
        trn_sm_sub = trn_sub1 - trn_sub2
        
        rot_sm_loss.append(rot_sm_sub)
        trn_sm_loss.append(trn_sm_sub)
    rot_sm_loss = torch.cat(rot_sm_loss,0)
    rot_sm_loss = rot_angle(rot_sm_loss).mean()*1e-1
    trn_sm_loss = torch.cat(trn_sm_loss,0)
    trn_sm_loss = trn_sm_loss.norm(2,-1).mean()
    root_sm_loss = rot_sm_loss + trn_sm_loss 
    root_sm_loss = root_sm_loss * 0.1
    return root_sm_loss


def compute_root_sm_loss(rtk_all, data_offset):
    rot_sm_loss = []
    trans_sm_loss = []
    for didx in range(len(data_offset)-1):
        stt_idx = data_offset[didx]
        end_idx = data_offset[didx+1]
        rot_sm_sub = rtk_all[stt_idx:end_idx-1,:3,:3].matmul(
                      rtk_all[stt_idx+1:end_idx,:3,:3].permute(0,2,1))
        trans_sm_sub =  rtk_all[stt_idx:end_idx-1,:3,3] - \
                        rtk_all[stt_idx+1:end_idx,:3,3]
        rot_sm_loss.append(rot_sm_sub)
        trans_sm_loss.append(trans_sm_sub)
    rot_sm_loss = torch.cat(rot_sm_loss,0)
    rot_sm_loss = rot_angle(rot_sm_loss).mean()*1e-3
    trans_sm_loss = torch.cat(trans_sm_loss,0)
    trans_sm_loss = trans_sm_loss.norm(2,-1).mean()*0.1
    root_sm_loss = rot_sm_loss + trans_sm_loss 
    return root_sm_loss


def shape_init_loss(pts, faces,  mlp, embed, bound_factor, use_ellips=True):
    # compute sdf loss wrt to a mesh
    # construct mesh
    mesh = trimesh.Trimesh(pts.cpu(), faces=faces.cpu())
    device = next(mlp.parameters()).device

    # Sample points
    nsample =10000
    obj_bound = pts.abs().max(0)[0][None,None]
    bound = obj_bound * bound_factor
    pts_samp = torch.rand(1,nsample,3).to(device)*2*bound-bound

    # outside: positive
    if use_ellips:
        # signed distance to a ellipsoid
        dis = (pts_samp/obj_bound).pow(2).sum(2).view(-1)
        dis = torch.sqrt(dis)
        dis = dis  - 1 
        dis = dis * obj_bound.mean()
    else:
        # signed distance to a sphere
        dis = (pts_samp).pow(2).sum(2).view(-1)
        dis = torch.sqrt(dis)
        dis = dis  - obj_bound.min()

    # compute sdf
    pts_embedded = embed(pts_samp)
    y = evaluate_mlp(mlp, pts_embedded, chunk=pts_samp.shape[0], 
            xyz=None,code=None,sigma_only=True)
    
    sdf = -y.view(-1) # positive: outside
    shape_loss = (sdf - dis).pow(2).mean()
    return shape_loss
    
def bone_init_loss(pts,  mlp, embed, bound_factor, use_ellips=True):
    # compute sdf loss wrt to a mesh
    # construct mesh
    # mesh = trimesh.Trimesh(pts.cpu(), faces=faces.cpu())
    device = next(mlp.parameters()).device

    # Sample points
    nsample =10000
    obj_bound = pts.abs().max(0)[0][None,None]
    bound = obj_bound * bound_factor
    pts_samp = torch.rand(1,nsample,3).to(device)*2*bound-bound

    # outside: positive
    if use_ellips:
        # signed distance to a ellipsoid
        dis = (pts_samp/obj_bound).pow(2).sum(2).view(-1)
        dis = torch.sqrt(dis)
        dis = dis  - 1 
        dis = dis * obj_bound.mean()
    else:
        # signed distance to a sphere
        dis = (pts_samp).pow(2).sum(2).view(-1)
        dis = torch.sqrt(dis)
        dis = dis  - obj_bound.min()

    # compute sdf
    pts_embedded = embed(pts_samp)
    y = evaluate_mlp(mlp, pts_embedded, chunk=pts_samp.shape[0], 
            xyz=None,code=None,sigma_only=True)
    
    sdf = -y.view(-1) # positive: outside
    shape_loss = (sdf - dis).pow(2).mean()
    return shape_loss
