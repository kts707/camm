import torch
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import os
from math import pi,ceil

def image_grid(img, row, col):
    """
    img:     N,h,w,x
    collage: 1,.., x
    """
    bs,h,w,c=img.shape
    device = img.device
    collage = torch.zeros(h*row, w*col, c).to(device)
    for i in range(row):
        for j in range(col):
            collage[i*h:(i+1)*h,j*w:(j+1)*w] = img[i*col+j]
    return collage

def generate_sphere_vertices(r, rad=0.1/100, n=36):
	theta     = np.linspace(0, pi, ceil(n/2))
	phi       = np.linspace(-pi, pi, n)
	x0,y0,z0  = r
	[t,p]     = np.meshgrid(theta, phi)
	phi,theta = t.flatten(), p.flatten()
	x         = x0 + rad * np.sin(theta) * np.cos(phi)
	y         = y0 + rad * np.sin(theta) * np.sin(phi)
	z         = z0 + rad * np.cos(theta)
	return np.vstack( [x,y,z] ).T

def get_skeleton(joints, joint_connections):
    combined = []

    for j in range(joint_connections.shape[0]):
        
        a1 = joint_connections[j, 0]
        a2 = joint_connections[j, 1]
        A = joints[int(a1),:3]
        B = joints[int(a2),:3]
        vertsA = generate_sphere_vertices((A.detach().cpu()))
        vertsB = generate_sphere_vertices((B.detach().cpu()))
        vertsAB = np.vstack( [ vertsA , vertsB ] )
        pcAB    = trimesh.PointCloud(vertsAB)
        meshAB  = pcAB.convex_hull
        combined.append(meshAB)

    skel = trimesh.util.concatenate(combined)
    return skel

def get_skeleton_vis_v1(joints, joint_connections):
    combined = []
    combined2 = []
    elips = trimesh.creation.uv_sphere(radius=0.002,count=[16, 16])
    elips = trimesh.Trimesh(vertices=elips.vertices, faces=elips.faces)
    N_elips = len(elips.vertices)
    for j in range(joint_connections.shape[0]):
        
        a1 = joint_connections[j, 0]
        a2 = joint_connections[j, 1]
        A = joints[int(a1),:3]
        B = joints[int(a2),:3]
        vertsA = generate_sphere_vertices((A.detach().cpu()), rad=0.05/100)
        vertsB = generate_sphere_vertices((B.detach().cpu()), rad=0.05/100)
        vertsAB = np.vstack( [ vertsA , vertsB ] )
        pcAB    = trimesh.PointCloud(vertsAB)
        meshAB  = pcAB.convex_hull
        combined.append(meshAB)

    for j in range(joints.shape[0]):
        center1 = joints[j,:].detach().cpu().numpy()[None,:3]
        elips_verts1 = elips.vertices
        elips_verts1 = elips_verts1+center1

        joint1 = trimesh.Trimesh(vertices = elips_verts1, 
                                        faces=elips.faces)

        combined2.append(joint1)


    elips = trimesh.util.concatenate(combined2)
    colormap = np.zeros((joints.shape[0],3)) + 40
    colormap= np.tile(colormap[:,None], (1,N_elips,1)).reshape((-1,3))
    elips.visual.vertex_colors[:len(colormap),:3] = colormap

    skel = trimesh.util.concatenate(combined)
    # skel_colormap = np.zeros_like(skel.vertices) + 255
    # skel.visual.vertex_colors[:len(skel_colormap),:3] = skel_colormap
    
    all = trimesh.util.concatenate([elips, skel])
    return all

def get_skeleton_numpy(joints, joint_connections):
    combined = []

    for j in range(joint_connections.shape[0]):
        
        a1 = joint_connections[j, 0]
        a2 = joint_connections[j, 1]
        A = joints[int(a1),:3]
        B = joints[int(a2),:3]
        vertsA = generate_sphere_vertices((A))
        vertsB = generate_sphere_vertices((B))
        vertsAB = np.vstack( [ vertsA , vertsB ] )
        pcAB    = trimesh.PointCloud(vertsAB)
        meshAB  = pcAB.convex_hull
        combined.append(meshAB)

    skel = trimesh.util.concatenate(combined)
    return skel

def get_skeleton_numpy_vis_v1(joints, joint_connections):
    combined = []
    combined2 = []
    elips = trimesh.creation.uv_sphere(radius=0.002,count=[16, 16])
    elips = trimesh.Trimesh(vertices=elips.vertices, faces=elips.faces)
    N_elips = len(elips.vertices)
    for j in range(joint_connections.shape[0]):
        
        a1 = joint_connections[j, 0]
        a2 = joint_connections[j, 1]
        A = joints[int(a1),:3]
        B = joints[int(a2),:3]
        vertsA = generate_sphere_vertices((A), rad=0.05/100)
        vertsB = generate_sphere_vertices((B), rad=0.05/100)
        vertsAB = np.vstack( [ vertsA , vertsB ] )
        pcAB    = trimesh.PointCloud(vertsAB)
        meshAB  = pcAB.convex_hull
        combined.append(meshAB)

    for j in range(joints.shape[0]):
        center1 = joints[j,:][None,:3]
        elips_verts1 = elips.vertices
        elips_verts1 = elips_verts1+center1

        joint1 = trimesh.Trimesh(vertices = elips_verts1, 
                                        faces=elips.faces)

        combined2.append(joint1)


    elips = trimesh.util.concatenate(combined2)
    colormap = np.zeros((joints.shape[0],3)) + 40
    colormap= np.tile(colormap[:,None], (1,N_elips,1)).reshape((-1,3))
    elips.visual.vertex_colors[:len(colormap),:3] = colormap

    skel = trimesh.util.concatenate(combined)
    # skel_colormap = np.zeros_like(skel.vertices) + 255
    # skel.visual.vertex_colors[:len(skel_colormap),:3] = skel_colormap
    
    all = trimesh.util.concatenate([elips, skel])
    return all

def get_bone_skeleton_association(origins, ends):
    combined = []

    for j in range(origins.shape[0]):
        
        A = origins[j,:]
        B = ends[j,:]
        vertsA = generate_sphere_vertices((A),rad=0.3/500)
        vertsB = generate_sphere_vertices((B),rad=0.3/500)
        vertsAB = np.vstack( [ vertsA , vertsB ] )
        pcAB    = trimesh.PointCloud(vertsAB)
        meshAB  = pcAB.convex_hull
        combined.append(meshAB)

    result = trimesh.util.concatenate(combined)
    result.visual.vertex_colors = np.zeros((result.vertices.shape[0], 3)) + 255
    return result

def draw_skeleton_2d_numpy(joints, joint_connections, path, title=None):
    # xy plane
    joints_2d = joints[:,:2]
    plt.figure()
    for j in range(joint_connections.shape[0]):
        plt.text(joints_2d[j, 0], joints_2d[j, 1], j, fontsize="small")
        a1 = joint_connections[j, 0]
        a2 = joint_connections[j, 1]
        # print([a1, a2])
        plt.plot(joints_2d[[a1, a2], 0], joints_2d[[a1, a2], 1])
    plt.text(joints_2d[j+1, 0], joints_2d[j+1, 1], j+1, fontsize="small")
    if title is None:
        plt.title("connected joints in 2d xy-plane")
    else:
        plt.title(title+' xy-plane')
    plt.axis("off")
    plt.savefig(path[:-4]+'_xy'+path[-4:], dpi=300, facecolor='white',transparent=True)

    # xz plane
    joints_2d = joints[:,[0,2]]
    plt.figure()
    for j in range(joint_connections.shape[0]):
        plt.text(joints_2d[j, 0], joints_2d[j, 1], j, fontsize="small")
        a1 = joint_connections[j, 0]
        a2 = joint_connections[j, 1]
        # print([a1, a2])
        plt.plot(joints_2d[[a1, a2], 0], joints_2d[[a1, a2], 1])
    plt.text(joints_2d[j+1, 0], joints_2d[j+1, 1], j+1, fontsize="small")
    if title is None:
        plt.title("connected joints in 2d xz-plane")
    else:
        plt.title(title+' xz-plane')
    plt.axis("off")
    plt.savefig(path[:-4]+'_xz'+path[-4:], dpi=300, facecolor='white',transparent=True)

    # yz plane
    joints_2d = joints[:,1:]
    plt.figure()
    for j in range(joint_connections.shape[0]):
        plt.text(joints_2d[j, 0], joints_2d[j, 1], j, fontsize="small")
        a1 = joint_connections[j, 0]
        a2 = joint_connections[j, 1]
        # print([a1, a2])
        plt.plot(joints_2d[[a1, a2], 0], joints_2d[[a1, a2], 1])
    plt.text(joints_2d[j+1, 0], joints_2d[j+1, 1], j+1, fontsize="small")
    if title is None:
        plt.title("connected joints in 2d yz-plane")
    else:
        plt.title(title+' yz-plane')
    plt.axis("off")
    plt.savefig(path[:-4]+'_yz'+path[-4:], dpi=300, facecolor='white',transparent=True)

def draw_skeleton_2d(joints, joint_connections, path, title=None):
    # xy plane
    joints_2d = joints[:,:2]
    plt.figure()
    for j in range(joint_connections.shape[0]):
        plt.text(joints_2d[j, 0], joints_2d[j, 1], j, fontsize="small")
        a1 = joint_connections[j, 0]
        a2 = joint_connections[j, 1]
        # print([a1, a2])
        plt.plot(joints_2d[[a1, a2], 0].detach().cpu().numpy(), joints_2d[[a1, a2], 1].detach().cpu().numpy())
    plt.text(joints_2d[j+1, 0], joints_2d[j+1, 1], j+1, fontsize="small")
    if title is None:
        plt.title("connected joints in 2d xy-plane")
    else:
        plt.title(title+' xy-plane')
    plt.axis("off")
    plt.savefig(path[:-4]+'_xy'+path[-4:], dpi=300, facecolor='white',transparent=True)

    # xz plane
    xz_indices = torch.tensor([0, 2]).to(joints.device)
    joints_2d = torch.index_select(joints, 1, xz_indices)
    plt.figure()
    for j in range(joint_connections.shape[0]):
        plt.text(joints_2d[j, 0], joints_2d[j, 1], j, fontsize="small")
        a1 = joint_connections[j, 0]
        a2 = joint_connections[j, 1]
        # print([a1, a2])
        plt.plot(joints_2d[[a1, a2], 0].detach().cpu().numpy(), joints_2d[[a1, a2], 1].detach().cpu().numpy())
    plt.text(joints_2d[j+1, 0], joints_2d[j+1, 1], j+1, fontsize="small")
    if title is None:
        plt.title("connected joints in 2d xz-plane")
    else:
        plt.title(title+' xz-plane')
    plt.axis("off")
    plt.savefig(path[:-4]+'_xz'+path[-4:], dpi=300, facecolor='white',transparent=True)

    # yz plane
    joints_2d = joints[:,1:]
    plt.figure()
    for j in range(joint_connections.shape[0]):
        plt.text(joints_2d[j, 0], joints_2d[j, 1], j, fontsize="small")
        a1 = joint_connections[j, 0]
        a2 = joint_connections[j, 1]
        # print([a1, a2])
        plt.plot(joints_2d[[a1, a2], 0].detach().cpu().numpy(), joints_2d[[a1, a2], 1].detach().cpu().numpy())
    plt.text(joints_2d[j+1, 0], joints_2d[j+1, 1], j+1, fontsize="small")
    if title is None:
        plt.title("connected joints in 2d yz-plane")
    else:
        plt.title(title+' yz-plane')
    plt.axis("off")
    plt.savefig(path[:-4]+'_yz'+path[-4:], dpi=300, facecolor='white',transparent=True)

def paint_joint_skinning(canonical_mesh, skinning_weights, joints=None, joint_idx=None, saving_path=None, red_only=False):
    red_foreground_color = np.ones((canonical_mesh.vertices.shape[0], 3)) * [255, 0, 0]
    green_background_color = np.ones((canonical_mesh.vertices.shape[0], 3)) * [0, 255, 0]

    if len(skinning_weights.shape) == 3:
        skinning = skinning_weights.squeeze(1)
    else:
        skinning = skinning_weights
    if joint_idx is not None:
        if red_only:
            colors = skinning[:,joint_idx:joint_idx+1] * red_foreground_color
        else:
            colors = skinning[:,joint_idx:joint_idx+1] * red_foreground_color + (1 - skinning[:,joint_idx:joint_idx+1]) * green_background_color
        canonical_mesh.visual.vertex_colors = colors
        if saving_path:
            canonical_mesh.export(saving_path)
        else:
            canonical_mesh.export('skinning_joint_'+str(joint_idx)+'.obj')
    else:
        # save painting for all joints
        assert joints is not None
        for i in range(joints.shape[0]):
            if red_only:
                colors = skinning[:,i:i+1] * red_foreground_color
            else:
                colors = skinning[:,i:i+1] * red_foreground_color + (1 - skinning[:,i:i+1]) * green_background_color
            canonical_mesh.visual.vertex_colors = colors
            if saving_path:
                os.makedirs(saving_path, exist_ok=True) 
                canonical_mesh.export(os.path.join(saving_path,'skinning_joint_'+str(i)+'.obj'))
            else:
                canonical_mesh.export('skinning_joint_'+str(i)+'.obj')
