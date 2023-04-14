import sys, os
import pdb
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
os.environ["PYOPENGL_PLATFORM"] = "egl"
curr_dir = os.path.abspath(os.getcwd())
sys.path.insert(0,curr_dir)

import glob
from utils.io import save_vid
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import argparse
import trimesh
from nnutils.geom_utils import obj_to_cam, pinhole_cam, obj2cam_np
import pyrender
from pyrender import IntrinsicsCamera,Mesh, Node, Scene,OffscreenRenderer
import matplotlib
cmap = matplotlib.cm.get_cmap('cool')

from utils.io import draw_cams


parser = argparse.ArgumentParser(description='script to render cameras over epochs')
parser.add_argument('--testdir', default='',
                    help='path to test dir')
parser.add_argument('--cap_frame', default=-1,type=int,
                    help='number of frames to cap')
parser.add_argument('--first_idx', default=0,type=int,
                    help='first frame index to vis')
parser.add_argument('--last_idx', default=-1,type=int,
                    help='last frame index to vis')
parser.add_argument('--mesh_only', dest='mesh_only',action='store_true',
                    help='whether to only render rest mesh')
parser.add_argument('--bone_only', dest='bone_only',action='store_true',
                    help='whether to only render rest bone')
parser.add_argument('--draw_skel', dest='draw_skel',action='store_true',
                    help='whether to only render skeleton')             
args = parser.parse_args()

img_size = 1024

def main():
    # read all the data
    logname = args.testdir.split('/')[-2]
    varlist = [i for i in glob.glob('%s/vars_*.npy'%args.testdir) \
                        if 'latest.npy' not in i]
    varlist = sorted(varlist, 
            key=lambda x:int(x.split('/')[-1].split('vars_')[-1].split('.npy')[0]))
    
    # get first index that is used for optimization
    var = np.load(varlist[-1],allow_pickle=True)[()]
    var['rtk'] = var['rtk'][args.first_idx:args.last_idx] 
    first_valid_idx = np.linalg.norm(var['rtk'][:,:3,3], 2,-1)>0
    first_valid_idx = np.argmax(first_valid_idx)
    #varlist = varlist[1:]
    if args.cap_frame>-1:
        varlist = varlist[:args.cap_frame]
    size = len(varlist) - 1 
    # print(varlist)
    bone_list = [i for i in glob.glob('%s/bone_rest-*.obj'%args.testdir)]
    bone_list = sorted(bone_list, 
        key=lambda x:int(x.split('/')[-1].split('bone_rest-')[-1].split('.obj')[0]))
    
    if args.draw_skel:
        skel_list = [i for i in glob.glob('%s/skel_rest-*.obj'%args.testdir)]
        skel_list = sorted(skel_list, 
            key=lambda x:int(x.split('/')[-1].split('skel_rest-')[-1].split('.obj')[0]))        

    mesh_cams = []
    mesh_objs = []
    bone_objs = []
    skel_objs = []
    for var_path in varlist:
        # construct camera mesh
        var = np.load(var_path,allow_pickle=True)[()]
        var['rtk'] = var['rtk'][args.first_idx:args.last_idx] 
        mesh_cams.append(draw_cams(var['rtk'][first_valid_idx:]))
        mesh_objs.append(var['mesh_rest'])

    for bone_path in bone_list:
        bone_objs.append(trimesh.load(bone_path,process=False))

    if args.draw_skel:
        for skel_path in skel_list:
            skel_objs.append(trimesh.load(skel_path,process=False))

    frames = []
    # process cameras
    for i in range(size):
        # print(i)
        refcam = var['rtk'][first_valid_idx].copy()
        ## median camera trans
        #mtrans = np.median(np.linalg.norm(var['rtk'][first_valid_idx:,:3,3],2,-1)) 
        # max camera trans
        mtrans = np.max(np.linalg.norm(var['rtk'][first_valid_idx:,:3,3],2,-1)) 
        refcam[:2,3] = 0  # trans xy
        refcam[2,3] = 4*mtrans # depth
        refcam[3,:2] = 4*img_size/2 # fl
        refcam[3,2] = img_size/2
        refcam[3,3] = img_size/2
        # vp_rmat = refcam[:3,:3]
        # if args.mesh_only or args.bone_only: refcam[3,:2] *= 2 # make it appear larger
        # else:
        #     vp_rmat = cv2.Rodrigues(np.asarray([np.pi/2,0,0]))[0].dot(vp_rmat) # bev
        # refcam[:3,:3] = vp_rmat

        refcam[3,:2] *= 2
        # load vertices
        refmesh = mesh_cams[i]
        refface = torch.Tensor(refmesh.faces[None]).cuda()
        verts = torch.Tensor(refmesh.vertices[None]).cuda()

        # render
        Rmat =  torch.Tensor(refcam[None,:3,:3]).cuda()
        Tmat =  torch.Tensor(refcam[None,:3,3]).cuda()
        ppoint =refcam[3,2:]
        focal = refcam[3,:2]

        verts = obj_to_cam(verts, Rmat, Tmat)


        r = OffscreenRenderer(img_size, img_size)
        
        scene = Scene(ambient_light=0.4*np.asarray([1.,1.,1.,1.]))
        direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=6.0)
            
        smooth=True

        if not args.bone_only:
            mesh_obj = mesh_objs[i]
            if len(mesh_obj.vertices)>0:
                mesh_obj.vertices = obj2cam_np(mesh_obj.vertices, Rmat, Tmat)
                mesh_obj=Mesh.from_trimesh(mesh_obj,smooth=smooth)
                mesh_obj._primitives[0].material.RoughnessFactor=1.
                scene.add_node( Node(mesh=mesh_obj))
        
        if not args.mesh_only:
            bone_obj = bone_objs[i]
            if len(bone_obj.vertices)>0:
                bone_obj.vertices = obj2cam_np(bone_obj.vertices, Rmat, Tmat)
                bone_obj=Mesh.from_trimesh(bone_obj,smooth=smooth)
                bone_obj._primitives[0].material.RoughnessFactor=1.
                scene.add_node( Node(mesh=bone_obj))        
        
        if args.draw_skel:
            skel_obj = skel_objs[i]
            if len(skel_obj.vertices)>0:
                skel_obj.vertices = obj2cam_np(skel_obj.vertices, Rmat, Tmat)
                skel_obj=Mesh.from_trimesh(skel_obj,smooth=smooth)
                skel_obj._primitives[0].material.RoughnessFactor=1.
                scene.add_node( Node(mesh=skel_obj))                

        cam = IntrinsicsCamera(
                focal[0],
                focal[0],
                ppoint[0],
                ppoint[1],
                znear=1e-3,zfar=1000)
        cam_pose = -np.eye(4)
        cam_pose[0,0]=1
        cam_pose[-1,-1]=1
        cam_node = scene.add(cam, pose=cam_pose)
        light_pose =np.asarray([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]],dtype=float)
        light_pose[:3,:3] = cv2.Rodrigues(np.asarray([np.pi,0,0]))[0]

        direc_l_node = scene.add(direc_l, pose=light_pose)
        color, depth = r.render(scene,flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL | pyrender.RenderFlags.SKIP_CULL_FACES)
        r.delete()
        
        # save image
        color = color.astype(np.uint8)
        color = cv2.putText(color, 'epoch: %02d'%(i), (30,50), 
                cv2.FONT_HERSHEY_SIMPLEX,2, (256,0,0), 2)
        if args.bone_only:
            imoutpath = '%s/evolution-bone-only-%02d.png'%(args.testdir,i)
        elif args.mesh_only:
            imoutpath = '%s/evolution-mesh-only-%02d.png'%(args.testdir,i)
        else:
            imoutpath = '%s/evolution-%02d.png'%(args.testdir,i)
        cv2.imwrite(imoutpath,color[:,:,::-1] )
        frames.append(color)

    if args.bone_only:
        if args.draw_skel:
            save_vid('%s/evolution-skel'%args.testdir, frames, suffix='.gif') 
            save_vid('%s/evolution-skel'%args.testdir, frames, suffix='.mp4',upsample_frame=-1)
        else:
            save_vid('%s/evolution-bone-only'%args.testdir, frames, suffix='.gif') 
            save_vid('%s/evolution-bone-only'%args.testdir, frames, suffix='.mp4',upsample_frame=-1)
    elif args.mesh_only:
        save_vid('%s/evolution-mesh-only'%args.testdir, frames, suffix='.gif') 
        save_vid('%s/evolution-mesh-only'%args.testdir, frames, suffix='.mp4',upsample_frame=-1) 
    else:
        save_vid('%s/evolution'%args.testdir, frames, suffix='.gif') 
        save_vid('%s/evolution'%args.testdir, frames, suffix='.mp4',upsample_frame=-1)

if __name__ == '__main__':
    main()