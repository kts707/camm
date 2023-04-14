import sys, os
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
os.environ["PYOPENGL_PLATFORM"] = "egl"
curr_dir = os.path.abspath(os.getcwd())
sys.path.insert(0,curr_dir)

import glob
from utils.io import save_vid
import numpy as np
import torch
import cv2
import argparse
import trimesh
from nnutils.geom_utils import  obj2cam_np
import pyrender
from pyrender import IntrinsicsCamera,Mesh, Node, Scene,OffscreenRenderer



parser = argparse.ArgumentParser(description='script to render cameras over epochs')
parser.add_argument('--testdir', default='',
                    help='path to test dir')
parser.add_argument('--meshdir', default='',
                    help='path to test dir')
parser.add_argument('--first_idx', default=0,type=int,
                    help='first frame index to vis')
parser.add_argument('--last_idx', default=-1,type=int,
                    help='last frame index to vis')
parser.add_argument('--draw_mesh', dest='draw_mesh',action='store_true',
                    help='whether to only render rest mesh')
parser.add_argument('--vp', default=0, type=int,
                    help='which viewpoint to render 0,1,2')
parser.add_argument('--draw_skel', dest='draw_skel',action='store_true',
                    help='whether to only render skeleton')             
args = parser.parse_args()

img_size = 1024

def main():
    var = np.load('%s/vars_latest.npy'%args.testdir,allow_pickle=True)[()]
    var['rtk'] = var['rtk'][args.first_idx:args.last_idx] 
    first_valid_idx = np.linalg.norm(var['rtk'][:,:3,3], 2,-1)>0
    first_valid_idx = np.argmax(first_valid_idx)

    # total number of frames
    size = 100

    if args.draw_mesh:
        mesh_list = [i for i in glob.glob('%s/mesh-*.obj'%args.meshdir)]
        mesh_list = sorted(mesh_list, 
            key=lambda x:int(x.split('/')[-1].split('mesh-')[-1].split('.obj')[0]))    
    
    if args.draw_skel:
        skel_list = [i for i in glob.glob('%s/skel-*.obj'%args.meshdir)]
        skel_list = sorted(skel_list, 
            key=lambda x:int(x.split('/')[-1].split('skel-')[-1].split('.obj')[0]))        

    mesh_objs = []
    skel_objs = []


    if args.draw_skel:
        for skel_path in skel_list:
            skel_objs.append(trimesh.load(skel_path,process=False))
        size = len(skel_objs)

    if args.draw_mesh:
        for mesh_path in mesh_list:
            mesh_objs.append(trimesh.load(mesh_path,process=False))
        size = len(mesh_objs)
    frames = []

    # process cameras
    if args.vp==1:
        vp_rmat = cv2.Rodrigues(np.asarray([0,np.pi/2,0]))[0]
    elif args.vp==2:
        vp_rmat = cv2.Rodrigues(np.asarray([np.pi/2,0,0]))[0]
    else:
        vp_rmat = cv2.Rodrigues(np.asarray([0.,0,0]))[0]
    for i in range(size):
        refcam = var['rtk'][first_valid_idx].copy()
        # max camera trans
        mtrans = np.max(np.linalg.norm(var['rtk'][first_valid_idx:,:3,3],2,-1)) 
        refcam[:2,3] = 0  # trans xy
        refcam[2,3] = 4*mtrans # depth
        refcam[3,:2] = 4*img_size/2 # fl
        refcam[3,2] = img_size/2
        refcam[3,3] = img_size/2

        refcam[3,:2] *= 2
        refcam[:3,:3] = vp_rmat.dot(refcam[:3,:3])
        # render
        Rmat =  torch.Tensor(refcam[None,:3,:3]).cuda()
        Tmat =  torch.Tensor(refcam[None,:3,3]).cuda()
        ppoint =refcam[3,2:]
        focal = refcam[3,:2]

        r = OffscreenRenderer(img_size, img_size)
        
        scene = Scene(ambient_light=0.4*np.asarray([1.,1.,1.,1.]))
        direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=6.0)
            
        smooth=True

        if args.draw_mesh:
            mesh_obj = mesh_objs[i]
            if len(mesh_obj.vertices)>0:
                mesh_obj.vertices = obj2cam_np(mesh_obj.vertices, Rmat, Tmat)
                mesh_obj=Mesh.from_trimesh(mesh_obj,smooth=smooth)
                mesh_obj._primitives[0].material.RoughnessFactor=1.
                scene.add_node( Node(mesh=mesh_obj))    
        
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
        if args.draw_mesh:
            imoutpath = '%s/mesh-%01d-%02d.png'%(args.meshdir,args.vp,i)
        elif args.draw_skel:
            imoutpath = '%s/skel-%01d-%02d.png'%(args.meshdir,args.vp,i)
        else:
            raise NotImplementedError
        cv2.imwrite(imoutpath,color[:,:,::-1] )
        frames.append(color)

    if args.draw_skel:
        save_vid('%s/animate-skel-%01d'%(args.meshdir, args.vp), frames, suffix='.gif') 
        save_vid('%s/animate-skel-%01d'%(args.meshdir,args.vp), frames, suffix='.mp4',upsample_frame=-1)
    elif args.draw_mesh:
        save_vid('%s/animate-mesh-%01d'%(args.meshdir, args.vp), frames, suffix='.gif') 
        save_vid('%s/animate-mesh-%01d'%(args.meshdir, args.vp), frames, suffix='.mp4',upsample_frame=-1)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    main()