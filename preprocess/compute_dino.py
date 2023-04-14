import torch
import sys
import os
from pathlib import Path
from dino_feature_extractor import ViTExtractor
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import binary_erosion
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

pickle_file_path=sys.argv[1]
dim=int(sys.argv[2])

# parameters for DINO ViT
model_type = 'dino_vits8'
stride = 8
layer = 11
facet = 'key'
load_size_ratio = 0.5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

img_size = 512

print('---------------DINO ViT info---------------')
print('data info file:',pickle_file_path)
print('model_type:',model_type)
print('stride:',stride)
print('ViT layer: layer ',layer)
print('feature descriptor type:',facet)
print('load size ratio:',load_size_ratio)
print('dimension after PCA:',dim)

def compute_crop_params(mask, img_size=512, crop_factor=1.2, flip=0):
    #ss=time.time()
    indices = np.where(mask>0); xid = indices[1]; yid = indices[0]
    center = ( (xid.max()+xid.min())//2, (yid.max()+yid.min())//2)
    length = ( (xid.max()-xid.min())//2, (yid.max()-yid.min())//2)
    length = (int(crop_factor*length[0]), int(crop_factor*length[1]))
    
    #print('center:%f'%(time.time()-ss))

    maxw=img_size;maxh=img_size
    orisize = (2*length[0], 2*length[1])
    alp =  [orisize[0]/maxw  ,orisize[1]/maxw]
    
    # intrinsics induced by augmentation: augmented to to original img
    # correct cx,cy at clip space (not tx, ty)
    if flip==0:
        pps  = np.asarray([float( center[0] - length[0] ), float( center[1] - length[1]  )])
    else:
        pps  = np.asarray([-float( center[0] - length[0] ), float( center[1] - length[1]  )])
    kaug = np.asarray([alp[0], alp[1], pps[0], pps[1]])

    x0,y0  =np.meshgrid(range(maxw),range(maxh))
    A = np.eye(3)
    B = np.asarray([[alp[0],0,(center[0]-length[0])],
                    [0,alp[1],(center[1]-length[1])],
                    [0,0,1]]).T
    hp0 = np.stack([x0,y0,np.ones_like(x0)],-1)  # screen coord
    hp0 = np.dot(hp0,A.dot(B))                   # image coord
    return kaug, hp0, A,B


with open(pickle_file_path, 'rb') as f:
    data_info = pickle.load(f)

imgpaths = data_info['imgpaths']
mskpaths = data_info['mskpaths']
frameid_list = data_info['frameid_list']
dt_list = data_info['dt_list']
base_path = 'database/DAVIS/DINO_Pixels/Full-Resolution/'
filtered_img_path_base = 'filtered_imgs'
load_pair = True
overwrite = True

if not os.path.isdir(filtered_img_path_base):
    os.makedirs(filtered_img_path_base)

with torch.no_grad():
    extractor = ViTExtractor(model_type, stride, device=device)

    for i in range(len(frameid_list)):
        img1_path = imgpaths[frameid_list[i][0][0]]
        msk1_path = mskpaths[frameid_list[i][0][0]]

        seqname_sub = img1_path.split('/')[-2]
        frameid_sub = img1_path.split('/')[-1].split('.')[0]
        dt = dt_list[i]


        img = cv2.imread(img1_path)
        h, w, _ = img.shape

        mask = cv2.imread(msk1_path,0)
        msk1_cv2 = (mask>0).astype(float)
        img_filterd = img * msk1_cv2[...,None]

        img_path_filtered = '%s/filtered_img_%s_%s.jpg'%(filtered_img_path_base, seqname_sub, frameid_sub)


        cv2.imwrite(img_path_filtered,img_filterd)

        image1_batch, image1_pil = extractor.preprocess(img_path_filtered, load_size_ratio=load_size_ratio)
        image_features = extractor.extract_raw_features(image1_batch.to(device), layer, facet)


        image_features = torch.nn.functional.normalize(image_features, p=2, dim=2)
        feature_shape = (1,image_features.shape[-1],extractor.num_patches[0],extractor.num_patches[1])
        image_features_pca = image_features.view(feature_shape[2]*feature_shape[3],-1).cpu().numpy()


        pca = PCA(n_components=dim).fit(image_features_pca)
        image_features_pca = torch.from_numpy(pca.transform(image_features_pca)).float()
        image_features_pca = image_features_pca.view(1,feature_shape[2],feature_shape[3],-1).permute(0,3,1,2)

        upsampled_features_pca = torch.nn.functional.interpolate(image_features_pca, size=(h,w), mode='bilinear', align_corners=True)


        # resample in 512x512 image size
        mask = mask/np.sort(np.unique(mask))[1]
        occluder = mask==255
        mask[occluder] = 0
        if mask.shape[0]!=img.shape[0] or mask.shape[1]!=img.shape[1]:
            mask = cv2.resize(mask, img.shape[:2][::-1],interpolation=cv2.INTER_NEAREST)
            mask = binary_erosion(mask,iterations=2)
        mask = np.expand_dims(mask, 2)

        kaug, hp0, A, B= compute_crop_params(mask)
        #print('crop params:%f'%(time.time()-ss))
        x0 = hp0[:,:,0].astype(np.float32)
        y0 = hp0[:,:,1].astype(np.float32)
        mask_rszd = torch.from_numpy(cv2.remap(mask.astype(int),x0,y0,interpolation=cv2.INTER_NEAREST)).int().view(1,img_size,img_size).to(upsampled_features_pca.device)

        x_grid = torch.from_numpy(x0).float().view(img_size,img_size,1) * (2 / w) - 1
        y_grid = torch.from_numpy(y0).float().view(img_size,img_size,1) * (2 / h) - 1

        sampling_grid = torch.cat([x_grid, y_grid],-1).unsqueeze(0)
        resampled_features = torch.nn.functional.grid_sample(input=upsampled_features_pca, grid=sampling_grid)

        resampled_features = resampled_features * mask_rszd

        if load_pair:
            img2_path = imgpaths[frameid_list[i][0][1]]
            msk2_path = mskpaths[frameid_list[i][0][1]]

            seqname_sub2 = img2_path.split('/')[-2]
            frameid_sub2 = img2_path.split('/')[-1].split('.')[0]


            img2 = cv2.imread(img2_path)
            h2, w2, _ = img2.shape

            mask2 = cv2.imread(msk2_path,0)
            msk2_cv2 = (mask2>0).astype(float)
            img2_filterd = img2 * msk2_cv2[...,None]

            img2_path_filtered = '%s/filtered_img_%s_%s.jpg'%(filtered_img_path_base, seqname_sub2, frameid_sub2)
            cv2.imwrite(img2_path_filtered,img2_filterd)

            image2_batch, image2_pil = extractor.preprocess(img2_path_filtered, load_size_ratio=load_size_ratio)
            image_features2 = extractor.extract_raw_features(image2_batch.to(device), layer, facet)

            image_features2 = torch.nn.functional.normalize(image_features2, p=2, dim=2)
            feature_shape2 = (1,image_features2.shape[-1],extractor.num_patches[0],extractor.num_patches[1])
            image_features2_pca = image_features2.view(feature_shape2[2]*feature_shape2[3],-1).cpu().numpy()

            pca = PCA(n_components=dim).fit(image_features2_pca)
            image_features2_pca = torch.from_numpy(pca.transform(image_features2_pca)).float()
            image_features2_pca = image_features2_pca.view(1,feature_shape2[2],feature_shape2[3],-1).permute(0,3,1,2)

            upsampled_features2_pca = torch.nn.functional.interpolate(image_features2_pca, size=(h2,w2), mode='bilinear', align_corners=True)

            # resample in 512x512 image size
            mask2 = mask2/np.sort(np.unique(mask2))[1]
            occluder2 = mask2==255
            mask2[occluder2] = 0
            if mask2.shape[0]!=img2.shape[0] or mask2.shape[1]!=img2.shape[1]:
                mask2 = cv2.resize(mask2, img2.shape[:2][::-1],interpolation=cv2.INTER_NEAREST)
                mask2 = binary_erosion(mask2,iterations=2)
            mask2 = np.expand_dims(mask2, 2)

            kaug2, hp0_2, A2, B2= compute_crop_params(mask2)
            x0_2 = hp0_2[:,:,0].astype(np.float32)
            y0_2 = hp0_2[:,:,1].astype(np.float32)

            mask_rszd2 = torch.from_numpy(cv2.remap(mask2.astype(int),x0_2,y0_2,interpolation=cv2.INTER_NEAREST)).int().view(1,img_size,img_size).to(upsampled_features2_pca.device)

            x_grid2 = torch.from_numpy(x0_2).float().view(img_size,img_size,1) * (2 / w2) - 1
            y_grid2 = torch.from_numpy(y0_2).float().view(img_size,img_size,1) * (2 / h2) - 1

            sampling_grid2 = torch.cat([x_grid2, y_grid2],-1).unsqueeze(0)
            resampled_features2 = torch.nn.functional.grid_sample(input=upsampled_features2_pca, grid=sampling_grid2)

            resampled_features2 = resampled_features2 * mask_rszd2

            resampled_features = resampled_features.view(1,-1,1, dim, img_size, img_size)
            resampled_features2 = resampled_features2.view(1,-1,1, dim, img_size, img_size)
            resampled_features = torch.cat([resampled_features, resampled_features2], dim=1)


        resampled_features_numpy = resampled_features.view(1,-1,1, dim, img_size, img_size).detach().cpu().numpy()

        save_dir = '%s/%s'%(base_path, seqname_sub)
        save_dir_t = '%s/%d_%s'%(save_dir, dt, frameid_sub)
        print(save_dir_t)
        if (not overwrite) and os.path.exists(save_dir_t):
            continue
        if not os.path.isdir(save_dir_t):
            os.makedirs(save_dir_t)

        for idy in range(img_size):
            save_path = '%s/%04d.npy'%(save_dir_t, idy)
            dino_feat_idy = resampled_features_numpy[...,idy,:]
            np.save(save_path, dino_feat_idy)