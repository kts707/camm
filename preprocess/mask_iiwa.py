import cv2
import glob
import numpy as np
import os

import sys
curr_dir = os.path.abspath(os.getcwd())
sys.path.insert(0,curr_dir)
        

seqname=sys.argv[1]
datadir='tmp/%s/images/'%seqname
odir='database/DAVIS/'
imgdir= '%s/JPEGImages/Full-Resolution/%s'%(odir,seqname)
maskdir='%s/Annotations/Full-Resolution/%s'%(odir,seqname)

counter=0 
frames = []
for i,path in enumerate(sorted(glob.glob('%s/*'%datadir))):
    print(path)
    img = cv2.imread(path)
    pp = path.replace('images', 'masks')

    masks = cv2.imread(pp)
    h,w = img.shape[:2]
    (_, masks) = cv2.threshold(masks, 220, 255, cv2.THRESH_BINARY)

    cv2.imwrite('%s/%05d.jpg'%(imgdir,counter), img)
    cv2.imwrite('%s/%05d.png'%(maskdir,counter), masks)
  
    counter+=1