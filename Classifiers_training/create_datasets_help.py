# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 13:25:28 2019

@author: anton
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import skvideo.io

#Change current working directory
imagePath = "C:/Users/anton/Documents/Emp/Thesis/Python_code/gas_detect/train_classifier/raw_dataset/"
os.chdir(imagePath)

#Calculate descriptors only for the ROI
vid_backgr = skvideo.io.vread("C:/Users/anton/Documents/Emp/Thesis/Python_code/gas_detect/foregrd_det_backr_sub/vid_backgr.mp4",outputdict={"-pix_fmt": "gray"})[:, :, :, 0]
vid_prop = skvideo.io.vread("./propane/gas_propane_r.mp4",outputdict={"-pix_fmt": "gray"})[:, :, :, 0]

#Video blocks
stepx = 24
stepy = 24
lists = []

for k in range (vid_backgr.shape[0]):
    for i in range(0,vid_backgr.shape[1],stepy):
        for j in range(0,vid_backgr.shape[2],stepx):
            lists.append(vid_backgr[k,i:i+stepy,j:j+stepx])
            
myarray = np.asarray(lists)
vid_brg_cells = myarray.reshape(vid_backgr.shape[0],vid_backgr.shape[1]/stepy,vid_backgr.shape[2]/stepx,stepy,stepx)

#Select only areas of interest
vid_roi = np.zeros(shape=vid_brg_cells.shape)

for k in range(vid_brg_cells.shape[0]):
    for i in range(vid_brg_cells.shape[1]-1):
        for j in range(vid_brg_cells.shape[2]-1):
            if (np.sum(vid_brg_cells[k,i,j])!=0 and np.sum(vid_brg_cells[k,i,j+1])!=0 and np.sum(vid_brg_cells[k,i+1,j])!=0 and np.sum(vid_brg_cells[k,i+1,j+1])!=0):
                vid_roi [k,i,j] = vid_brg_cells[k,i,j]
                
exec(open("C:/Users/anton/Documents/Emp/Thesis/Python_code/gas_detect/train_classifier/raw_dataset/conc_im.py").read()) 

vid_con = np.zeros(shape=(vid_backgr.shape))
for k in range (vid_roi.shape[0]):
    vid_con[k] = conc_im(vid_roi[k]) 
    
skvideo.io.vwrite('vid_roi_dataset.mp4',vid_con)

#Divide video to patches
matrix = vid_prop

stepx = 48
stepy = 48
lists = []

for k in range(matrix.shape[0]):
    for i in range(0,matrix.shape[1],stepy):
        for j in range(0,matrix.shape[2],stepx):
            lists.append(matrix[k,i:i+stepy,j:j+stepx])
        
myarray = np.asarray(lists)
vid_cel = myarray.reshape(matrix.shape[0],5,7,48,48)

#Redivide video roi to 48*48 patches
matrix = vid_con
lists = []
for k in range(matrix.shape[0]):
    for i in range(0,matrix.shape[1],stepy):
        for j in range(0,matrix.shape[2],stepx):
            lists.append(matrix[k,i:i+stepy,j:j+stepx])

myarray = np.asarray(lists)
vid_roi = myarray.reshape(matrix.shape[0],5,7,48,48)

#Export training images from roi for static training sets
os.makedirs('new_roi')
currentFrame = 100000
#gas patches
for k in range(2640,vid_roi.shape[0]-1):
    for i in range(2,vid_roi.shape[1]-1):
        for j in range(2,vid_roi.shape[2]-1):
            if (np.sum(vid_roi[k,i,j])!=0):
                if((i!=2 and j!=1) or (i!=2 and j!=2)):
                    name = './new_roi/patch' + str(currentFrame) + '.png'
                    print ('Creating...' + name)
                    cv2.imwrite(name, vid_cel[k,i,j])
                    currentFrame += 1
#not gas patches                    
for k in range(2000,2300):
    for i in range(2,vid_roi.shape[1]-1):
        for j in range(2,vid_roi.shape[2]-1):
            if (np.sum(vid_roi[k,i,j])==0):
                name = './new_roi/patch' + str(currentFrame) + '.png'
                print ('Creating...' + name)
                cv2.imwrite(name, vid_cel[k,i,j])
                currentFrame += 1

#Calculate dynamic datasets 
currentFrame = 100000  

selec_prop = vid_cel[3178:3189,2,5] 

name = './new_roi/cuboic' + str(currentFrame) + '.mp4'
print ('Creating...' + name)
skvideo.io.vwrite(name,selec_prop)
currentFrame += 1            
