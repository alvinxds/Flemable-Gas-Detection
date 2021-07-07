# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 17:36:17 2019

@author: anton
"""

import numpy as np
import cv2
import os
import skvideo.io
import skvideo
import matplotlib.pyplot as plt
import cv2


#Change current working directory
imagePath = "C:/Users/anton/Documents/Emp/Thesis/Python_code/Data/gas_det/train_classifier/raw_dataset/"
os.chdir(imagePath)

vid = skvideo.io.vread("propane/gas_propane_r.mp4",outputdict={"-pix_fmt": "gray"})[:, :, :, 0]
#Divide video to patches
matrix = vid

stepx = 48
stepy = 48
lists = []

for k in range (matrix.shape[0]):
    for i in range(0,matrix.shape[1],stepy):
        for j in range(0,matrix.shape[2],stepx):
            lists.append(matrix[k,i:i+stepy,j:j+stepx])
        
myarray = np.asarray(lists)
vid_cel = myarray.reshape(matrix.shape[0],5,7,48,48)


#Save the patches to select positive and non-positive
a=1800
plt.imshow(matrix[a],cmap='gray')

#os.makedirs('/patches2')
currentFrame = 16320
selec_prop = vid_cel[2700:2800,3,4]

#static datasets
for k in range (selec_prop.shape[0]):
    for i in range (selec_prop.shape[1]):
        name = './patches/patch' + str(currentFrame) + '.png'
        print ('Creating...' + name)
        cv2.imwrite(name, selec_prop[k,i,j])
        currentFrame += 1


for k in range (selec_prop.shape[0],10):
    for i in range (selec_prop.shape[1]):
        for j in range (selec_prop.shape[2]):
            name = './patches/patch/' + str(currentFrame) + '.png'
            print ('Creating...' + name)
            cv2.imwrite(name, selec_prop[k,i,j])
            currentFrame += 1

plt.imshow(vid[2000],cmap='gray')

#dynamic datasets 
selec_prop = vid_cel[2700:2711,3,5]                 
for k in range (selec_prop.shape[0]):
    for i in range (2,selec_prop.shape[1]-1):
        for j in range (2,selec_prop.shape[2]-1):
            name = './patches/cuboic' + str(currentFrame) + '.mp4'
            print ('Creating...' + name)
            skvideo.io.vwrite(name,selec_prop[:,i,j])
            currentFrame += 1            

selec_prop = vid_cel[2678:2689,3,1]  
currentFrame = 1959

name = './patches/cuboic' + str(currentFrame) + '.mp4'
print ('Creating...' + name)
skvideo.io.vwrite(name,selec_prop)
currentFrame += 1            

            

