# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 14:59:55 2019

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
imagePath = "C:/Users/anton/Documents/Emp/Thesis/Python_code/Data/gas_det/train_classifier/"
os.chdir(imagePath)

vid_meth = skvideo.io.vread("raw_dataset/gas_methane.mp4",outputdict={"-pix_fmt": "gray"})[:, :, :, 0]
vid_mix = skvideo.io.vread("raw_dataset/gas_mixture.mp4",outputdict={"-pix_fmt": "gray"})[:, :, :, 0]
vid_prop = skvideo.io.vread("raw_dataset/gas_propane.mp4",outputdict={"-pix_fmt": "gray"})[:, :, :, 0]
vid_prop2 = skvideo.io.vread("raw_dataset/gas_propane2.mp4",outputdict={"-pix_fmt": "gray"})[:, :, :, 0]
vid_nogas = skvideo.io.vread("raw_dataset/non_gas.mp4",outputdict={"-pix_fmt": "gray"})[:, :, :, 0]

#Dinstinguish raw from lap-filter
plt.figure()
plt.imshow(vid_prop2[1580],cmap='gray')
#Methane Video
vid_meth_r = vid_meth[:3560,:,8:344] 
vid_meth_l = vid_meth[3800:,:,8:344] 
skvideo.io.vwrite('raw_dataset/gas_methane_r.mp4',vid_meth_r)
skvideo.io.vwrite('raw_dataset/gas_methane_l.mp4',vid_meth_l)

#Mixture Video
vid_mix_l = vid_mix[:,:,8:344]
skvideo.io.vwrite('raw_dataset/gas_mixture_l.mp4',vid_mix_l)

#Propane1 Video
vid_prop_r = vid_prop[:3810,:,8:344]
vid_prop_l = vid_prop[3880:,:,8:344]
skvideo.io.vwrite('raw_dataset/gas_propane_r.mp4',vid_prop_r)
skvideo.io.vwrite('raw_dataset/gas_propane_l.mp4',vid_prop_l)

#Propane2 Video
vid_prop2_l = vid_prop2[:,:,8:344]
skvideo.io.vwrite('raw_dataset/gas_propane2_l.mp4',vid_prop2_l)

#Non gas video
vid_nogas_r = vid_nogas[:,:,8:344]
skvideo.io.vwrite('raw_dataset/non_gas_l.mp4',vid_nogas_r)