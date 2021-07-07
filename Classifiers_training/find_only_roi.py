# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:41:36 2019

@author: anton
"""

import numpy as np
import os
import skvideo.io

#Change current working directory
imagePath = "C:/Users/anton/Documents/Emp/Thesis/Python_code/gas_detect/train_classifier/raw_dataset/"
os.chdir(imagePath)

vid = skvideo.io.vread("propane/gas_propane_r.mp4",outputdict={"-pix_fmt": "gray"})[:, :, :, 0]
backgr = skvideo.io.vread('C:/Users/anton/Documents/Emp/Thesis/Python_code/gas_detect/foregrd_det_backr_sub/vid_backgr2.mp4',outputdict={"-pix_fmt": "gray"})[:, :, :, 0]

#Change values of background to binary 0-1
for k in range(backgr.shape[0]):
    for i in range(backgr.shape[1]):
        for j in range(backgr.shape[2]):
            if (backgr[k,i,j] !=0):
                backgr[k,i,j] = 1

#Video blocks
stepx = 24
stepy = 24
lists = []

for k in range (vid.shape[0]):
    for i in range(0,vid.shape[1],stepy):
        for j in range(0,vid.shape[2],stepx):
            lists.append(vid[k,i:i+stepy,j:j+stepx])
            
myarray = np.asarray(lists)
vid_cells = myarray.reshape(vid.shape[0],vid.shape[1]/stepy,vid.shape[2]/stepx,stepy,stepx)

#Background to blocks
stepx = 24
stepy = 24
lists = []

for k in range (backgr.shape[0]):
    for i in range(0,backgr.shape[1],stepy):
        for j in range(0,backgr.shape[2],stepx):
            lists.append(backgr[k,i:i+stepy,j:j+stepx])
            
myarray = np.asarray(lists)
backgr_cells = myarray.reshape(backgr.shape[0],backgr.shape[1]/stepy,backgr.shape[2]/stepx,stepy,stepx)


#Calculate moving cells for the whole video
vidn = np.zeros(shape=(vid_cells.shape))
for k in range (vid_cells.shape[0]-1):
    for i in range (vid_cells.shape[1]):
        for j in range(vid_cells.shape[2]):
            if (np.sum(backgr_cells[k,i,j]) > 100):
                vidn[k,i,j] = vid_cells[k,i,j]
            else:
                vidn[k,i,j] = np.zeros((stepx,stepy))

exec(open("C:/Users/anton/Documents/Emp/Thesis/Python_code/gas_detect/train_classifier/raw_dataset/conc_im.py").read()) 

vid_con = np.zeros(shape=(vid.shape))
for k in range (vidn.shape[0]):
    vid_con[k] = conc_im(vidn[k]) 
    
skvideo.io.vwrite('vid_con.mp4',vid_con)

#Remove cells with with magnitude of flow
exec(open("C:/Users/anton/Documents/Emp/Thesis/Python_code/gas_detect/train_classifier/raw_dataset/hof_get_flow.py").read()) 

mag=np.zeros(shape=vid.shape)
ang=np.zeros(shape=vid.shape)
flow=np.zeros(shape=(3810,240,336,2))


for k in range(vid.shape[0]-1):
    flow[k], ang[k], mag[k] = getFlow(vid[k],vid[k+1])

skvideo.io.vwrite('mag_vid.mp4',mag)

stepx = 24
stepy = 24
lists = []

for k in range (mag.shape[0]):
    for i in range(0,mag.shape[1],stepy):
        for j in range(0,mag.shape[2],stepx):
            lists.append(mag[k,i:i+stepy,j:j+stepx])
            
myarray = np.asarray(lists)
mag_cells = myarray.reshape(mag.shape[0],mag.shape[1]/stepy,mag.shape[2]/stepx,stepy,stepx)

vidn2=np.zeros(shape=(vidn.shape))
for k in range(vidn2.shape[0]):
    for i in range(vidn2.shape[1]):
        for j in range(vidn2.shape[2]):
            if (np.mean(mag_cells[k,i,j]) < 10):
                vidn2[k,i,j] = vidn[k,i,j]
            else:
                vidn2[k,i,j] = np.zeros((stepx,stepy))
            
vid_nf = np.zeros(shape=(vid.shape))
for k in range (vid_nf.shape[0]):
    vid_nf[k] = conc_im(vidn2[k]) 

skvideo.io.vwrite('vid_nf.mp4',vid_nf)

vidnn = vidn.copy()
for k in range(vidn.shape[0]):
    for i in range(vidn.shape[1]):
        for j in range(vidn.shape[2]):
            if (np.mean(vidn2[k,i,j])!=0):
                vidnn[k,i,j] = vidn[k,i,j]
            else:
                vidnn[k,i,j] = np.zeros((stepx,stepy))    

vid_nfn = np.zeros(shape=(vid.shape))
for k in range (vid_nfn.shape[0]):
    vid_nfn[k] = conc_im(vidnn[k]) 

skvideo.io.vwrite('vid_nfn.mp4',vid_nfn)

exec(open("C:/Users/anton/Documents/Emp/Thesis/Python_code/gas_detect/train_classifier/raw_dataset/play_vid.py").read()) 
names = ("vid_con.mp4","vid_nfn.mp4")
win = ('con','nfn')
play_vid(names,win)