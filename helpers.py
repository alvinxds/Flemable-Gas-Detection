# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 13:43:35 2019

@author: anton
"""

import numpy as np
import matplotlib.pyplot as plt
import skvideo.io
import cv2
import scipy.signal as sg


#Selective Function
def function_selective_model(in_vid,out_vid,a,out_vid2,kernel_size,sizelab):
    
    cap = cv2.VideoCapture(in_vid)
    _,frame = cap.read()
    size = (int(cap.get(3)),int(cap.get(4))) 
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(out_vid, fourcc, 25.0, size, isColor=0)
    avg = np.float32(frame)
    
    while(True):
        _,frame = cap.read()
        
        if _ == True:
            cv2.accumulateWeighted(frame,avg,a)
            res = cv2.convertScaleAbs(avg)
            res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
            out.write(res)
        
        else:
            break
        
        if cv2.waitKey(80) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    cap.release()
    out.release()
    
    vid = skvideo.io.vread(in_vid,outputdict={"-pix_fmt": "gray"})[:, :, :, 0]
    vid_slc = skvideo.io.vread(out_vid,outputdict={"-pix_fmt": "gray"})[:, :, :, 0]
    
    dif = np.zeros(shape=vid_slc.shape)
    thr = np.zeros(shape=vid_slc.shape)
    c = np.zeros(shape=vid_slc.shape[0])
    
    for k in range(vid_slc.shape[0]-1):    
        dif[k] = vid[k]-vid_slc[k]
        dif[k][np.where(dif[k]>235)]=0;
        c[k],thr[k] = cv2.threshold(dif[k],100,255,cv2.THRESH_BINARY);
        
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    slc_open = np.zeros(shape=thr.shape)
    slc_close = np.zeros(shape=thr.shape)
    
    for k in range(thr.shape[0]):
        slc_open[k] = cv2.morphologyEx(thr[k], cv2.MORPH_OPEN, kernel)
        slc_close[k] = cv2.morphologyEx(slc_open[k], cv2.MORPH_CLOSE, kernel)
    
    slc_close=np.asarray(slc_close, dtype=np.uint8)
    
    bcc_labvid = np.zeros(slc_close.shape)
    list_sts = []

    for k in range(slc_close.shape[0]):
        bcc_labvid[k] = cv2.connectedComponentsWithStats(slc_close[k],4,cv2.CV_32S)[1]
        list_sts.append(cv2.connectedComponentsWithStats(slc_close[k],4,cv2.CV_32S)[2])    
    for k in range(len(list_sts)):
        list_sts[k] = np.column_stack((list_sts[k],np.arange(0,np.size(list_sts[k],0))))    
    list_n = []    
    for k in range(len(list_sts)):
        list_n.append(list_sts[k][:,5][np.where(list_sts[k][:,4]<sizelab)])               
    for k in range(bcc_labvid.shape[0]):
        for i in range(bcc_labvid.shape[1]):
            for j in range(bcc_labvid.shape[2]):
                if(bcc_labvid[k,i,j] in list_n[k]):
                    bcc_labvid[k,i,j]=0 
                if(bcc_labvid[k,i,j] !=0):
                    bcc_labvid[k,i,j]=255
    
    bcc_labvid = np.asarray(bcc_labvid,np.uint8)    
    skvideo.io.vwrite(out_vid2, bcc_labvid)
    
    return bcc_labvid

#MOG Function
def function_mog(in_vid,out_vid,out_vid2,kernel_size,sizelab):
    cap = cv2.VideoCapture(in_vid)
    subtractor = cv2.createBackgroundSubtractorMOG2(history=25, varThreshold=50, detectShadows=True)
    size = (int(cap.get(3)),int(cap.get(4))) 
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(out_vid, fourcc, 25.0, size, isColor=0)
    
    while True:
        _, frame = cap.read()
        
        if _ == True:
            mask = subtractor.apply(frame)
            out.write(mask)
        else:
            break
        
        if  cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()
    cap.release()
    out.release()
    
    vid_mog = skvideo.io.vread(out_vid,outputdict={"-pix_fmt": "gray"})[:, :, :, 0]
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    mog_open = np.zeros(shape=vid_mog.shape)
    mog_close = np.zeros(shape=vid_mog.shape)
    
    for k in range(vid_mog.shape[0]):
        mog_open[k] = cv2.morphologyEx(vid_mog[k], cv2.MORPH_OPEN, kernel)
        mog_close[k] = cv2.morphologyEx(mog_open[k], cv2.MORPH_CLOSE, kernel)
        
    mog_close=np.asarray(mog_close, dtype=np.uint8)
    
    bcc_labvid = np.zeros(mog_close.shape)
    list_sts = []

    for k in range(mog_close.shape[0]):
        bcc_labvid[k] = cv2.connectedComponentsWithStats(mog_close[k],4,cv2.CV_32S)[1]
        list_sts.append(cv2.connectedComponentsWithStats(mog_close[k],4,cv2.CV_32S)[2])    
    for k in range(len(list_sts)):
        list_sts[k] = np.column_stack((list_sts[k],np.arange(0,np.size(list_sts[k],0))))    
    list_n = []    
    for k in range(len(list_sts)):
        list_n.append(list_sts[k][:,5][np.where(list_sts[k][:,4]<sizelab)])               
    for k in range(bcc_labvid.shape[0]):
        for i in range(bcc_labvid.shape[1]):
            for j in range(bcc_labvid.shape[2]):
                if(bcc_labvid[k,i,j] in list_n[k]):
                    bcc_labvid[k,i,j]=0 
                if(bcc_labvid[k,i,j] !=0):
                    bcc_labvid[k,i,j]=255
    
    bcc_labvid = np.asarray(bcc_labvid,np.uint8)    
    skvideo.io.vwrite(out_vid2, bcc_labvid)
    
    return bcc_labvid  

#Get Flow
def getFlow(imPrev, imNew):
    flow = cv2.calcOpticalFlowFarneback(imPrev, imNew, flow=None, pyr_scale=.5, levels=3, winsize=9, iterations=1, poly_n=3, poly_sigma=1.1, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    ang = ang * (180/ np.pi / 2)
    mag = mag.astype(np.uint8)
    mag = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    ang = ang.astype(np.uint8)
    
    return mag,ang    

#Color Pallete 
def apply_custom_colormap(image_gray, cmap=plt.get_cmap('seismic')):

    assert image_gray.dtype == np.uint8, 'must be np.uint8 image'
    if image_gray.ndim == 3: image_gray = image_gray.squeeze(-1)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256))[:,0:3]    # color range RGBA => RGB
    color_range = (color_range*255.0).astype(np.uint8)         # [0,1] => [0,255]
    color_range = np.squeeze(np.dstack([color_range[:,2], color_range[:,1], color_range[:,0]]), 0)  # RGB => BGR

    # Apply colormap for each channel individually
    channels = [cv2.LUT(image_gray, color_range[:,i]) for i in range(3)]
    
    return np.dstack(channels)

#Binaray Connected Components
def bcc_fun(in_vid,out_vid,sizelab):
    in_vid = np.asarray(in_vid,np.uint8)
    bcc_labvid = np.zeros(in_vid.shape)
    list_sts = []

    for k in range(in_vid.shape[0]):
        bcc_labvid[k] = cv2.connectedComponentsWithStats(in_vid[k],4,cv2.CV_32S)[1]
        list_sts.append(cv2.connectedComponentsWithStats(in_vid[k],4,cv2.CV_32S)[2])    
    for k in range(len(list_sts)):
        list_sts[k] = np.column_stack((list_sts[k],np.arange(0,np.size(list_sts[k],0))))    
    list_n = []    
    for k in range(len(list_sts)):
        list_n.append(list_sts[k][:,5][np.where(list_sts[k][:,4]<sizelab)])               
    for k in range(bcc_labvid.shape[0]):
        for i in range(bcc_labvid.shape[1]):
            for j in range(bcc_labvid.shape[2]):
                if(bcc_labvid[k,i,j] in list_n[k]):
                    bcc_labvid[k,i,j]=0 
                if(bcc_labvid[k,i,j] !=0):
                    bcc_labvid[k,i,j]=255
    
    bcc_labvid = np.asarray(bcc_labvid,np.uint8)    
    skvideo.io.vwrite(out_vid, bcc_labvid)
    
    return bcc_labvid

#Binary Remove Shapes
def bcc_size(bccimg_in,N):
    num = bccimg_in.max()
    for i in range(1,num+1):
        pts = np.where(bccimg_in == i)
        if len(pts[0]) < N:
            bccimg_in[pts] = 0
    return bccimg_in

#Find centers
def find_centroids(img_in):
    contours, hierarchy =   cv2.findContours(img_in,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    centres = []
    for i in range(len(contours)):
        moments = cv2.moments(contours[i])
        if (moments['m00']!=0):
            centres.append((int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))) 
    return centres

