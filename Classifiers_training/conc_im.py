# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 23:58:43 2019

@author: anton
"""
import numpy as np

def conc_im(im):
    c = np.zeros(shape=(im.shape[1],24,48))
    a = np.concatenate([im[0,0],im[0,1]],axis=1)
    
    for i in range(im.shape[0]):
        c[i] = np.concatenate([im[i,0],im[i,1]],axis=1)
    
    b = c[1]
    z = c[2]
    d = c[3]
    f = c[4]
    g = c[5]
    h = c[6]
    j = c[7]
    k = c[8]
    l = c[9]
    
    for i in range(1,im.shape[1]-1):
        a = np.concatenate([a,im[0,i+1]],axis=1)
        b = np.concatenate([b,im[1,i+1]],axis=1)
        z = np.concatenate([z,im[2,i+1]],axis=1)
        d = np.concatenate([d,im[3,i+1]],axis=1)
        f = np.concatenate([f,im[4,i+1]],axis=1)
        g = np.concatenate([g,im[5,i+1]],axis=1)
        h = np.concatenate([h,im[6,i+1]],axis=1)
        j = np.concatenate([j,im[7,i+1]],axis=1)
        k = np.concatenate([k,im[8,i+1]],axis=1)
        l = np.concatenate([l,im[9,i+1]],axis=1)
        
                
    im2 = np.concatenate([a,b,z,d,f,g,h,j,k,l],axis=0).astype('uint8')
    return im2  