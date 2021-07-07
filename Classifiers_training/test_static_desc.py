# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:22:42 2019

@author: anton
"""

import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import skvideo.io
from sklearn.externals import joblib
import pywt
from skimage.feature import hog

#Change current working directory
imagePath = "C:/Users/anton/Documents/Emp/Thesis/Python_code/Data/gas_det/train_classifier/raw_dataset/"
os.chdir(imagePath)

#Test static classifier in images
clf_lbp_nri = joblib.load('./patches/lbp_nri_classif.pkl')
clf_lbp_uni = joblib.load('./patches/lbp_uni_classif.pkl')
clf_wav_eng = joblib.load('./patches/wavelet_energ_classif.pkl')
clf_eoh = joblib.load('./patches/eoh_classif.pkl')

vid= skvideo.io.vread("propane/gas_propane_r.mp4",outputdict={"-pix_fmt": "gray"})[:, :, :, 0]
img=vid[2900]

#plt.imshow(img,cmap='gray')

#Divide image to blocks to calculate descriptors

###########################Static Descriptors
#image blocks
stepx = 24
stepy = 24
lists = []

for i in range(0,img.shape[0],stepy):
    for j in range(0,img.shape[1],stepx):
        lists.append(img[i:i+stepy,j:j+stepx])        
myarray = np.asarray(lists)
blocks = myarray.reshape(img.shape[0]/stepy,img.shape[1]/stepx,stepy,stepx)

radius = 1
n_points = 8 * radius
method1 = 'nri_uniform'
method2 = 'uniform'

lbp_nri = []
lbp_uni = []
#Calculate nri and uni LBPs
for i in range(blocks.shape[0]):
    for j in range(blocks.shape[1]):
            lbp_nri.append(local_binary_pattern(blocks[i,j], n_points, radius, method1))
            lbp_uni.append(local_binary_pattern(blocks[i,j], n_points, radius, method2))

#Calculate wavelets energy
coef = []
wave_enrg = []

for i in range(blocks.shape[0]):
    for j in range(blocks.shape[1]):
        coef.append(pywt.wavedec2(blocks[i,j], 'db1', level=2))            

ll, lh, hl, hh = [],[],[],[]
for k in range (len(coef)):
    ll.append(coef[k][0])
    lh.append(coef[k][0][0])
    hl.append(coef[k][0][1])
    hh.append(coef[k][0][2])
    wave_enrg.append(lh[k]**2+hl[k]**2+hh[k]**2)

#Calculate edge and orientations
eoh = []
for i in range(blocks.shape[0]):
    for j in range(blocks.shape[1]):
        eoh.append(hog(blocks[i,j], orientations=18, pixels_per_cell=(4, 4),cells_per_block=(1, 1),
                    block_norm="L1", visualize=False, feature_vector=True))

#Predictions                     
y_pred_lbpnri = [] 
y_pred_lbpuni = []
y_pred_waverg = []                
y_pred_eoh = []

for k in range(len(lbp_nri)):
    lbp_nri[k] = lbp_nri[k].reshape(1,stepy*stepx) 
    lbp_uni[k] = lbp_uni[k].reshape(1,stepy*stepx) 
    wave_enrg[k] = wave_enrg[k].reshape(1,wave_enrg[k].shape[0])
    eoh[k] = eoh[k].reshape(1,eoh[k].shape[0])
    
    y_pred_lbpnri.append(clf_lbp_nri.predict(lbp_nri[k])) 
    y_pred_lbpuni.append(clf_lbp_uni.predict(lbp_uni[k])) 
    y_pred_waverg.append(clf_wav_eng.predict(wave_enrg[k]))
    y_pred_eoh.append(clf_eoh.predict(eoh[k]))

pred = []
for k in range(len(y_pred_lbpnri)):    
    if (y_pred_lbpnri[k]==1 and y_pred_lbpuni[k]==1 and y_pred_waverg[k]==1 or y_pred_eoh[k]==1):
        pred.append(np.ones((stepx,stepy)))
    else:
        pred.append(np.zeros((stepx,stepy)))

pred = np.asarray(pred)    
pred = pred.reshape(img.shape[0]/stepy,img.shape[1]/stepx,stepy,stepx)        

exec(open("C:/Users/anton/Documents/Emp/Thesis/Python_code/Data/gas_det/train_classifier/raw_dataset/conc_im.py").read()) 
pred = conc_im(pred)   
color = (255,0,0)
clone = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

plt.figure()
for i in range(0,img.shape[0],stepy):
    for j in range(0,img.shape[1],stepx):
        if np.mean(pred[i:i+stepy,j:j+stepx])==1:
            cv2.rectangle(clone, (i, j), (i + stepy, j + stepx), color, 2)
                   
        plt.imshow(clone,cmap='gray')
            

          
            
            
            
            
            