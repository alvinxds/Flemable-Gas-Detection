# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 19:15:45 2019

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
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

#Change current working directory
imagePath = "C:/Users/anton/Documents/Emp/Thesis/Python_code/gas_detect/train_classifier/raw_dataset/"
os.chdir(imagePath)

#SVM classifiers
clf_svm_lbp_nri = joblib.load('./patches/lbp_nri_svm_classif.pkl')
clf_svm_lbp_uni = joblib.load('./patches/lbp_uni_svm_classif.pkl')
clf_svm_wav_eng = joblib.load('./patches/wavelet_energ_svm_classif.pkl')
clf_svm_eoh = joblib.load('./patches/eoh_svm_classif.pkl')
clf_svm_lbptop = joblib.load('./patches/lbptop_svm_classif.pkl')
clf_svm_hoghof = joblib.load('./patches/hoghof_svm_classif.pkl')

#Adaboost classifiers
#SVM classifiers
clf_Adab_lbp_nri = joblib.load('./patches/lbp_nri_Adaboost_classif.pkl')
clf_Adab_lbp_uni = joblib.load('./patches/lbp_uni_Adaboost_classif.pkl')
clf_Adab_wav_eng = joblib.load('./patches/wavelet_energ_Adaboost_classif.pkl')
clf_Adab_eoh = joblib.load('./patches/eoh_Adaboost_classif.pkl')
clf_Adab_lbptop = joblib.load('./patches/lbptop_Adaboost_classif.pkl')
clf_Adab_hoghof = joblib.load('./patches/hoghof_Adaboost_classif.pkl')

#Calculate descriptors only for the ROI
vid = skvideo.io.vread("./vid_nfn.mp4",outputdict={"-pix_fmt": "gray"})[:, :, :, 0]
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

#LBPTOP
radius = 1
n_points = 8 * radius
method = 'nri_uniform'

#LBPTOP for gas cuboics
lbp_gxy = []
lbp_gxt = []
lbp_gyt = []              

for k in range(vid_cells.shape[0]):
    for i in range(vid_cells.shape[1]):
        for j in range(vid_cells.shape[2]):
            if (np.mean(vid_cells[k,i,j] !=0)):
                lbp_gxy.append(local_binary_pattern(vid_cells[k,i,j], n_points, radius, method))
                lbp_gxt.append(local_binary_pattern(vid_cells[k:k+11,i,j,11,:], n_points, radius, method))
                lbp_gyt.append(local_binary_pattern(vid_cells[k:k+11,i,j,:,11], n_points, radius, method))
            else:
                lbp_gxy.append(vid_cells[k,i,j])
                lbp_gxt.append(np.zeros((11,24)))
                lbp_gyt.append(np.zeros((11,24)))
                                
new_index = range(0, 59)            


#uni=[]
#for k in range (len(lbp_gxy)):
#    uni.append(np.unique(lbp_gxy[k],return_counts=True))
#
#myarray_uni = np.asarray(uni)            
#uni = myarray_uni.reshape(3810,10,14,2)
#unis = []
#for k in range(uni.shape[0]):
#    for i in range(uni.shape[1]):
#        for j in range(uni.shape[2]):
#            unis.append(np.column_stack((uni[k,i,j][0], uni[k,i,j][1])))
#
#
#for k in range (len(unis)):
#    unis[k] = pd.DataFrame(unis[k])
#    unis[k] = unis[k].reindex(new_index,fill_value=0)

          
for k in range (len(lbp_gxy)):
    lbp_gxy[k] = pd.DataFrame(lbp_gxy[k])
    lbp_gxy[k] = pd.melt(lbp_gxy[k])
    lbp_gxy[k] = lbp_gxy[k].groupby(['value']).count()
    lbp_gxy[k] = lbp_gxy[k].reindex(new_index,fill_value=0)
    lbp_gxy[k] = pd.concat([lbp_gxy[k],((lbp_gxy[k]/lbp_gxy[k].sum())*100)/100],axis=1)
    lbp_gxy[k].reset_index(level=0, inplace=True)

for k in range (len(lbp_gxt)):
    lbp_gxt[k] = pd.DataFrame(lbp_gxt[k])
    lbp_gxt[k] = pd.melt(lbp_gxt[k])
    lbp_gxt[k] = lbp_gxt[k].groupby(['value']).count()
    lbp_gxt[k] = lbp_gxt[k].reindex(new_index,fill_value=0)
    lbp_gxt[k] = pd.concat([lbp_gxt[k],((lbp_gxt[k]/lbp_gxt[k].sum())*100)/100],axis=1)
    lbp_gxt[k].reset_index(level=0, inplace=True) 

for k in range (len(lbp_gyt)):
    lbp_gyt[k] = pd.DataFrame(lbp_gyt[k])
    lbp_gyt[k] = pd.melt(lbp_gyt[k])
    lbp_gyt[k] = lbp_gyt[k].groupby(['value']).count()
    lbp_gyt[k] = lbp_gyt[k].reindex(new_index,fill_value=0)
    lbp_gyt[k] = pd.concat([lbp_gyt[k],((lbp_gyt[k]/lbp_gyt[k].sum())*100)/100],axis=1)
    lbp_gyt[k].reset_index(level=0, inplace=True)    

lbptop1 = np.zeros(shape=(len(lbp_gxt),lbp_gxt[0].shape[0]*2,lbp_gxt[0].shape[1]))
lbptop = np.zeros(shape=(len(lbp_gxt),lbp_gxt[0].shape[0]*3,lbp_gxt[0].shape[1]))

for k in range (len(lbp_gxy)):
    lbptop1[k] = lbp_gxy[k].append(lbp_gxt[k])
    lbptop[k] = np.vstack((lbptop1[k],lbp_gyt[k]))
    
lbptop2=[]
for k in range (lbptop.shape[0]):  
    lbptop2.append(lbptop[k])
    lbptop2[k] = pd.DataFrame(lbptop2[k])
    lbptop2[k] = lbptop2[k].drop([0], axis=1) 
    lbptop2[k] = lbptop2[k].drop([1], axis=1)
    lbptop2[k] = lbptop2[k].values
    
lbptop = list(lbptop2)

with open('lbptop', 'wb') as f:
    pickle.dump(lbptop, f)

lbptop = joblib.load('lbptop')

#Predictions
y_pred_svm_lbptop = []
y_pred_Adab_lbptop = []

for k in range(len(lbptop)):
    lbptop[k] = lbptop[k].reshape(1,lbptop[k].shape[0]) 
    y_pred_svm_lbptop.append(clf_svm_lbptop.predict(lbptop[k])) 
    y_pred_Adab_lbptop.append(clf_Adab_lbptop.predict(lbptop[k])) 
               
#HOGHOF descriptor
exec(open("C:/Users/anton/Documents/Emp/Thesis/Python_code/gas_detect/train_classifier/raw_dataset/hof_get_flow.py").read()) 

flow = []

for k in range(vid_cells.shape[0]-1):
    for i in range(vid_cells.shape[1]):
        for j in range(vid_cells.shape[2]):
            flow.append(getFlow(vid_cells[k,i,j],vid_cells[k+1,i,j]))

hofs = []
flown = []        
for k in range(len(flow)):
    flown.append(flow[k][0])

flown=np.asarray(flown) 
   
for k in range(flown.shape[0]):  
    hofs.append(hof(flown[k], visualise=False, normalise=False))

from skimage.feature import hog
hog_g = []

for k in range(vid_cells.shape[0]-1):
    for i in range(vid_cells.shape[1]):
        for j in range(vid_cells.shape[2]):
            hog_g.append(hog(vid_cells[k,i,j], orientations=9, pixels_per_cell=(8, 8),cells_per_block=(3, 3),block_norm="L2", visualize=False))

hoghof = []
for k in range (len(hog_g)):    
    hoghof.append(np.column_stack((hog_g[k],hofs[k])))

for k in range (len(hoghof)):
    hoghof[k] = pd.DataFrame(hoghof[k])
    hoghof[k].columns=['hog','hof']

hoghofs=np.zeros(shape=(len(hoghof),hoghof[0].shape[0]*2))   

for k in range (len(hoghof)):
    hoghofs[k,:81] = hoghof[k]['hog']
    hoghofs[k,81:] = hoghof[k]['hof']
    
hoghof = []
for k in range(len(hoghofs)):
    hoghof.append(hoghofs[k])
    
y_pred_svm_hoghof = []
y_pred_Adab_hoghof = []

for k in range(len(hoghof)):
    hoghof[k] = hoghof[k].reshape(1,hoghof[k].shape[0]) 
    y_pred_svm_hoghof.append(clf_svm_hoghof.predict(hoghof[k])) 
    y_pred_Adab_hoghof.append(clf_Adab_hoghof.predict(hoghof[k]))

#Load static descriptors
###########################Static Descriptors
#image blocks

radius = 1
n_points = 8 * radius
method1 = 'nri_uniform'
method2 = 'uniform'

lbp_nri = []
lbp_uni = []
#Calculate nri and uni LBPs
for k in range (vid_cells.shape[0]):
    for i in range(vid_cells.shape[1]):
        for j in range(vid_cells.shape[2]):
            lbp_nri.append(local_binary_pattern(vid_cells[k,i,j], n_points, radius, method1))
            lbp_uni.append(local_binary_pattern(vid_cells[k,i,j], n_points, radius, method2))

#Calculate wavelets energy
coef = []
wave_enrg = []

for k in range (vid_cells.shape[0]):
    for i in range(vid_cells.shape[1]):
        for j in range(vid_cells.shape[2]):
            coef.append(pywt.wavedec2(vid_cells[k,i,j], 'db1', level=2))            

ll, lh, hl, hh = [],[],[],[]
for k in range (len(coef)):
    ll.append(coef[k][0])
    lh.append(coef[k][0][0])
    hl.append(coef[k][0][1])
    hh.append(coef[k][0][2])
    wave_enrg.append(lh[k]**2+hl[k]**2+hh[k]**2)

#Calculate edge and orientations
eoh = []
for k in range (vid_cells.shape[0]):
    for i in range(vid_cells.shape[1]):
        for j in range(vid_cells.shape[2]):
            eoh.append(hog(vid_cells[k,i,j], orientations=18, pixels_per_cell=(4, 4),cells_per_block=(1, 1),
                    block_norm="L1", visualize=False, feature_vector=True))

#Predictions                     
y_pred_svm_lbpnri = [] 
y_pred_svm_lbpuni = []
y_pred_svm_waverg = []                
y_pred_svm_eoh = []

y_pred_Adab_lbpnri = [] 
y_pred_Adab_lbpuni = []
y_pred_Adab_waverg = []                
y_pred_Adab_eoh = []

for k in range(len(lbp_nri)):
    lbp_nri[k] = lbp_nri[k].reshape(1,stepy*stepx) 
    lbp_uni[k] = lbp_uni[k].reshape(1,stepy*stepx) 
    wave_enrg[k] = wave_enrg[k].reshape(1,wave_enrg[k].shape[0])
    eoh[k] = eoh[k].reshape(1,eoh[k].shape[0])
    
    y_pred_svm_lbpnri.append(clf_svm_lbp_nri.predict(lbp_nri[k])) 
    y_pred_svm_lbpuni.append(clf_svm_lbp_uni.predict(lbp_uni[k])) 
    y_pred_svm_waverg.append(clf_svm_wav_eng.predict(wave_enrg[k]))
    y_pred_svm_eoh.append(clf_svm_eoh.predict(eoh[k]))
    
    y_pred_Adab_lbpnri.append(clf_Adab_lbp_nri.predict(lbp_nri[k])) 
    y_pred_Adab_lbpuni.append(clf_Adab_lbp_uni.predict(lbp_uni[k])) 
    y_pred_Adab_waverg.append(clf_Adab_wav_eng.predict(wave_enrg[k]))
    y_pred_Adab_eoh.append(clf_Adab_eoh.predict(eoh[k]))

#Save predictions
with open('y_pred_svm_lbptop.pkl', 'wb') as f:
    pickle.dump(y_pred_svm_lbptop, f)
with open('y_pred_svm_hoghof.pkl', 'wb') as f:
    pickle.dump(y_pred_svm_hoghof, f)
with open('y_pred_svm_lbpnri.pkl', 'wb') as f:
    pickle.dump(y_pred_svm_lbpnri, f)
with open('y_pred_svm_lbpuni.pkl', 'wb') as f:
    pickle.dump(y_pred_svm_lbpuni, f)
with open('y_pred_svm_waverg.pkl', 'wb') as f:
    pickle.dump(y_pred_svm_waverg, f)
with open('y_pred_svm_eoh.pkl', 'wb') as f:
    pickle.dump(y_pred_svm_eoh, f)
    
with open('y_pred_Adab_lbptop.pkl', 'wb') as f:
    pickle.dump(y_pred_Adab_lbptop, f)
with open('y_pred_Adab_hoghof.pkl', 'wb') as f:
    pickle.dump(y_pred_Adab_hoghof, f)
with open('y_pred_Adab_lbpnri.pkl', 'wb') as f:
    pickle.dump(y_pred_Adab_lbpnri, f)
with open('y_pred_Adab_lbpuni.pkl', 'wb') as f:
    pickle.dump(y_pred_Adab_lbpuni, f)
with open('y_pred_Adab_waverg.pkl', 'wb') as f:
    pickle.dump(y_pred_Adab_waverg, f)
with open('y_pred_Adab_eoh.pkl', 'wb') as f:
    pickle.dump(y_pred_Adab_eoh, f)    

#Load classifiers 
with open('y_pred_svm_eoh.pkl', 'rb') as f:
    y_pred_svm_eoh = pickle.load(f)
with open('y_pred_svm_lbptop.pkl', 'rb') as f:
    y_pred_svm_lbptop = pickle.load(f)
with open('y_pred_svm_hoghof.pkl', 'rb') as f:
    y_pred_svm_hoghof = pickle.load(f)
with open('y_pred_svm_lbpnri.pkl', 'rb') as f:
    y_pred_svm_lbpnri = pickle.load(f)
with open('y_pred_svm_lbpuni.pkl', 'rb') as f:
    y_pred_svm_lbpuni = pickle.load(f)
with open('y_pred_svm_waverg.pkl', 'rb') as f:
    y_pred_svm_waverg = pickle.load(f)

with open('y_pred_Adab_eoh.pkl', 'rb') as f:
    y_pred_Adab_eoh = pickle.load(f)
with open('y_pred_Adab_lbptop.pkl', 'rb') as f:
    y_pred_Adab_lbptop = pickle.load(f)
with open('y_pred_Adab_hoghof.pkl', 'rb') as f:
    y_pred_Adab_hoghof = pickle.load(f)
with open('y_pred_Adab_lbpnri.pkl', 'rb') as f:
    y_pred_Adab_lbpnri = pickle.load(f)
with open('y_pred_Adab_lbpuni.pkl', 'rb') as f:
    y_pred_Adab_lbpuni = pickle.load(f)
with open('y_pred_Adab_waverg.pkl', 'rb') as f:
    y_pred_Adab_waverg = pickle.load(f)

#Delete elements for space
del flow,flown,hofs,hog_g,lists,myarray
del eoh,hh,hl,wave_enrg,lbp_nri,lbp_uni

#Mix descripotrs SVM
pred = []
for k in range(len(y_pred_svm_hoghof)):    
    if ((y_pred_svm_lbpnri==1 and y_pred_svm_lbpuni==1 and y_pred_svm_lbptop==1 ) or (y_pred_svm_hoghof[k]==1 and y_pred_svm_waverg==1 and y_pred_svm_eoh[k]==1)):
        pred.append(np.ones((stepx,stepy)))
    else:
        pred.append(np.zeros((stepx,stepy)))    

myarray = np.asarray(pred)
pred_cells = np.zeros(shape=(vid.shape[0]*vid.shape[1]/stepy*vid.shape[2]/stepx,stepy,stepx))

for k in range(myarray.shape[0]):
    pred_cells[k] = myarray[k]

pred_cells = pred_cells.reshape(vid_cells.shape)
        
exec(open("C:/Users/anton/Documents/Emp/Thesis/Python_code/gas_detect/train_classifier/raw_dataset/conc_im.py").read()) 
preds = np.zeros(shape=(vid.shape))

for k in range(pred_cells.shape[0]):
    preds[k] = conc_im(pred_cells[k])

preds=preds*100
skvideo.io.vwrite('svm_predictions.mp4',preds)

#Mix descripotrs Adaboost
pred = []
for k in range(len(y_pred_Adab_hoghof)):    
    if ((y_pred_Adab_lbpnri==1 and y_pred_Adab_lbpuni==1 and y_pred_Adab_waverg==1 and y_pred_Adab_eoh[k]==1) or (y_pred_Adab_hoghof[k]==1 or y_pred_Adab_lbptop==1)):
        pred.append(np.ones((stepx,stepy)))
    else:
        pred.append(np.zeros((stepx,stepy)))    

myarray = np.asarray(pred)
pred_cells = np.zeros(shape=(vid.shape[0]*vid.shape[1]/stepy*vid.shape[2]/stepx,stepy,stepx))

for k in range(myarray.shape[0]):
    pred_cells[k] = myarray[k]

pred_cells = pred_cells.reshape(vid_cells.shape)
        
exec(open("C:/Users/anton/Documents/Emp/Thesis/Python_code/gas_detect/train_classifier/raw_dataset/conc_im.py").read()) 
preds = np.zeros(shape=(vid.shape))

for k in range(pred_cells.shape[0]):
    preds[k] = conc_im(pred_cells[k])

preds=preds*100
skvideo.io.vwrite('Adab_predictions.mp4',preds)

#Play video 
preds_svm = skvideo.io.vread("svm_predictions.mp4",outputdict={"-pix_fmt": "gray"})[:, :, :, 0]
preds_Adab = skvideo.io.vread("Adab_predictions.mp4",outputdict={"-pix_fmt": "gray"})[:, :, :, 0]


color = (255,0,0)

#Test to image
vids = skvideo.io.vread("propane/gas_propane_r.mp4",outputdict={"-pix_fmt": "gray"})[:, :, :, 0]

img = vids[3000]
predss = preds[3000]
color = (255,0,0)
clone = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

plt.figure()
for i in range(0,img.shape[0],stepy):
    for j in range(0,img.shape[1],stepx):
        if np.mean(predss[i:i+stepy,j:j+stepx])==100:
            cv2.rectangle(clone, (i, j), (i + stepy, j + stepx), color, 2)
            
        plt.imshow(clone,cmap='gray')             

#Test to video
cap = cv2.VideoCapture("./propane/gas_propane_r.mp4")
cap2 = cv2.VideoCapture("./svm_predictions.mp4")      

size = (int(cap.get(3)),int(cap.get(4))) 

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('gas_det_vid.mp4', fourcc, 25.0, size, isColor=1)


while(cap.isOpened()):
    ret,frame = cap.read()
    pret,preds_f = cap2.read()
    
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    preds_f = cv2.cvtColor(preds_f, cv2.COLOR_RGB2GRAY)
    clone = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
    if ret == True:
        for i in range(0,frame.shape[0],stepy):
            for j in range(0,frame.shape[1],stepx):
                if (np.mean(preds_f[i:i+stepy,j:j+stepx]) == 97):
                    cv2.rectangle(clone, (i, j), (i + stepy, j + stepx), color, 2)
                    cv2.imshow('frame',clone)
                else:
                     cv2.imshow('frame',clone) 
               
                #out.write(clone)
   
           
    else:
        break

    if cv2.waitKey(10)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
out.release()

