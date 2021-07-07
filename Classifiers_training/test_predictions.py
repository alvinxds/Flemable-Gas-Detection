# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 18:41:24 2019

@author: anton
"""
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import skvideo.io
import pickle

#Change current working directory
imagePath = "C:/Users/anton/Documents/Emp/Thesis/Python_code/gas_detect/train_classifier/raw_dataset/"
os.chdir(imagePath)

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

##########Method 1 only static descriptors
print y_pred_svm_lbptop.count(1)
#Mix descripotrs SVM
#Check number of preds
pred = []
for k in range(len(y_pred_svm_lbpnri)):    
    if (((y_pred_svm_lbpnri[k]==1 and y_pred_svm_eoh[k]==1 and y_pred_svm_lbpuni[k]==1))):
        pred.append(1)
    else:
        pred.append(0)

print pred.count(1)

for k in range(len(pred)-2):    
    if (pred[k]==1 and pred[k+1]==0 and pred[k+2]==0):
        pred[k]=0
    elif (pred[k]==1 and pred[k+1]==0 and pred[k+2]==1):
        pred[k+1]==1

print pred.count(1)

preds = []
for k in range(len(pred)):    
    if (pred[k]==1):
        preds.append(np.ones((stepx,stepy)))
    else:
        preds.append(np.zeros((stepx,stepy)))    

myarray = np.asarray(preds)
pred_cells = np.zeros(shape=(vid.shape[0]*vid.shape[1]/stepy*vid.shape[2]/stepx,stepy,stepx))

for k in range(myarray.shape[0]):
    pred_cells[k] = myarray[k]

pred_cells = pred_cells.reshape(vid_cells.shape)
        
exec(open("C:/Users/anton/Documents/Emp/Thesis/Python_code/gas_detect/train_classifier/raw_dataset/conc_im.py").read()) 
preds = np.zeros(shape=(vid.shape))

for k in range(pred_cells.shape[0]):
    preds[k] = conc_im(pred_cells[k])

preds=preds*100
skvideo.io.vwrite('svm_static_predictions.mp4',preds)

#Mix descripotrs Adaboost
print y_pred_Adab_hoghof.count(1)
#Mix descripotrs SVM
#Check number of preds
pred = []
for k in range(len(y_pred_svm_lbpnri)):    
    if (((y_pred_Adab_lbpnri[k]==1 and y_pred_Adab_lbpuni[k]==1 and y_pred_Adab_eoh[k]==1 ))):
        pred.append(1)
    else:
        pred.append(0)

print pred.count(1)

for k in range(len(pred)-2):    
    if (pred[k]==1 and pred[k+1]==0 and pred[k+2]==0):
        pred[k]=0
    elif (pred[k]==1 and pred[k+1]==0 and pred[k+2]==1):
        pred[k+1]==1

print pred.count(1)

preds = []
for k in range(len(pred)):    
    if (pred[k]==1):
        preds.append(np.ones((stepx,stepy)))
    else:
        preds.append(np.zeros((stepx,stepy)))    

myarray = np.asarray(preds)
pred_cells = np.zeros(shape=(vid.shape[0]*vid.shape[1]/stepy*vid.shape[2]/stepx,stepy,stepx))

for k in range(myarray.shape[0]):
    pred_cells[k] = myarray[k]

pred_cells = pred_cells.reshape(vid_cells.shape)
        
exec(open("C:/Users/anton/Documents/Emp/Thesis/Python_code/gas_detect/train_classifier/raw_dataset/conc_im.py").read()) 
preds = np.zeros(shape=(vid.shape))

for k in range(pred_cells.shape[0]):
    preds[k] = conc_im(pred_cells[k])

preds=preds*100
skvideo.io.vwrite('Adab_static_predictions.mp4',preds)          

#Test to video
cap = cv2.VideoCapture("./propane/gas_propane_r.mp4")
cap2 = cv2.VideoCapture("./Adab_static_predictions.mp4")      

size = (int(cap.get(3)),int(cap.get(4))) 

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('gasdet_svm_static_desc.mp4', fourcc, 25.0, size, isColor=1)
color = (255,0,0)

while(cap.isOpened()):
    ret, frame = cap.read()
    pret, preds_f = cap2.read()
    
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    preds_f = cv2.cvtColor(preds_f, cv2.COLOR_RGB2GRAY)
    clone = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
    
    if ret == True:
        for i in range(0,frame.shape[0],stepy):
            for j in range(0,frame.shape[1],stepx):
                if (np.mean(preds_f[i:i+stepy,j:j+stepx]) == 97):
                    cv2.rectangle(clone, (i, j), (i + stepy, j + stepx), color, 2)
                    cv2.imshow('frame',clone)
                    #out.write(clone)
                else:
                     cv2.imshow('frame',clone) 
                     #out.write(clone)
          
    else:
        break

    if cv2.waitKey(5)&0xFF==ord('q'):
        break

cap.release()
cap2.release()
cv2.destroyAllWindows()
out.release()

##########Method 2 only dynamic descriptors

#Mix descripotrs SVM
pred = []
for k in range(len(y_pred_svm_lbptop)):    
    if (((y_pred_svm_lbptop[k]==1 and y_pred_svm_hoghof[k]==1))):
        pred.append(1)
    else:
        pred.append(0)

print pred.count(1)

for k in range(len(pred)-2):    
    if (pred[k]==1 and pred[k+1]==0 and pred[k+2]==0):
        pred[k]=0
    elif (pred[k]==1 and pred[k+1]==0 and pred[k+2]==1):
        pred[k+1]==1

print pred.count(1)

preds = []
for k in range(len(pred)):    
    if (pred[k]==1):
        preds.append(np.ones((stepx,stepy)))
    else:
        preds.append(np.zeros((stepx,stepy)))    

myarray = np.asarray(preds)
pred_cells = np.zeros(shape=(vid.shape[0]*vid.shape[1]/stepy*vid.shape[2]/stepx,stepy,stepx))

for k in range(myarray.shape[0]):
    pred_cells[k] = myarray[k]

pred_cells = pred_cells.reshape(vid_cells.shape)
        
exec(open("C:/Users/anton/Documents/Emp/Thesis/Python_code/gas_detect/train_classifier/raw_dataset/conc_im.py").read()) 
preds = np.zeros(shape=(vid.shape))

for k in range(pred_cells.shape[0]):
    preds[k] = conc_im(pred_cells[k])

preds=preds*100
skvideo.io.vwrite('svm_dynamic_predictions.mp4',preds)

#Mix descripotrs Adaboost
pred = []
for k in range(len(y_pred_Adab_lbptop)):    
    if (((y_pred_Adab_lbptop[k]==1 and y_pred_Adab_hoghof[k]==1))):
        pred.append(1)
    else:
        pred.append(0)

print pred.count(1)

for k in range(len(pred)-2):    
    if (pred[k]==1 and pred[k+1]==0 and pred[k+2]==0):
        pred[k]=0
    elif (pred[k]==1 and pred[k+1]==0 and pred[k+2]==1):
        pred[k+1]==1

print pred.count(1)

preds = []
for k in range(len(pred)):    
    if (pred[k]==1):
        preds.append(np.ones((stepx,stepy)))
    else:
        preds.append(np.zeros((stepx,stepy)))    

myarray = np.asarray(preds)
pred_cells = np.zeros(shape=(vid.shape[0]*vid.shape[1]/stepy*vid.shape[2]/stepx,stepy,stepx))

for k in range(myarray.shape[0]):
    pred_cells[k] = myarray[k]

pred_cells = pred_cells.reshape(vid_cells.shape)
        
exec(open("C:/Users/anton/Documents/Emp/Thesis/Python_code/gas_detect/train_classifier/raw_dataset/conc_im.py").read()) 
preds = np.zeros(shape=(vid.shape))

for k in range(pred_cells.shape[0]):
    preds[k] = conc_im(pred_cells[k])

preds=preds*100
skvideo.io.vwrite('Adab_dynamic_predictions.mp4',preds)

#Test to video
cap = cv2.VideoCapture("./propane/gas_propane_r.mp4")
cap2 = cv2.VideoCapture("./svm_dynamic_predictions.mp4")      

size = (int(cap.get(3)),int(cap.get(4))) 

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('gas_det_vid.mp4', fourcc, 25.0, size, isColor=1)
color = (255,0,0)

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

##########Method 3 mix descriptors

#Mix descripotrs SVM
pred = []
for k in range(len(y_pred_svm_lbptop)):    
    if ((y_pred_svm_lbpnri[k]==1 and y_pred_svm_eoh[k]==1 and y_pred_svm_lbpuni[k]==1) or (y_pred_svm_lbptop[k]==1 and y_pred_svm_hoghof[k]==1)):
        pred.append(1)
    else:
        pred.append(0)

print pred.count(1)

for k in range(len(pred)-2):    
    if (pred[k]==1 and pred[k+1]==0 and pred[k+2]==0):
        pred[k]=0
    elif (pred[k]==1 and pred[k+1]==0 and pred[k+2]==1):
        pred[k+1]==1

print pred.count(1)

preds = []
for k in range(len(pred)):    
    if (pred[k]==1):
        preds.append(np.ones((stepx,stepy)))
    else:
        preds.append(np.zeros((stepx,stepy)))    

myarray = np.asarray(preds)
pred_cells = np.zeros(shape=(vid.shape[0]*vid.shape[1]/stepy*vid.shape[2]/stepx,stepy,stepx))

for k in range(myarray.shape[0]):
    pred_cells[k] = myarray[k]

pred_cells = pred_cells.reshape(vid_cells.shape)
        
exec(open("C:/Users/anton/Documents/Emp/Thesis/Python_code/gas_detect/train_classifier/raw_dataset/conc_im.py").read()) 
preds = np.zeros(shape=(vid.shape))

for k in range(pred_cells.shape[0]):
    preds[k] = conc_im(pred_cells[k])

preds=preds*100
skvideo.io.vwrite('svm_mix_predictions.mp4',preds)

#Mix descripotrs Adaboost
pred = []
for k in range(len(y_pred_Adab_lbptop)):    
    if ((y_pred_Adab_lbpnri[k]==1 and y_pred_Adab_eoh[k]==1 and y_pred_Adab_lbpuni[k]==1 and y_pred_Adab_waverg[k]==1) or (y_pred_Adab_lbptop[k]==1 and y_pred_Adab_hoghof[k]==1)):
        pred.append(1)
    else:
        pred.append(0)

print pred.count(1)

for k in range(len(pred)-2):    
    if (pred[k]==1 and pred[k+1]==0 and pred[k+2]==0):
        pred[k]=0
    elif (pred[k]==1 and pred[k+1]==0 and pred[k+2]==1):
        pred[k+1]==1

print pred.count(1)

preds = []
for k in range(len(pred)):    
    if (pred[k]==1):
        preds.append(np.ones((stepx,stepy)))
    else:
        preds.append(np.zeros((stepx,stepy)))    

myarray = np.asarray(preds)
pred_cells = np.zeros(shape=(vid.shape[0]*vid.shape[1]/stepy*vid.shape[2]/stepx,stepy,stepx))

for k in range(myarray.shape[0]):
    pred_cells[k] = myarray[k]

pred_cells = pred_cells.reshape(vid_cells.shape)
        
exec(open("C:/Users/anton/Documents/Emp/Thesis/Python_code/gas_detect/train_classifier/raw_dataset/conc_im.py").read()) 
preds = np.zeros(shape=(vid.shape))

for k in range(pred_cells.shape[0]):
    preds[k] = conc_im(pred_cells[k])

preds=preds*100
skvideo.io.vwrite('Adab_mix_predictions.mp4',preds)

#Test to video
cap = cv2.VideoCapture("./propane/gas_propane_r.mp4")
cap2 = cv2.VideoCapture("./Adab_mix_predictions.mp4")      

size = (int(cap.get(3)),int(cap.get(4))) 

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('gas_det_Adab_mix.mp4', fourcc, 25.0, size, isColor=1)


while(cap.isOpened()):
    ret, frame = cap.read()
    pret,preds_f = cap2.read()
    
    if ret == True:
        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        preds_f = cv2.cvtColor(preds_f, cv2.COLOR_RGB2GRAY)
        clone = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
        
        out.write(clone)
        
        for i in range(0,frame.shape[0],stepy):
            for j in range(0,frame.shape[1],stepx):
                if (np.mean(preds_f[i:i+stepy,j:j+stepx]) == 97):
                    cv2.rectangle(clone, (i, j), (i + stepy, j + stepx), color, 2)
                    #cv2.imshow('frame',clone)
                    out.write(clone)
                #else:
                     #cv2.imshow('frame',clone) 
                     #out.write(clone)
               
        #out.write(clone)
   
           
    else:
        break

    if cv2.waitKey(80)&0xFF==ord('q'):
        break

cap.release()
cap2.release()
cv2.destroyAllWindows()
out.release()