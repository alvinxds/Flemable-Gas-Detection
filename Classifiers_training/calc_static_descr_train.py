# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:22:34 2019

@author: anton
"""

import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from pylab import rcParams
import glob
import pandas as pd
import ggplot as gp
import skvideo.io
from sklearn.externals import joblib
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from pandas import *

#Change current working directory
imagePath = "C:/Users/anton/Documents/Emp/Thesis/Python_code/gas_detect/train_classifier/raw_dataset/patches/"
os.chdir(imagePath)

rcParams['figure.figsize'] = 6, 4

#Open patches for calculating static descriptors
#First load positive 'gas' patches
gas_img = [cv2.imread(file) for file in glob.glob("./static_dataset/gas_n/*.png")]
for i in range (len(gas_img)):
    gas_img[i] = cv2.cvtColor(gas_img[i],cv2.COLOR_BGR2GRAY)
    
#Then load negative 'non-gas' patches
nongas_img = [cv2.imread(file) for file in glob.glob("./static_dataset/nongas_n/*.png")]
for i in range (len(nongas_img)):
    nongas_img[i] = cv2.cvtColor(nongas_img[i],cv2.COLOR_BGR2GRAY)


#Divide patches to blocks to calculate descriptors

###########################Static Descriptors
#Gas blocks
stepx = 24
stepy = 24
lists = []

for k in range (len(gas_img)):
    for i in range(0,gas_img[0].shape[0],stepy):
        for j in range(0,gas_img[0].shape[1],stepx):
            lists.append(gas_img[k][i:i+stepy,j:j+stepx])
        
myarray = np.asarray(lists)
gas_blocks = myarray.reshape(len(gas_img),2,2,stepy,stepx)

#Nongas blocks
lists = []
for k in range (len(nongas_img)):
    for i in range(0,nongas_img[0].shape[0],stepy):
        for j in range(0,nongas_img[0].shape[1],stepx):
            lists.append(nongas_img[k][i:i+stepy,j:j+stepx])

myarray = np.asarray(lists)            
nongas_blocks = myarray.reshape(len(nongas_img),2,2,stepy,stepx)

#Calculate LBP descriptors
#First train classifier with nri_uniform lbp
radius = 1
n_points = 8 * radius
image = gas_img[0]
method1 = 'nri_uniform'
method2 = 'uniform'
exec(open("C:/Users/anton/Documents/Emp/Thesis/Python_code/gas_detect/train_classifier/raw_dataset/load_data_multi_samples.py").read())  

#Calculate LBPs for positive patches-blocks
lbp_gas_nri = np.zeros(shape=gas_blocks.shape)
lbp_gas_uni = np.zeros(shape=gas_blocks.shape)
for k in range (gas_blocks.shape[0]):
    for i in range (gas_blocks.shape[1]):
        for j in range (gas_blocks.shape[2]):
            lbp_gas_nri[k,i,j] = local_binary_pattern(gas_blocks[k,i,j], n_points, radius, method1)
            lbp_gas_uni[k,i,j] = local_binary_pattern(gas_blocks[k,i,j], n_points, radius, method2)

lbp_gas_nri_desc = lbp_gas_nri.reshape(lbp_gas_nri.shape[0],2,2,stepx*stepy)
lbp_gas_uni_desc = lbp_gas_uni.reshape(lbp_gas_uni.shape[0],2,2,stepx*stepy)

#Calculate LBPs for negative patches-blocks
lbp_nongas_nri = np.zeros(shape=nongas_blocks.shape)
lbp_nongas_uni = np.zeros(shape=nongas_blocks.shape)
for k in range (nongas_blocks.shape[0]):
    for i in range (nongas_blocks.shape[1]):
        for j in range (nongas_blocks.shape[2]):
            lbp_nongas_nri[k,i,j] = local_binary_pattern(nongas_blocks[k,i,j], n_points, radius, method1) 
            lbp_nongas_uni[k,i,j] = local_binary_pattern(nongas_blocks[k,i,j], n_points, radius, method2)

lbp_nongas_nri_desc = lbp_nongas_nri.reshape(lbp_nongas_nri.shape[0],2,2,stepx*stepy)
lbp_nongas_uni_desc = lbp_nongas_uni.reshape(lbp_nongas_uni.shape[0],2,2,stepx*stepy)

#Create lbp-nri classifier
samples=[]
labels=[]
for k in range (lbp_gas_nri_desc.shape[0]):
    for i in range (lbp_gas_nri_desc.shape[1]):
        for j in range (lbp_gas_nri_desc.shape[2]):
            samples.append(lbp_gas_nri_desc[k,i,j])
            labels.append(1)
for k in range (lbp_nongas_nri_desc.shape[0]):
    for i in range (lbp_nongas_nri_desc.shape[1]):
        for j in range (lbp_nongas_nri_desc.shape[2]):
            samples.append(lbp_nongas_nri_desc[k,i,j])
            labels.append(0)
            
# Convert objects to Numpy Objects
samples = np.float32(samples)
labels = np.array(labels)          

# assign a random permutation
rand = np.random.RandomState(321)
shuffle = rand.permutation(len(samples))
samples = samples[shuffle]
labels = labels[shuffle]    
   
ratio = 0.7
classes = 2

train_set, train_lab, other_set,other_lab = load_data_multi_samples(samples, labels, ratio, classes)
# train SVM classifier
print '... training Linear-SVM'                            
clf_lin_pca = svm.LinearSVC()
clf_lin_pca.fit(train_set, train_lab)  
# validate SVM
print '... testing Linear-SVM'
predictions = clf_lin_pca.predict(other_set) 
# calculate errors
errors = np.where(predictions != other_lab)
lin_pca_er = float(errors[0].shape[0])/other_lab.shape[0]  
print '... Linear-SVM error:%f'%(100*lin_pca_er) 
joblib.dump(clf_lin_pca, 'lbp_nri_svm_classif.pkl')  

# train Adaboost classifier
print '... training AdaBoost-Classifier'                            
clf_lin_pca = AdaBoostClassifier(n_estimators=100, random_state=0)
clf_lin_pca.fit(train_set, train_lab)  
# validate SVM
print '... testing AdaBoost-Classifier'
predictions = clf_lin_pca.predict(other_set) 
# calculate errors
errors = np.where(predictions != other_lab)
lin_pca_er = float(errors[0].shape[0])/other_lab.shape[0]  
print '... AdaBoost-Classifier error:%f'%(100*lin_pca_er) 
joblib.dump(clf_lin_pca, 'lbp_nri_Adaboost_classif.pkl') 

#Create lbp-uni classifier
samples=[]
labels=[]
for k in range (lbp_gas_uni_desc.shape[0]):
    for i in range (lbp_gas_uni_desc.shape[1]):
        for j in range (lbp_gas_uni_desc.shape[2]):
            samples.append(lbp_gas_uni_desc[k,i,j])
            labels.append(1)
for k in range (lbp_nongas_uni_desc.shape[0]):
    for i in range (lbp_nongas_uni_desc.shape[1]):
        for j in range (lbp_nongas_uni_desc.shape[2]):
            samples.append(lbp_nongas_uni_desc[k,i,j])
            labels.append(0)
            
# Convert objects to Numpy Objects
samples = np.float32(samples)
labels = np.array(labels)          

# assign a random permutation
rand = np.random.RandomState(321)
shuffle = rand.permutation(len(samples))
samples = samples[shuffle]
labels = labels[shuffle]    
   
ratio = 0.7
classes = 2

train_set, train_lab, other_set,other_lab = load_data_multi_samples(samples, labels, ratio, classes)
# train SVM classifier
print '... training Linear-SVM'                            
clf_lin_pca = svm.LinearSVC()
clf_lin_pca.fit(train_set, train_lab)  
# validate SVM
print '... testing Linear-SVM'
predictions = clf_lin_pca.predict(other_set) 
# calculate errors
errors = np.where(predictions != other_lab)
lin_pca_er = float(errors[0].shape[0])/other_lab.shape[0]  
print '... Linear-SVM error:%f'%(100*lin_pca_er) 
joblib.dump(clf_lin_pca, 'lbp_uni_svm_classif.pkl')  

# train Adaboost classifier
print '... training AdaBoost-Classifier'                            
clf_lin_pca = AdaBoostClassifier(n_estimators=100, random_state=0)
clf_lin_pca.fit(train_set, train_lab)  
# validate SVM
print '... testing AdaBoost-Classifier'
predictions = clf_lin_pca.predict(other_set) 
# calculate errors
errors = np.where(predictions != other_lab)
lin_pca_er = float(errors[0].shape[0])/other_lab.shape[0]  
print '... AdaBoost-Classifier error:%f'%(100*lin_pca_er) 
joblib.dump(clf_lin_pca, 'lbp_uni_Adaboost_classif.pkl')  

#Create histograms for selected blocks
lbp_g = lbp_gas_nri[1210,1,1]       
lbp_ng = lbp_nongas_nri[0,1,1]
    
lbp_g = pd.DataFrame(lbp_g)
lbp_g = pd.melt(lbp_g)
lbp_ng = pd.DataFrame(lbp_ng)
lbp_ng = pd.melt(lbp_ng)

#LBP uniform (bins=59)
lbp_g = lbp_g.groupby(['value']).count()
new_index = range(0, 59)
lbp_g = lbp_g.reindex(new_index,fill_value=0)
lbp_g = pd.concat([lbp_g,((lbp_g/lbp_g.sum())*100)/100],axis=1)
lbp_g.columns=['Count','Normalized Occurrence']
lbp_g.reset_index(level=0, inplace=True)
lbp_g[['value']] = lbp_g[['value']]+1

g_lbpg_riu = gp.ggplot(gp.aes(x='value',weight='Normalized Occurrence'), data = lbp_g) + \
              gp.geom_bar(stat='identity',fill ="darkblue",alpha=0.6)+ \
              gp.ggtitle("LBP Uniform Index Gas") + \
              gp.xlab("LBP Index") + \
              gp.ylab("Normalized Occurrence") + \
              gp.scale_y_continuous(limits=(0,0.2)) + \
              gp.scale_x_continuous(limits=(-1,60), breaks=(1,10,20,30,40,50,59),labels=[1,10,20,30,40,50,59]) 

lbp_ng = lbp_ng.groupby(['value']).count()
new_index = range(0, 59)
lbp_ng = lbp_ng.reindex(new_index,fill_value=0)
lbp_ng = pd.concat([lbp_ng,((lbp_ng/lbp_ng.sum())*100)/100],axis=1)
lbp_ng.columns=['Count','Normalized Occurrence']
lbp_ng.reset_index(level=0, inplace=True)
lbp_ng[['value']] = lbp_ng[['value']]+1

g_lbpng_riu = gp.ggplot(gp.aes(x='value',weight='Normalized Occurrence'), data = lbp_ng) + \
              gp.geom_bar(stat='identity',fill ="darkblue",alpha=0.6)+ \
              gp.ggtitle("LBP Uniform Index Non-Gas") + \
              gp.xlab("LBP Index") + \
              gp.ylab("Normalized Occurrence") + \
              gp.scale_y_continuous(limits=(0,0.2)) + \
              gp.scale_x_continuous(limits=(-1,60), breaks=(1,10,20,30,40,50,59),labels=[1,10,20,30,40,50,59]) 

plt.figure()              
g_lbpg_riu.show()
plt.figure()
g_lbpng_riu.show()  

exec(open("C:/Users/anton/Documents/Emp/Thesis/Python_code/gas_detect/train_classifier/raw_dataset/conc_im.py").read()) 

a = conc_im(gas_blocks[1210])
    

#LBP rotaion-invariant uniform (bins=10)
lbp_griu = lbp_gas_uni[1210,1,1]       
lbp_ngriu = lbp_nongas_uni[0,1,1]
    
lbp_griu = pd.DataFrame(lbp_griu)
lbp_griu = pd.melt(lbp_griu)
lbp_ngriu = pd.DataFrame(lbp_ngriu)
lbp_ngriu = pd.melt(lbp_ngriu)


lbp_griu = lbp_griu.groupby(['value']).count()
lbp_griu = pd.concat([lbp_griu,((lbp_griu/lbp_griu.sum())*100)/100],axis=1)
lbp_griu.columns=['Count','Normalized Occurrence']
lbp_griu.reset_index(level=0, inplace=True)
lbp_griu[['value']] = lbp_griu[['value']]+1
              
g_lbpriu = gp.ggplot(gp.aes(x='value',weight='Normalized Occurrence'), data = lbp_griu) + \
            gp.geom_bar(stat='identity',fill ="darkblue",alpha=0.6) + \
            gp.ggtitle("LBP Rotation-Invariant Uniform Index Gas") + \
            gp.xlab("LBP Index") + \
            gp.ylab("Normalized Occurrence") + \
            gp.scale_y_continuous(limits=(0,0.45))
            
lbp_ngriu = lbp_ngriu.groupby(['value']).count()
lbp_ngriu = pd.concat([lbp_ngriu,((lbp_ngriu/lbp_ngriu.sum())*100)/100],axis=1)
lbp_ngriu.columns=['Count','Normalized Occurrence']
lbp_ngriu.reset_index(level=0, inplace=True)
lbp_ngriu[['value']] = lbp_ngriu[['value']]+1
              
ng_lbpriu = gp.ggplot(gp.aes(x='value',weight='Normalized Occurrence'), data = lbp_ngriu) + \
            gp.geom_bar(stat='identity',fill ="darkblue",alpha=0.6) + \
            gp.ggtitle("LBP Rotation-Invariant Uniform Index Non-Gas") + \
            gp.xlab("LBP Index") + \
            gp.ylab("Normalized Occurrence") + \
            gp.scale_y_continuous(limits=(0,0.45))

plt.figure()
g_lbpriu.show()
plt.figure()
ng_lbpriu.show()       
   

#Calculate Discrete Wavelet Transform
#Train classifier wavelet energy
import pywt
a=20
b=10

coef_g=[]
for k in range (gas_blocks.shape[0]):
    for i in range (gas_blocks.shape[1]):
        for j in range (gas_blocks.shape[2]):
            coef_g.append(pywt.wavedec2(gas_blocks[k,i,j], 'db1', level=2))

ll, lh, hl, hh,wv_energ_g = [],[],[],[],[]
for k in range (len(coef_g)):
    ll.append(coef_g[k][0])
    lh.append(coef_g[k][0][0])
    hl.append(coef_g[k][0][1])
    hh.append(coef_g[k][0][2])
    wv_energ_g.append(lh[k]**2+hl[k]**2+hh[k]**2)
    
coef_ng=[]
for k in range (nongas_blocks.shape[0]):
    for i in range (nongas_blocks.shape[1]):
        for j in range (nongas_blocks.shape[2]):
            coef_ng.append(pywt.wavedec2(nongas_blocks[k,i,j], 'db1', level=2))

ll, lh, hl, hh,wv_energ_ng = [],[],[],[],[]
for k in range (len(coef_ng)):
    ll.append(coef_ng[k][0])
    lh.append(coef_ng[k][0][0])
    hl.append(coef_ng[k][0][1])
    hh.append(coef_ng[k][0][2])
    wv_energ_ng.append(lh[k]**2+hl[k]**2+hh[k]**2)
    
#Create wavelet energy classifier
samples=[]
labels=[]
for k in range (len(wv_energ_g)):
    samples.append(wv_energ_g[k])
    labels.append(1)
for k in range (len(wv_energ_ng)):
    samples.append(wv_energ_ng[k])
    labels.append(0)
            
# Convert objects to Numpy Objects
samples = np.float32(samples)
labels = np.array(labels)          

# assign a random permutation
rand = np.random.RandomState(321)
shuffle = rand.permutation(len(samples))
samples = samples[shuffle]
labels = labels[shuffle]    
   
ratio = 0.75
classes = 2

train_set, train_lab, other_set,other_lab = load_data_multi_samples(samples, labels, ratio, classes)
# train SVM classifier
print '... training Linear-SVM'                            
clf_lin_pca = svm.LinearSVC()
clf_lin_pca.fit(train_set, train_lab)  
# validate SVM
print '... testing Linear-SVM'
predictions = clf_lin_pca.predict(other_set) 
# calculate errors
errors = np.where(predictions != other_lab)
lin_pca_er = float(errors[0].shape[0])/other_lab.shape[0]  
print '... Linear-SVM error:%f'%(100*lin_pca_er) 
joblib.dump(clf_lin_pca, 'wavelet_energ_svm_classif.pkl')     

# train Adaboost classifier
print '... training AdaBoost-Classifier'                            
clf_lin_pca = AdaBoostClassifier(n_estimators=100, random_state=0)
clf_lin_pca.fit(train_set, train_lab)  
# validate SVM
print '... testing AdaBoost-Classifier'
predictions = clf_lin_pca.predict(other_set) 
# calculate errors
errors = np.where(predictions != other_lab)
lin_pca_er = float(errors[0].shape[0])/other_lab.shape[0]  
print '... AdaBoost-Classifier error:%f'%(100*lin_pca_er) 
joblib.dump(clf_lin_pca, 'wavelet_energ_Adaboost_classif.pkl')     

#Plot histogram
fig = plt.figure()
plt.suptitle(r'Distribution of Wavelet Energy',fontsize="x-large")
plt.subplots_adjust(hspace=0.7)
plt.subplot(2, 2, 1)
plt.xlabel('Wavelet Energy')
plt.ylabel('Occurance')
plt.grid()
coeffs2 = pywt.wavedec2(gas_img[a], 'db1', level=2)
LL, (LH, HL, HH), (LH2, HL2, HH2) = coeffs2
E = LH**2 + HL**2 + HH**2
plt.hist(E.flatten(),color="gray",lw=0.5,alpha=0.8)
plt.subplot(2, 2, 2)
plt.xlabel('Wavelet Energy')
plt.ylabel('Occurance')
plt.grid()
coeffs = pywt.wavedec2(nongas_img[b], 'db1', level=2)
LLn, (LHn, HLn, HHn), (LH2n, HL2n, HH2n) = coeffs
En = LHn**2 + HLn**2 + HHn**2
plt.hist(En.flatten(),color="gray",lw=0.5,alpha=0.8)
plt.subplot(2, 2, 3)
plt.xticks([])
plt.yticks([])
plt.imshow(gas_img[1],cmap='gray')
plt.subplot(2, 2, 4)
plt.xticks([])
plt.yticks([])
plt.imshow(nongas_img[b],cmap='gray')

fig = plt.figure()
plt.suptitle(r'Wavelet coefficients Distribution',fontsize="x-large")
plt.subplots_adjust(hspace=0.7)
plt.subplot(2, 2, 1)
plt.xlabel('Coefficient Value')
plt.ylabel('Occurance')
plt.grid()
plt.ylim(0,100)
plt.xlim(-250,250)
coeffs = pywt.wavedec2(gas_img[100], 'db1', level=2)
LL, (LHg, HL, HH), (LH2, HL2, HH2) = coeffs
wave_h=LHg.flatten()
(hist,bins,patches) = plt.hist(wave_h,color="gray",lw=0.5,alpha=0.8)
plt.subplot(2, 2, 2)
plt.xlabel('Coefficient Value')
plt.ylabel('Occurance')
plt.grid()
plt.ylim(0,100)
plt.xlim(-250,250)
coeffs2 = pywt.wavedec2(nongas_img[10], 'db1', level=2)
LL, (LHng, HL, HH), (LH2, HL2, HH2) = coeffs2
wave_nh=LHng.flatten()
(hist,bins,patches) = plt.hist(wave_nh,color="gray",lw=0.5,alpha=0.8)
plt.subplot(2, 2, 3)
plt.xticks([])
plt.yticks([])
plt.imshow(gas_img[100],cmap='gray')
plt.subplot(2, 2, 4)
plt.xticks([])
plt.yticks([])
plt.imshow(nongas_img[10],cmap='gray')

#Calculate Edge Orientation Histogram Descriptor
from skimage.feature import hog

eoh_gas = np.zeros(shape=(gas_blocks.shape[0],2,2,648))
for k in range (gas_blocks.shape[0]):
    for i in range (gas_blocks.shape[1]):
        for j in range (gas_blocks.shape[2]):
            eoh_gas[k,i,j] = hog(gas_blocks[k,i,j], orientations=18, pixels_per_cell=(4, 4),cells_per_block=(1, 1),
                    block_norm="L1", visualize=False, feature_vector=True)


eoh_ngas = np.zeros(shape=(nongas_blocks.shape[0],2,2,648))
for k in range (nongas_blocks.shape[0]):
    for i in range (nongas_blocks.shape[1]):
        for j in range (nongas_blocks.shape[2]):
            eoh_ngas[k,i,j] = hog(nongas_blocks[k,i,j], orientations=18, pixels_per_cell=(4, 4),cells_per_block=(1, 1),
                    block_norm="L1", visualize=False, feature_vector=True)

#Create EOH classifier
#Create wavelet energy classifier
samples=[]
labels=[]
for k in range (eoh_gas.shape[0]):
    for i in range (eoh_gas.shape[1]):
        for j in range (eoh_gas.shape[2]):
            samples.append(eoh_gas[k,i,j])
            labels.append(1)
    
for k in range (eoh_ngas.shape[0]):
    for i in range (eoh_ngas.shape[1]):
        for j in range (eoh_ngas.shape[2]):
            samples.append(eoh_ngas[k,i,j])
            labels.append(0)
            
# Convert objects to Numpy Objects
samples = np.float32(samples)
labels = np.array(labels)          

# assign a random permutation
rand = np.random.RandomState(321)
shuffle = rand.permutation(len(samples))
samples = samples[shuffle]
labels = labels[shuffle]    
   
ratio = 0.7
classes = 2

train_set, train_lab, other_set,other_lab = load_data_multi_samples(samples, labels, ratio, classes)
# train SVM classifier
print '... training Linear-SVM'                            
clf_lin_pca = svm.LinearSVC()
clf_lin_pca.fit(train_set, train_lab)  
# validate SVM
print '... testing Linear-SVM'
predictions = clf_lin_pca.predict(other_set) 
# calculate errors
errors = np.where(predictions != other_lab)
lin_pca_er = float(errors[0].shape[0])/other_lab.shape[0]  
print '... Linear-SVM error:%f'%(100*lin_pca_er) 
joblib.dump(clf_lin_pca, 'eoh_svm_classif.pkl')

# train Adaboost classifier
print '... training AdaBoost-Classifier'                            
clf_lin_pca = AdaBoostClassifier(n_estimators=100, random_state=0)
clf_lin_pca.fit(train_set, train_lab)  
# validate SVM
print '... testing AdaBoost-Classifier'
predictions = clf_lin_pca.predict(other_set) 
# calculate errors
errors = np.where(predictions != other_lab)
lin_pca_er = float(errors[0].shape[0])/other_lab.shape[0]  
print '... AdaBoost-Classifier error:%f'%(100*lin_pca_er) 
joblib.dump(clf_lin_pca, 'eoh_Adaboost_classif.pkl')

#Create histogram           
stepx = 18
a=4
b=280
for k in range(eoh_gas.shape[1]):
    for j in range(eoh_gas.shape[2]):
        cg = eoh_gas[a,k,j,:18] + eoh_gas[a,k,j,18:36]
        cng = eoh_ngas[b,k,j,:18] + eoh_ngas[b,k,j,18:36]

for k in range(eoh_gas.shape[1]):
    for j in range(eoh_gas.shape[2]):
        for i in range(36,eoh_gas.shape[3],stepx):
            cg = cg + eoh_gas[a,k,j,i:i+stepx]    
            cng = cng + eoh_ngas[b,k,j,i:i+stepx]    
            
fig = plt.figure()
plt.suptitle(r'Edge Orientation Histogram',fontsize="x-large")
plt.subplots_adjust(hspace=0.7)
plt.subplot(2, 2, 1)
plt.xlabel('Edge Orientation (degrees)')
plt.ylabel('Magnituted-weighted Occurance')
plt.ylim(0,60)
plt.grid()
ydata=cg 
xdata = range(len(ydata))
bucket_names = np.tile(np.arange(18)+1, 1 * 1)      
plt.bar(xdata, ydata, align='center', alpha=0.8, width=0.9)
plt.xticks(xdata, bucket_names * 10, rotation=90)
plt.subplot(2, 2, 2)
plt.xlabel('Edge Orientation (degrees)')
plt.ylabel('Magnituted-weighted Occurance')
plt.ylim(0,60)
plt.grid()
ydatang=cng 
xdatang = range(len(ydatang))
bucket_namesng = np.tile(np.arange(18)+1, 1 * 1)      
plt.bar(xdatang, ydatang, align='center', alpha=0.8, width=0.9)
plt.xticks(xdatang, bucket_namesng * 10, rotation=90)
plt.subplot(2, 2, 3)
plt.imshow(gas_img[a], cmap='gray')
plt.xticks([])
plt.yticks([])
plt.subplot(2, 2, 4)
plt.imshow(nongas_img[b], cmap='gray')
plt.xticks([])
plt.yticks([])

   
    