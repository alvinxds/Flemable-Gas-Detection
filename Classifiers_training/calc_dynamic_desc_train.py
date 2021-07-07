# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 12:03:09 2019

@author: anton
"""

######################Dynamic descriptors    
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
from sklearn import svm
from sklearn.externals import joblib
from sklearn.ensemble import AdaBoostClassifier

#Change current working directory
imagePath = "C:/Users/anton/Documents/Emp/Thesis/Python_code/gas_detect/train_classifier/raw_dataset/patches/"
os.chdir(imagePath)

rcParams['figure.figsize'] = 6, 4


#Load dynamic dataset
#First load positive 'gas' patches
gas_vid = [skvideo.io.vread(file,outputdict={"-pix_fmt": "gray"})[:, :, :, 0] for file in glob.glob("./dynamic_dataset/gas/*.mp4")]

for k in range(len(gas_vid)):
    if(len(gas_vid[k]) != len(gas_vid[0])):
        del gas_vid[k]
    else:
        print 'all ok'
   
#Then load negative 'non-gas' patches
nongas_vid = [skvideo.io.vread(file,outputdict={"-pix_fmt": "gray"})[:, :, :, 0] for file in glob.glob("./dynamic_dataset/nongas/*.mp4")]

for k in range(len(nongas_vid)):
    if(len(nongas_vid[k]) != len(nongas_vid[0])):
        del nongas_vid[k]
    else:
        print 'all ok'
        

#Gas blocks
stepx = 24
stepy = 24
lists = []

for p in range(len(gas_vid)):
    for k in range (gas_vid[0].shape[0]):
        for i in range(0,gas_vid[0].shape[1],stepy):
            for j in range(0,gas_vid[0].shape[2],stepx):
                lists.append(gas_vid[p][k,i:i+stepy,j:j+stepx])
        
myarray = np.asarray(lists)
gas_vid_cells = myarray.reshape(len(gas_vid),gas_vid[0].shape[0],2,2,stepy,stepx)

#NonGas blocks
stepx = 24
stepy = 24
lists = []

for p in range(len(nongas_vid)):
    for k in range (nongas_vid[0].shape[0]):
        for i in range(0,nongas_vid[0].shape[1],stepy):
            for j in range(0,nongas_vid[0].shape[2],stepx):
                lists.append(nongas_vid[p][k,i:i+stepy,j:j+stepx])
        
myarray = np.asarray(lists)
nongas_vid_cells = myarray.reshape(len(nongas_vid),nongas_vid[0].shape[0],2,2,stepy,stepx)

#################Calculate LBP-Three Orthogonal Planes
radius = 1
n_points = 8 * radius
method = 'nri_uniform'

#LBPTOP for gas cuboics
lbp_gxy = np.zeros(shape=(gas_vid_cells.shape[0],2,2,24,24))
lbp_gxt = np.zeros(shape=(gas_vid_cells.shape[0],2,2,11,24))
lbp_gyt = np.zeros(shape=(gas_vid_cells.shape[0],2,2,11,24))


for k in range (gas_vid_cells.shape[0]):
    for i in range (gas_vid_cells.shape[2]):
        for j in range(gas_vid_cells.shape[3]):
            lbp_gxy[k,i,j] = local_binary_pattern(gas_vid_cells[k,4,i,j,:,:], n_points, radius, method)
            lbp_gxt[k,i,j] = local_binary_pattern(gas_vid_cells[k,:,i,j,11,:], n_points, radius, method)
            lbp_gyt[k,i,j] = local_binary_pattern(gas_vid_cells[k,:,i,j,:,11], n_points, radius, method)

ls_gxy=[]
ls_gxt=[]
ls_gyt=[]
new_index = range(0, 59)

for k in range (lbp_gxy.shape[0]):
    for i in range (lbp_gxy.shape[1]):
        for j in range (lbp_gxy.shape[2]):
            ls_gxy.append(lbp_gxy[k,i,j])
            ls_gxt.append(lbp_gxt[k,i,j])
            ls_gyt.append(lbp_gyt[k,i,j])

lbptops=[]            
for k in range (len(ls_gxy)):
    ls_gxy[k] = pd.DataFrame(ls_gxy[k])
    ls_gxy[k] = pd.melt(ls_gxy[k])
    ls_gxy[k] = ls_gxy[k].groupby(['value']).count()
    ls_gxy[k] = ls_gxy[k].reindex(new_index,fill_value=0)  
    ls_gxy[k] = pd.concat([ls_gxy[k],((ls_gxy[k]/ls_gxy[k].sum())*100)/100],axis=1)
    ls_gxy[k].columns=['Count','Normalized Occurrence']
    ls_gxy[k].reset_index(level=0, inplace=True)
    
    ls_gxt[k] = pd.DataFrame(ls_gxt[k])
    ls_gxt[k] = pd.melt(ls_gxt[k])
    ls_gxt[k] = ls_gxt[k].groupby(['value']).count()
    ls_gxt[k] = ls_gxt[k].reindex(new_index,fill_value=0)  
    ls_gxt[k] = pd.concat([ls_gxt[k],((ls_gxt[k]/ls_gxt[k].sum())*100)/100],axis=1)
    ls_gxt[k].columns=['Count','Normalized Occurrence']
    ls_gxt[k].reset_index(level=0, inplace=True)
    
    ls_gyt[k] = pd.DataFrame(ls_gyt[k])
    ls_gyt[k] = pd.melt(ls_gyt[k])
    ls_gyt[k] = ls_gyt[k].groupby(['value']).count()
    ls_gyt[k] = ls_gyt[k].reindex(new_index,fill_value=0)  
    ls_gyt[k] = pd.concat([ls_gyt[k],((ls_gyt[k]/ls_gyt[k].sum())*100)/100],axis=1)
    ls_gyt[k].columns=['Count','Normalized Occurrence']
    ls_gyt[k].reset_index(level=0, inplace=True)

lbptop1 = np.zeros(shape=(len(ls_gxt),ls_gxt[0].shape[0]*2,ls_gxt[0].shape[1]))
lbptop = np.zeros(shape=(len(ls_gxt),ls_gxt[0].shape[0]*3,ls_gxt[0].shape[1]))

for k in range (len(ls_gxy)):    
    lbptop1[k] = ls_gxy[k].append(ls_gxt[k])
    lbptop[k] = np.vstack((lbptop1[k],ls_gyt[k]))
    
lbptop2=[]
for k in range (lbptop.shape[0]):  
    lbptop2.append(lbptop[k])
    lbptop2[k] = pd.DataFrame(lbptop2[k])

lbptop_g = list(lbptop2)
new_index = range(0, 177)
ni = pd.DataFrame(new_index)
for k in range (len(lbptop_g)):
    lbptop_g[k].columns=['values','Count','Normalized Occurrence']
    lbptop_g[k] = pd.concat([lbptop_g[k],ni],axis=1)
    lbptop_g[k] = lbptop_g[k].drop(['values'], axis=1)
    lbptop_g[k].columns=['Count','Normalized Occurrence','Values']
    lbptop_g[k]['Values'] = lbptop_g[k]['Values']+1
    lbptop_g[k] = lbptop_g[k].values
    lbptop_g[k] = lbptop_g[k][:,1]
    #lbptop_g[k] = lbptop_g[k].reshape(1,lbptop_g[k].shape[0])
    

#Plot them
plt.figure()
plt.bar(range(len(lbptop_g[10])), lbptop_g[10]["Count"], color='blue')
plt.xticks(np.arange(0, 180, step=30))

#LBPTOP for nongas cuboics
lbp_ngxy = np.zeros(shape=(nongas_vid_cells.shape[0],2,2,24,24))
lbp_ngxt = np.zeros(shape=(nongas_vid_cells.shape[0],2,2,11,24))
lbp_ngyt = np.zeros(shape=(nongas_vid_cells.shape[0],2,2,11,24))


for k in range (nongas_vid_cells.shape[0]):
    for i in range (nongas_vid_cells.shape[2]):
        for j in range(nongas_vid_cells.shape[3]):
            lbp_ngxy[k,i,j] = local_binary_pattern(nongas_vid_cells[k,4,i,j,:,:], n_points, radius, method)
            lbp_ngxt[k,i,j] = local_binary_pattern(nongas_vid_cells[k,:,i,j,11,:], n_points, radius, method)
            lbp_ngyt[k,i,j] = local_binary_pattern(nongas_vid_cells[k,:,i,j,:,11], n_points, radius, method)

ls_ngxy=[]
ls_ngxt=[]
ls_ngyt=[]
new_index = range(0, 59)

for k in range (lbp_ngxy.shape[0]):
    for i in range (lbp_ngxy.shape[1]):
        for j in range (lbp_ngxy.shape[2]):
            ls_ngxy.append(lbp_ngxy[k,i,j])
            ls_ngxt.append(lbp_ngxt[k,i,j])
            ls_ngyt.append(lbp_ngyt[k,i,j])

lbptops=[]            
for k in range (len(ls_ngxy)):
    ls_ngxy[k] = pd.DataFrame(ls_ngxy[k])
    ls_ngxy[k] = pd.melt(ls_ngxy[k])
    ls_ngxy[k] = ls_ngxy[k].groupby(['value']).count()
    ls_ngxy[k] = ls_ngxy[k].reindex(new_index,fill_value=0)  
    ls_ngxy[k] = pd.concat([ls_ngxy[k],((ls_ngxy[k]/ls_ngxy[k].sum())*100)/100],axis=1)
    ls_ngxy[k].columns=['Count','Normalized Occurrence']
    ls_ngxy[k].reset_index(level=0, inplace=True)
    
    ls_ngxt[k] = pd.DataFrame(ls_ngxt[k])
    ls_ngxt[k] = pd.melt(ls_ngxt[k])
    ls_ngxt[k] = ls_ngxt[k].groupby(['value']).count()
    ls_ngxt[k] = ls_ngxt[k].reindex(new_index,fill_value=0)  
    ls_ngxt[k] = pd.concat([ls_ngxt[k],((ls_ngxt[k]/ls_ngxt[k].sum())*100)/100],axis=1)
    ls_ngxt[k].columns=['Count','Normalized Occurrence']
    ls_ngxt[k].reset_index(level=0, inplace=True)
    
    ls_ngyt[k] = pd.DataFrame(ls_ngyt[k])
    ls_ngyt[k] = pd.melt(ls_ngyt[k])
    ls_ngyt[k] = ls_ngyt[k].groupby(['value']).count()
    ls_ngyt[k] = ls_ngyt[k].reindex(new_index,fill_value=0)  
    ls_ngyt[k] = pd.concat([ls_ngyt[k],((ls_ngyt[k]/ls_ngyt[k].sum())*100)/100],axis=1)
    ls_ngyt[k].columns=['Count','Normalized Occurrence']
    ls_ngyt[k].reset_index(level=0, inplace=True)

lbptop1 = np.zeros(shape=(len(ls_ngxt),ls_ngxt[0].shape[0]*2,ls_ngxt[0].shape[1]))
lbptop = np.zeros(shape=(len(ls_ngxt),ls_ngxt[0].shape[0]*3,ls_ngxt[0].shape[1]))

for k in range (len(ls_ngxy)):    
    lbptop1[k] = ls_ngxy[k].append(ls_ngxt[k])
    lbptop[k] = np.vstack((lbptop1[k],ls_ngyt[k]))
    
lbptop2=[]
for k in range (lbptop.shape[0]):  
    lbptop2.append(lbptop[k])
    lbptop2[k] = pd.DataFrame(lbptop2[k])

lbptop_ng = list(lbptop2)
new_index = range(0, 177)
ni = pd.DataFrame(new_index)
for k in range (len(lbptop_ng)):
    lbptop_ng[k].columns=['values','Count','Normalized Occurrence']
    lbptop_ng[k] = pd.concat([lbptop_ng[k],ni],axis=1)
    lbptop_ng[k] = lbptop_ng[k].drop(['values'], axis=1)
    lbptop_ng[k].columns=['Count','Normalized Occurrence','Values']
    lbptop_ng[k]['Values'] = lbptop_ng[k]['Values']+1
    lbptop_ng[k] = lbptop_ng[k].values
    lbptop_ng[k] = lbptop_ng[k][:,1]
    #lbptop_ng[k] = lbptop_ng[k].reshape(1,lbptop_ng[k].shape[0])


#Plot them
#plt.figure()
#plt.bar(range(len(lbptop_ng[10])), lbptop_ng[10]["Count"], color='blue')
#plt.xticks(np.arange(0, 180, step=30))

#Train lbptop classifier
exec(open("C:/Users/anton/Documents/Emp/Thesis/Python_code/gas_detect/train_classifier/raw_dataset/load_data_multi_samples.py").read())  

#Create lbp-nri classifier
samples=[]
labels=[]
for k in range (len(lbptop_g)):
    samples.append(lbptop_g[k])
    labels.append(1)
for k in range (len(lbptop_ng)):
    samples.append(lbptop_ng[k])
    labels.append(0)
            
# Convert objects to Numpy Objects
samples = np.asarray(samples)
labels = np.array(labels)          

# assign a random permutation
rand = np.random.RandomState(321)
shuffle = rand.permutation(len(samples))
samples = samples[shuffle]
labels = labels[shuffle]    
   
ratio = 0.70
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
joblib.dump(clf_lin_pca, 'lbptop_svm_classif.pkl')

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
joblib.dump(clf_lin_pca, 'lbptop_Adaboost_classif.pkl')

################Calculate HoGHoF

#Histogram of Optical Flow
#For Gas
exec(open("C:/Users/anton/Documents/Emp/Thesis/Python_code/gas_detect/train_classifier/raw_dataset/hof_get_flow.py").read()) 

flow = []

for p in range(gas_vid_cells.shape[0]):
    for k in range(gas_vid_cells.shape[1]-1):
        for i in range(gas_vid_cells.shape[2]):
            for j in range(gas_vid_cells.shape[3]):
                flow.append(getFlow(gas_vid_cells[p,k,i,j],gas_vid_cells[p,k+1,i,j]))

hof_g = []
        
for k in range(len(flow)):
    hof_g.append(hof(flow[k][0], visualise=False, normalise=True))

#For NonGas
flowng = []
    
for p in range(nongas_vid_cells.shape[0]):
    for k in range(nongas_vid_cells.shape[1]-1):
        for i in range(nongas_vid_cells.shape[2]):
            for j in range(nongas_vid_cells.shape[3]):
                flowng.append(getFlow(nongas_vid_cells[p,k,i,j],nongas_vid_cells[p,k+1,i,j]))

hof_ng = []
        
for k in range(len(flowng)):
    hof_ng.append(hof(flowng[k][0], visualise=False, normalise=True))  


hof_g = np.asarray(hof_g)
hof_g = np.reshape(hof_g,(101,10,2,2,81))   

hof_ng = np.asarray(hof_ng)
hof_ng = np.reshape(hof_ng,(160,10,2,2,81))   

#Create histogram
b=gas_vid[2]
plt.imshow(b[0],cmap='gray')     

#Gas    
g_hof = hof_g[2,0] 
g_hof = g_hof[0,0] + g_hof[0,1] + g_hof[1,0] + g_hof[1,1]      

a = []  

a1=np.sum(g_hof[:9])
a2=np.sum(g_hof[10:19])    
a3=np.sum(g_hof[20:29])       
a4=np.sum(g_hof[30:39])       
a5=np.sum(g_hof[40:49])        
a6=np.sum(g_hof[50:59])        
a7=np.sum(g_hof[60:69]) 
a8=np.sum(g_hof[70:80])    

a.extend([a1,a2,a3,a4,a5,a6,a7,a8])
a = np.asarray(a)
y_pos = np.arange(len(a))+1
plt.bar(y_pos, a, align='center', alpha=0.5)

plt.ylabel('Normalized Occurance')
plt.title('Flow Orientation Index Of Gas Patch')
plt.show()



#NonGas 
b=nongas_vid[4]
plt.imshow(b[0],cmap='gray')
   
ng_hof = hof_ng[4,0]
ng_hof = ng_hof[0,0] + ng_hof[0,1] + ng_hof[1,0] + ng_hof[1,1] 

a = []  

a1=np.sum(ng_hof[:9])
a2=np.sum(ng_hof[10:19])    
a3=np.sum(ng_hof[20:29])       
a4=np.sum(ng_hof[30:39])       
a5=np.sum(ng_hof[40:49])        
a6=np.sum(ng_hof[50:59])        
a7=np.sum(ng_hof[60:69]) 
a8=np.sum(ng_hof[70:80])    

a.extend([a1,a2,a3,a4,a5,a6,a7,a8])
a = np.asarray(a)
y_pos = np.arange(len(a))+1
plt.bar(y_pos, a, align='center', alpha=0.5)

plt.ylabel('Normalized Occurance')
plt.title('Flow Orientation Index Of Non-Gas Patch')
plt.show()


    
#Histogram of Oriented Gradients
from skimage.feature import hog

hog_g = []

for p in range(gas_vid_cells.shape[0]):
    for k in range(gas_vid_cells.shape[1]-1):
        for i in range(gas_vid_cells.shape[2]):
            for j in range(gas_vid_cells.shape[3]):
                hog_g.append(hog(gas_vid_cells[p,k,i,j], orientations=9, pixels_per_cell=(8, 8),cells_per_block=(3, 3),block_norm="L2", visualize=False))
    
hog_ng = []

for p in range(nongas_vid_cells.shape[0]):
    for k in range(nongas_vid_cells.shape[1]-1):
        for i in range(nongas_vid_cells.shape[2]):
            for j in range(nongas_vid_cells.shape[3]):
                hog_ng.append(hog(nongas_vid_cells[p,k,i,j], orientations=9, pixels_per_cell=(8, 8),cells_per_block=(3, 3),block_norm="L2", visualize=False))

#For gas      
hoghof = []
for k in range (len(hog_g)):    
    hoghof.append(np.column_stack((hog_g[k],hof_g[k])))

for k in range (len(hoghof)):
    hoghof[k] = pd.DataFrame(hoghof[k])
    hoghof[k].columns=['hog','hof']

hoghofs=np.zeros(shape=(len(hoghof),hoghof[0].shape[0]*2))   

for k in range (len(hoghof)):
    hoghofs[k,:81] = hoghof[k]['hog']
    hoghofs[k,81:] = hoghof[k]['hof']
    
hoghof_g = list(hoghofs)
    
hoghof = []
for k in range (len(hog_ng)):    
    hoghof.append(np.column_stack((hog_ng[k],hof_ng[k])))

for k in range (len(hoghof)):
    hoghof[k] = pd.DataFrame(hoghof[k])
    hoghof[k].columns=['hog','hof']

hoghofs=np.zeros(shape=(len(hoghof),hoghof[0].shape[0]*2))   

for k in range (len(hoghof)):
    hoghofs[k,:81] = hoghof[k]['hog']
    hoghofs[k,81:] = hoghof[k]['hof']
    
hoghof_ng = list(hoghofs)







#Create lbp-nri classifier
samples=[]
labels=[]
for k in range (len(hoghof_g)):
    samples.append(hoghof_g[k])
    labels.append(1)
for k in range (len(hoghof_ng)):
    samples.append(hoghof_ng[k])
    labels.append(0)
            
# Convert objects to Numpy Objects
samples = np.asarray(samples)
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
joblib.dump(clf_lin_pca, 'hoghof_svm_classif.pkl')

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
joblib.dump(clf_lin_pca, 'hoghof_Adaboost_classif.pkl')
