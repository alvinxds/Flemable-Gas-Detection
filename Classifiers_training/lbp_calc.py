# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:07:33 2019

@author: anton
"""
from skimage.feature import local_binary_pattern
import numpy as np
import pandas as pd

def lbp_calc(img, radius, n_points, method):
    lbp = local_binary_pattern(img, n_points, radius, method)
    lbp = pd.DataFrame(lbp)
    lbp = pd.melt(lbp)
    lbp = lbp.groupby(['value']).count()
    
    if (method == 'nri_uniform'):
        new_index = range(0, 59)
        lbp = lbp.reindex(new_index,fill_value=0)
        lbp = pd.concat([lbp,((lbp/lbp.sum())*100)/100],axis=1)
        lbp.columns=['Count','Normalized Occurrence']
        lbp.reset_index(level=0, inplace=True)
        lbp[['value']] = lbp[['value']]+1
        
    if (method == 'uniform'):
        lbp = pd.concat([lbp,((lbp/lbp.sum())*100)/100],axis=1)
        lbp.columns=['Count','Normalized Occurrence']
        lbp.reset_index(level=0, inplace=True)
        lbp[['value']] = lbp[['value']]+1
        
        return lbp