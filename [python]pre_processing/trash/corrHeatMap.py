#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 15:42:51 2021

@author: yutasuzuki
"""

import numpy as np
import matplotlib.pyplot as plt
from pre_processing import pre_processing,re_sampling
from rejectBlink_PCA import rejectBlink_PCA
import json
import os
from pixel_size import pixel_size
import warnings
import matplotlib.patches as patches
import pandas as pd

warnings.simplefilter('ignore')

## ########## initial settings ###################
saveFileLocs = '../data/'

f = open(os.path.join(saveFileLocs + str('data2.json')))
dat = json.load(f)
f.close()

df = pd.DataFrame()
df = {'PDR':[],'sub':[],'condition':[]}

y = np.array(dat['PDR'])
pupil = []
mmCondition = ["Bottom","Center","Left","Right","Top"]

for iSub in np.unique(dat['sub']):
    
    for iCond in np.arange(1,6):
        ##### Glare #####
        ind_glare = np.argwhere((np.array(dat['sub']) == np.int64(iSub)) &
                      (np.array(dat['condition']) == iCond)).reshape(-1)
        ##### Control #####
        ind_control = np.argwhere((np.array(dat['sub']) == np.int64(iSub)) &
                      (np.array(dat['condition']) == iCond+5)).reshape(-1)
        
        
        df['PDR'] = np.r_[df['PDR'],np.mean( y[ind_control,500:].mean(axis=0) - y[ind_glare,500:].mean(axis=0))]
        df['sub'] = np.r_[df['sub'],iSub]
        df['condition'] = np.r_[df['condition'],int(iCond)]


df_plot = pd.DataFrame()
df_plot['PDR'] = df['PDR']
df_plot['sub'] = df['sub']
df_plot['condition'] = [str(mmCondition[int(mm-1)]) for mm in df['condition']]


center = df_plot[df_plot['condition'] == 'Center']['PDR']
df_plot['center'] = np.array(np.repeat(center, list(np.ones(len(center))*5)))

df_plot['GPI'] = df_plot['PDR'] / df_plot['center']

# df_plot = df_plot[df_plot['sub'] != 10]

# numOfsub = len(np.unique(dat['sub']))
# plt.figure(figsize=(8, 6))
# plt.rcParams["font.size"] = 18
# plt.errorbar(mmCondition, df_plot.groupby(["condition"]).mean()['GPI'], 
#              yerr = df_plot.groupby(["condition"]).std()['GPI']/np.sqrt(numOfsub), 
#              capsize=5, fmt='o', markersize=10, ecolor='black', 
#              markeredgecolor = "black", 
#              color='w')

plt.figure(figsize=(8, 6)) 
plt.rcParams["font.size"] = 18
plt.plot(df_plot[df_plot['condition'] == 'Center']['center'],df_plot[df_plot['condition'] == 'Top']['PDR'],'o')
plt.plot(df_plot[df_plot['condition'] == 'Center']['center'],df_plot[df_plot['condition'] == 'Bottom']['PDR'],'o')

from scipy.stats import pearsonr
x = df_plot[df_plot['condition'] == 'Center']['center']
y = df_plot[df_plot['condition'] == 'Top']['PDR']

# ndarray を 1次元配列に変換して渡す
a, b = pearsonr(np.ravel(x), np.ravel(y))
print("相関係数:", a)
print("p値:", b)